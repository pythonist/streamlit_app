import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from alert_engine import AlertEngine
from config import CONFIG
from data_ingestion import DataIngestion
from entity_resolution import EntityResolution
from feature_engineering import FeatureEngineering
from feedback_loop import FeedbackLoop
from graph_analytics import GraphAnalytics
from multiclass_model import MulticlassModel
from sequence_models import SequenceModels
from utils import print_step


def champion_challenger_and_kpi(test_df, y_test, test_pred, test_pred_ch, test_prob):
    if test_df is None or len(test_df) == 0 or test_prob is None or len(test_prob) == 0:
        empty_comparison = pd.DataFrame(columns=["model", "accuracy", "macro_f1", "weighted_f1"])
        empty_kpi = pd.DataFrame(columns=["metric", "before", "after", "delta"])
        return empty_comparison, empty_kpi

    comparison = pd.DataFrame(
        {
            "model": ["champion_calibrated", "challenger_logistic"],
            "accuracy": [accuracy_score(y_test, test_pred), accuracy_score(y_test, test_pred_ch)],
            "macro_f1": [f1_score(y_test, test_pred, average="macro"), f1_score(y_test, test_pred_ch, average="macro")],
            "weighted_f1": [f1_score(y_test, test_pred, average="weighted"), f1_score(y_test, test_pred_ch, average="weighted")],
        }
    )
    test_df = test_df.copy()
    test_df["before_score"] = test_prob.max(axis=1)
    test_df["after_score"] = test_df.get("final_mule_score", test_df["before_score"])
    before_high = int((test_df["before_score"] >= 0.7).sum())
    after_high = int((test_df["after_score"] >= 0.7).sum())
    before_precision = test_df.loc[test_df["before_score"] >= 0.7, "label"].isin(CONFIG["risky_labels"]).mean() if before_high > 0 else 0.0
    after_precision = test_df.loc[test_df["after_score"] >= 0.7, "label"].isin(CONFIG["risky_labels"]).mean() if after_high > 0 else 0.0
    kpi_table = pd.DataFrame(
        {
            "metric": ["high_risk_volume", "precision_proxy"],
            "before": [before_high, round(float(before_precision), 4)],
            "after": [after_high, round(float(after_precision), 4)],
        }
    )
    kpi_table["delta"] = kpi_table["after"] - kpi_table["before"]
    return comparison, kpi_table


def run_timed(step_name, timings, callback):
    start = time.perf_counter()
    result = callback()
    timings[step_name] = round(time.perf_counter() - start, 4)
    return result


def build_manifest(outputs, timings):
    raw_tables = outputs.get("raw_tables", {})
    txn_tables = outputs.get("txn_tables", {})
    feature_df = outputs.get("feature_df")
    graph_features = outputs.get("graph_features")
    ring_df = outputs.get("ring_df")
    alert_output = outputs.get("alert_output")
    model6_artifacts = outputs.get("model6_artifacts")

    manifest = {
        "run_profile": CONFIG.get("run_profile", "Standard"),
        "random_state": CONFIG.get("random_state", 42),
        "stage_timings_seconds": timings,
        "table_counts": {
            "raw_tables": len(raw_tables),
            "txn_tables": len(txn_tables),
            "feature_rows": int(len(feature_df)) if isinstance(feature_df, pd.DataFrame) else 0,
            "graph_nodes": int(len(graph_features)) if isinstance(graph_features, pd.DataFrame) else 0,
            "ring_candidates": int(len(ring_df)) if isinstance(ring_df, pd.DataFrame) else 0,
            "alerts": int(len(alert_output)) if isinstance(alert_output, pd.DataFrame) else 0,
        },
        "model_summary": {
            "class_names": list(model6_artifacts.label_encoder.classes_) if model6_artifacts else [],
            "feature_count": len(outputs.get("feature_cols", [])),
            "calibration_method": CONFIG.get("calibration_method", "sigmoid"),
            "champion_model": CONFIG.get("classifier_model", "Random Forest"),
            "challenger_model": CONFIG.get("challenger_model", "Logistic Regression"),
        },
        "threshold_summary": {
            "selected_threshold": outputs.get("threshold_opt", {}).get("selected_threshold") if outputs.get("threshold_opt") else None,
            "avg_daily_alerts": outputs.get("threshold_opt", {}).get("avg_daily_alerts") if outputs.get("threshold_opt") else None,
        },
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    return manifest


def save_manifest(manifest, path="pipeline_manifest.json"):
    manifest_path = Path(path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest_path


def run_pipeline():
    print_step("FINAL PRODUCTION PACKAGE PIPELINE START")
    timings = {}
    outputs = {}

    np.random.seed(CONFIG["random_state"])

    ingestion = DataIngestion()
    raw_tables = run_timed("Raw Tables", timings, ingestion.generate_raw_tables)
    txn_tables = run_timed("Txn Tables", timings, lambda: ingestion.generate_transaction_tables(raw_tables))
    outputs["raw_tables"] = raw_tables
    outputs["txn_tables"] = txn_tables

    entity = EntityResolution()
    entity_views = run_timed("Entity Views", timings, lambda: entity.build_entity_views(raw_tables))
    events = run_timed("Unified Events", timings, lambda: entity.build_unified_events(txn_tables))
    single_view = run_timed("Single View", timings, lambda: entity.build_single_view(events, entity_views))
    outputs["entity_views"] = entity_views
    outputs["events"] = events
    outputs["single_view"] = single_view

    feat = FeatureEngineering()
    clean_df = run_timed("EDA", timings, lambda: feat.run_eda_and_imputation(single_view))
    feature_df = run_timed("Features", timings, lambda: feat.feature_engineering(clean_df))
    outputs["feature_df"] = feature_df

    graph = GraphAnalytics()
    feature_df, graph_features = run_timed("Graph", timings, lambda: graph.model1_graph_analytics(feature_df))
    feature_df, ring_df = run_timed("Rings", timings, lambda: graph.model2_ring_detection(feature_df))
    outputs["feature_df"] = feature_df
    outputs["graph_features"] = graph_features
    outputs["ring_df"] = ring_df

    modeler = MulticlassModel()
    train_df, valid_df, test_df = run_timed(
        "Split",
        timings,
        lambda: modeler.split_time_based(feature_df, CONFIG["train_end"], CONFIG["valid_end"]),
    )
    outputs["train_df"] = train_df
    outputs["valid_df"] = valid_df
    outputs["test_df"] = test_df

    seq = SequenceModels()
    model3, train_df, valid_df, test_df = run_timed("Hazard", timings, lambda: seq.model3_hazard(train_df, valid_df, test_df))
    model4, train_df, valid_df, test_df = run_timed("HMM", timings, lambda: seq.model4_hmm(train_df, valid_df, test_df))
    model5_outputs = run_timed("Sequence", timings, lambda: seq.model5_lstm_and_transformer(train_df, valid_df, test_df))
    outputs["model3_hazard"] = model3
    outputs["model4_hmm"] = model4
    outputs["model5_outputs"] = model5_outputs
    outputs["train_df"] = train_df
    outputs["valid_df"] = valid_df
    outputs["test_df"] = test_df

    model_results = run_timed(
        "Classifier",
        timings,
        lambda: modeler.model6_multiclass(train_df, valid_df, test_df),
    )
    (
        model6_artifacts,
        feature_cols,
        y_valid,
        y_test,
        valid_prob,
        test_prob,
        valid_pred,
        test_pred,
        valid_pred_ch,
        test_pred_ch,
        feature_importance,
    ) = model_results
    outputs["model6_artifacts"] = model6_artifacts
    outputs["feature_cols"] = feature_cols
    outputs["y_valid"] = y_valid
    outputs["y_test"] = y_test
    outputs["valid_prob"] = valid_prob
    outputs["test_prob"] = test_prob
    outputs["valid_pred"] = valid_pred
    outputs["test_pred"] = test_pred
    outputs["valid_pred_ch"] = valid_pred_ch
    outputs["test_pred_ch"] = test_pred_ch
    outputs["feature_importance"] = feature_importance

    alert_engine = AlertEngine()
    test_df = run_timed(
        "Decision",
        timings,
        lambda: alert_engine.model7_decision_engine(test_df, test_prob, model6_artifacts, model5_outputs),
    )
    alert_output, threshold_tbl, channel_thresholds_tbl, class_thresholds_tbl, enriched_alert_df, th_opt = run_timed(
        "Alert Pack",
        timings,
        lambda: alert_engine.model8_alert_pack(test_df, test_prob, model6_artifacts),
    )
    outputs["test_df"] = test_df
    outputs["alert_output"] = alert_output
    outputs["threshold_table"] = threshold_tbl
    outputs["channel_thresholds"] = channel_thresholds_tbl
    outputs["class_thresholds"] = class_thresholds_tbl
    outputs["enriched_alert_df"] = enriched_alert_df
    outputs["threshold_opt"] = th_opt

    feedback = FeedbackLoop()
    feedback_outputs = run_timed("Feedback", timings, lambda: feedback.weak_supervision_and_feedback(feature_df, alert_output))
    outputs["feedback_outputs"] = feedback_outputs

    comparison_tbl, kpi_tbl = champion_challenger_and_kpi(test_df, y_test, test_pred, test_pred_ch, test_prob)
    outputs["comparison_tbl"] = comparison_tbl
    outputs["kpi_tbl"] = kpi_tbl

    governance_summary = {
        "model1_graph_analytics": True,
        "model2_ring_detection": True,
        "model3_hazard_model": True,
        "model4_hmm": True,
        "model5_lstm_transformer_ready": True,
        "model6_multiclass_typology": True,
        "model7_decision_engine": True,
        "model8_alert_pack": True,
        "calibration": CONFIG.get("calibration_method", "sigmoid"),
        "time_based_validation": True,
        "feature_count": len(feature_cols),
        "classes": list(model6_artifacts.label_encoder.classes_) if model6_artifacts else [],
        "step_timings_seconds": timings,
    }
    outputs["governance_summary"] = governance_summary

    manifest = build_manifest(outputs, timings)
    outputs["pipeline_manifest"] = manifest
    manifest_path = save_manifest(manifest)
    outputs["manifest_path"] = str(manifest_path)
    print(json.dumps(governance_summary, indent=2))
    print(f"Manifest written to: {manifest_path}")
    return outputs


def main():
    outputs = run_pipeline()
    print("Pipeline complete.")
    return outputs


if __name__ == "__main__":
    main()
