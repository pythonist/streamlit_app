import numpy as np
import pandas as pd

from config import CONFIG
from utils import assign_priority_band, print_step, safe_percentile_rank


class AlertEngine:
    def _daily_counts(self, df, score_col, threshold):
        if df.empty or score_col not in df.columns or "event_ts" not in df.columns:
            return pd.Series(dtype=float)
        event_dates = pd.to_datetime(df["event_ts"], errors="coerce").dt.date
        flagged = df[score_col] >= threshold
        return df.assign(flagged=flagged).groupby(event_dates)["flagged"].sum()

    def _threshold_metrics(self, df, score_col, thresholds):
        if df.empty or score_col not in df.columns:
            return pd.DataFrame(columns=["threshold", "avg_daily_alerts", "precision_proxy", "flagged_volume"])

        thresholds = np.round(np.asarray(list(thresholds), dtype=float), 4)
        if thresholds.size == 0:
            return pd.DataFrame(columns=["threshold", "avg_daily_alerts", "precision_proxy", "flagged_volume"])

        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(-np.inf).to_numpy()
        flagged_matrix = scores[:, None] >= thresholds[None, :]
        flagged_volume = flagged_matrix.sum(axis=0).astype(int)

        precision_proxy = np.zeros(len(thresholds), dtype=float)
        if "label" in df.columns:
            risky_mask = df["label"].astype(str).isin(CONFIG["risky_labels"]).to_numpy(dtype=bool)
            risky_hits = (flagged_matrix & risky_mask[:, None]).sum(axis=0)
            precision_proxy = np.divide(
                risky_hits,
                flagged_volume,
                out=np.zeros(len(thresholds), dtype=float),
                where=flagged_volume > 0,
            )

        avg_daily = np.zeros(len(thresholds), dtype=float)
        if "event_ts" in df.columns:
            event_dates = pd.to_datetime(df["event_ts"], errors="coerce").dt.floor("D")
            valid_dates = event_dates.notna().to_numpy()
            if valid_dates.any():
                date_codes = pd.factorize(event_dates[valid_dates], sort=False)[0]
                valid_flagged = flagged_matrix[valid_dates]
                counts_by_day = np.zeros((date_codes.max() + 1, len(thresholds)), dtype=np.int32)
                for idx in range(len(thresholds)):
                    np.add.at(counts_by_day[:, idx], date_codes, valid_flagged[:, idx].astype(np.int32))
                avg_daily = counts_by_day.mean(axis=0) if len(counts_by_day) else avg_daily

        return pd.DataFrame(
            {
                "threshold": thresholds,
                "avg_daily_alerts": np.round(avg_daily, 2),
                "precision_proxy": np.round(precision_proxy, 4),
                "flagged_volume": flagged_volume,
            }
        )

    def model7_decision_engine(self, test_df, test_prob, model6_artifacts, model5_outputs=None):
        print_step("STEP 15: MODEL7 DECISION ENGINE")
        out = test_df.copy()
        if out.empty:
            return out

        out["model_top_prob"] = np.max(test_prob, axis=1) if len(test_prob) else np.zeros(len(out))
        for c, default in [
            ("hazard_score", 0.0),
            ("cust_graph_pagerank", 0.0),
            ("cust_graph_cycle_flag", 0),
            ("hmm_sequence_anomaly_score", 0.0),
            ("ring_max_risk_score", 0.0),
            ("behavioral_risk_score", 0.0),
            ("sequence_score", 0.0),
        ]:
            if c not in out.columns:
                out[c] = default

        out["lstm_emerging_prob"] = 0.0
        if model5_outputs is not None and len(model5_outputs.get("test_ids", [])) > 0:
            test_lstm_map = dict(zip(model5_outputs["test_ids"], model5_outputs["test_lstm_prob"]))
            out["lstm_emerging_prob"] = out["customer_id"].map(test_lstm_map).fillna(0)

        out["graph_pagerank_rank"] = safe_percentile_rank(out["cust_graph_pagerank"])
        out["hmm_rank"] = safe_percentile_rank(out["hmm_sequence_anomaly_score"])
        out["ring_rank"] = safe_percentile_rank(out["ring_max_risk_score"])
        out["behavioral_rank"] = safe_percentile_rank(out["behavioral_risk_score"])

        out["score_component_model"] = out["model_top_prob"].fillna(0)
        out["score_component_sequence"] = out["sequence_score"].fillna(0)
        out["score_component_behavioral"] = out["behavioral_risk_score"].fillna(0)
        out["score_component_hazard"] = out["hazard_score"].fillna(0)
        out["score_component_sequence_model"] = out["lstm_emerging_prob"].fillna(0)
        out["score_component_hmm"] = out["hmm_rank"].fillna(0)
        out["score_component_ring"] = out["ring_rank"].fillna(0)
        out["score_component_graph"] = out["graph_pagerank_rank"].fillna(0)
        out["score_component_cycle"] = out["cust_graph_cycle_flag"].fillna(0)

        raw_score = (
            0.25 * out["score_component_model"]
            + 0.12 * out["score_component_sequence"]
            + 0.13 * out["score_component_behavioral"]
            + 0.15 * out["score_component_hazard"]
            + 0.10 * out["score_component_sequence_model"]
            + 0.10 * out["score_component_hmm"]
            + 0.10 * out["score_component_ring"]
            + 0.03 * out["score_component_graph"]
            + 0.02 * out["score_component_cycle"]
        )
        out["final_mule_score"] = raw_score.clip(0, 0.99)
        out["final_mule_score_rank"] = safe_percentile_rank(out["final_mule_score"])

        pred_labels = model6_artifacts.label_encoder.inverse_transform(np.argmax(test_prob, axis=1)) if len(test_prob) else np.array([])
        if len(pred_labels):
            out["primary_category"] = pred_labels
        else:
            out["primary_category"] = "unknown"

        first_time_counterparty = out["first_time_counterparty"] if "first_time_counterparty" in out.columns else pd.Series(0, index=out.index)
        fanout_flag = out["fanout_flag"] if "fanout_flag" in out.columns else pd.Series(0, index=out.index)
        dormant_activation_flag = out["dormant_activation_flag"] if "dormant_activation_flag" in out.columns else pd.Series(0, index=out.index)
        channel_series = out["channel"] if "channel" in out.columns else pd.Series("", index=out.index)
        transaction_type = out["transaction_type"] if "transaction_type" in out.columns else pd.Series("", index=out.index)

        primary_category = out["primary_category"].astype(str).copy()
        primary_category.loc[out["behavioral_risk_score"].fillna(0) >= 0.65] = "layering_mule"
        primary_category.loc[
            (out["lstm_emerging_prob"].fillna(0) >= 0.7) & (first_time_counterparty.fillna(0) == 1)
        ] = "first_time_mule"
        primary_category.loc[
            channel_series.astype(str).isin(["ATM", "BRANCH"])
            & (transaction_type.astype(str).str.upper() == "CASHOUT")
        ] = "cashout_mule"
        primary_category.loc[fanout_flag.fillna(0) == 1] = "fanout_mule"
        primary_category.loc[
            (dormant_activation_flag.fillna(0) == 1) & (out["sequence_score"].fillna(0) >= 0.55)
        ] = "sleeper_mule"
        out["primary_category"] = primary_category
        return out

    def build_reasoning(self, df):
        out = df.reset_index(drop=True).copy()
        if out.empty:
            return out

        n_rows = len(out)
        primary_reason = np.array([""] * n_rows, dtype=object)
        primary_var = np.array([""] * n_rows, dtype=object)
        secondary_reason = np.array([""] * n_rows, dtype=object)
        secondary_var = np.array([""] * n_rows, dtype=object)
        secondary_cat = np.array([""] * n_rows, dtype=object)
        tertiary_reason = np.array([""] * n_rows, dtype=object)
        tertiary_var = np.array([""] * n_rows, dtype=object)
        tertiary_cat = np.array([""] * n_rows, dtype=object)

        def series_or_default(column_name, default=0):
            return out[column_name] if column_name in out.columns else pd.Series(default, index=out.index)

        reason_specs = [
            (series_or_default("first_time_counterparty").fillna(0) == 1, "First-time counterparty observed", "first_time_counterparty", "counterparty_risk"),
            (series_or_default("first_time_device").fillna(0) == 1, "First-time device used", "first_time_device", "device_risk"),
            (series_or_default("shared_device_risk").fillna(0) == 1, "Shared device across customers", "shared_device_customer_count", "device_risk"),
            (series_or_default("shared_ip_risk").fillna(0) == 1, "Shared IP across customers", "shared_ip_customer_count", "network_risk"),
            (series_or_default("fanout_flag").fillna(0) == 1, "Rapid fan-out behavior", "daily_unique_counterparties", "behavioral_risk"),
            (series_or_default("velocity_flag").fillna(0) == 1, "High transaction velocity", "daily_txn_count", "behavioral_risk"),
            (series_or_default("dormant_activation_flag").fillna(0) == 1, "Dormant account activation", "dormancy_days", "sequence_risk"),
            (pd.to_numeric(series_or_default("transition_score"), errors="coerce").fillna(0) > 0.6, "Unusual transition path", "transition_score", "sequence_risk"),
            (pd.to_numeric(series_or_default("cust_graph_pagerank"), errors="coerce").fillna(0) > 0.001, "High graph centrality", "cust_graph_pagerank", "network_risk"),
            (pd.to_numeric(series_or_default("ring_count"), errors="coerce").fillna(0) > 0, "Account linked to ring or cycle", "ring_count", "network_risk"),
            (pd.to_numeric(series_or_default("behavioral_risk_score"), errors="coerce").fillna(0) >= 0.6, "Elevated behavioural risk pattern", "behavioral_risk_score", "behavioral_risk"),
        ]

        for mask, reason, variable, category in reason_specs:
            active = np.asarray(mask, dtype=bool)

            assign_primary = active & (primary_reason == "")
            primary_reason[assign_primary] = reason
            primary_var[assign_primary] = variable

            remaining = active & ~assign_primary
            assign_secondary = remaining & (secondary_reason == "")
            secondary_reason[assign_secondary] = reason
            secondary_var[assign_secondary] = variable
            secondary_cat[assign_secondary] = category

            remaining = remaining & ~assign_secondary
            assign_tertiary = remaining & (tertiary_reason == "")
            tertiary_reason[assign_tertiary] = reason
            tertiary_var[assign_tertiary] = variable
            tertiary_cat[assign_tertiary] = category

        no_reason = primary_reason == ""
        primary_reason[no_reason] = "Model-based typology assignment"
        primary_var[no_reason] = "model_top_prob"
        secondary_cat[secondary_reason == ""] = ""
        tertiary_cat[tertiary_reason == ""] = ""

        out["primary_reason"] = primary_reason
        out["primary_variables"] = primary_var
        out["secondary_category"] = secondary_cat
        out["secondary_reason"] = secondary_reason
        out["secondary_variables"] = secondary_var
        out["tertiary_category"] = tertiary_cat
        out["tertiary_reason"] = tertiary_reason
        out["tertiary_variables"] = tertiary_var

        out["reasons"] = (
            pd.Series(primary_reason, index=out.index).astype(str)
            + np.where(secondary_reason != "", "; " + secondary_reason.astype(str), "")
            + np.where(tertiary_reason != "", "; " + tertiary_reason.astype(str), "")
        )
        return out

    def threshold_table(self, df, score_col, daily_capacity):
        if df.empty or score_col not in df.columns:
            return pd.DataFrame(columns=["threshold", "avg_daily_alerts", "within_capacity", "precision_proxy", "flagged_volume"])

        thresholds = np.arange(0.1, 0.96, 0.05)
        summary = self._threshold_metrics(df, score_col, thresholds)
        summary["within_capacity"] = (summary["avg_daily_alerts"] <= daily_capacity).astype(int)
        return summary[["threshold", "avg_daily_alerts", "within_capacity", "precision_proxy", "flagged_volume"]]

    def optimize_threshold_by_capacity(self, df, score_col, daily_capacity):
        if df.empty or score_col not in df.columns:
            return {"selected_threshold": None, "avg_daily_alerts": None}

        thresholds = np.arange(0.1, 0.96, 0.02)
        summary = self._threshold_metrics(df, score_col, thresholds)
        candidates = summary[summary["avg_daily_alerts"] <= daily_capacity]
        if candidates.empty:
            return {"selected_threshold": None, "avg_daily_alerts": None}
        selected = candidates.sort_values("threshold", ascending=False).iloc[0]
        return {"selected_threshold": round(float(selected["threshold"]), 2), "avg_daily_alerts": round(float(selected["avg_daily_alerts"]), 2)}

    def channel_specific_thresholds(self, df, score_col, daily_capacity):
        if df.empty or score_col not in df.columns or "channel" not in df.columns:
            return pd.DataFrame(columns=["channel", "selected_threshold", "avg_daily_alerts"])
        rows = []
        alloc = max(1, daily_capacity // max(1, df["channel"].nunique()))
        for ch, part in df.groupby("channel"):
            opt = self.optimize_threshold_by_capacity(part, score_col, alloc)
            rows.append({"channel": ch, **opt})
        return pd.DataFrame(rows)

    def class_specific_thresholds(self, df, prob_cols, daily_capacity):
        if df.empty or not prob_cols:
            return pd.DataFrame(columns=["probability_column", "selected_threshold", "avg_daily_alerts"])
        rows = []
        alloc = max(1, daily_capacity // max(1, len(prob_cols)))
        for col in prob_cols:
            summary = self._threshold_metrics(df, col, np.arange(0.1, 0.96, 0.02))
            candidates = summary[summary["avg_daily_alerts"] <= alloc]
            found = candidates.sort_values("threshold", ascending=False).iloc[0] if not candidates.empty else None
            rows.append(
                {
                    "probability_column": col,
                    "selected_threshold": round(float(found["threshold"]), 2) if found is not None else None,
                    "avg_daily_alerts": round(float(found["avg_daily_alerts"]), 2) if found is not None else None,
                }
            )
        return pd.DataFrame(rows)

    def model8_alert_pack(self, test_df, test_prob, model6_artifacts):
        print_step("STEP 16: MODEL8 ALERT PACK")
        out = self.build_reasoning(test_df.copy())
        for i, cls in enumerate(model6_artifacts.label_encoder.classes_):
            out[f"prob_{cls}"] = test_prob[:, i] if len(test_prob) else 0.0
        channel_exposure_map = out.groupby("customer_id")["channel"].agg(
            lambda values: ",".join(sorted({str(item) for item in values.dropna()}))
        )
        out["channel_exposure"] = out["customer_id"].map(channel_exposure_map).fillna("")
        out["risk_tier"] = pd.cut(
            out["final_mule_score"],
            bins=[-0.001, 0.55, 0.75, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)
        out["priority_band"] = np.select(
            [out["final_mule_score"] >= 0.85, out["final_mule_score"] >= 0.65],
            ["P1", "P2"],
            default="P3",
        )
        out["alert_rank"] = out["final_mule_score"].rank(method="dense", ascending=False).astype(int)
        alert_output = out[
            [
                "customer_id",
                "account_id",
                "event_ts",
                "channel_exposure",
                "final_mule_score",
                "final_mule_score_rank",
                "risk_tier",
                "priority_band",
                "alert_rank",
                "primary_category",
                "primary_reason",
                "primary_variables",
                "secondary_category",
                "secondary_reason",
                "secondary_variables",
                "tertiary_category",
                "tertiary_reason",
                "tertiary_variables",
                "score_component_model",
                "score_component_sequence",
                "score_component_behavioral",
                "score_component_hazard",
                "score_component_sequence_model",
                "score_component_hmm",
                "score_component_ring",
                "score_component_graph",
                "score_component_cycle",
            ]
            + [c for c in out.columns if c.startswith("prob_")]
        ]
        th_table = self.threshold_table(out, "final_mule_score", CONFIG["alert_daily_capacity"])
        th_opt = self.optimize_threshold_by_capacity(out, "final_mule_score", CONFIG["alert_daily_capacity"])
        ch_th = self.channel_specific_thresholds(out, "final_mule_score", CONFIG["alert_daily_capacity"])
        cls_th = self.class_specific_thresholds(out, [c for c in out.columns if c.startswith("prob_")], CONFIG["alert_daily_capacity"])
        return alert_output, th_table, ch_th, cls_th, out, th_opt
