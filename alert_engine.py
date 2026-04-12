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

        def strengthen_category(row):
            if row.get("dormant_activation_flag", 0) == 1 and row.get("sequence_score", 0) >= 0.55:
                return "sleeper_mule"
            if row.get("fanout_flag", 0) == 1:
                return "fanout_mule"
            if row.get("channel", "") in ["ATM", "BRANCH"] and str(row.get("transaction_type", "")).upper() == "CASHOUT":
                return "cashout_mule"
            if row.get("lstm_emerging_prob", 0) >= 0.7 and row.get("first_time_counterparty", 0) == 1:
                return "first_time_mule"
            if row.get("behavioral_risk_score", 0) >= 0.65:
                return "layering_mule"
            return row["primary_category"]

        out["primary_category"] = out.apply(strengthen_category, axis=1)
        return out

    def build_reasoning(self, df):
        def reason_builder(row):
            reasons, variables = [], []
            if row.get("first_time_counterparty", 0) == 1:
                reasons.append("First-time counterparty observed")
                variables.append("first_time_counterparty")
            if row.get("first_time_device", 0) == 1:
                reasons.append("First-time device used")
                variables.append("first_time_device")
            if row.get("shared_device_risk", 0) == 1:
                reasons.append("Shared device across customers")
                variables.append("shared_device_customer_count")
            if row.get("shared_ip_risk", 0) == 1:
                reasons.append("Shared IP across customers")
                variables.append("shared_ip_customer_count")
            if row.get("fanout_flag", 0) == 1:
                reasons.append("Rapid fan-out behavior")
                variables.append("daily_unique_counterparties")
            if row.get("velocity_flag", 0) == 1:
                reasons.append("High transaction velocity")
                variables.append("daily_txn_count")
            if row.get("dormant_activation_flag", 0) == 1:
                reasons.append("Dormant account activation")
                variables.append("dormancy_days")
            if row.get("transition_score", 0) > 0.6:
                reasons.append("Unusual transition path")
                variables.append("transition_score")
            if row.get("cust_graph_pagerank", 0) > 0.001:
                reasons.append("High graph centrality")
                variables.append("cust_graph_pagerank")
            if row.get("ring_count", 0) > 0:
                reasons.append("Account linked to ring/cycle")
                variables.append("ring_count")
            if row.get("behavioral_risk_score", 0) >= 0.6:
                reasons.append("Elevated behavioural risk pattern")
                variables.append("behavioral_risk_score")
            if len(reasons) == 0:
                reasons = ["Model-based typology assignment"]
                variables = ["model_top_prob"]
            return pd.Series(
                {
                    "primary_reason": reasons[0] if len(reasons) > 0 else None,
                    "primary_variables": variables[0] if len(variables) > 0 else None,
                    "secondary_category": "network_risk",
                    "secondary_reason": reasons[1] if len(reasons) > 1 else None,
                    "secondary_variables": variables[1] if len(variables) > 1 else None,
                    "tertiary_category": "sequence_risk",
                    "tertiary_reason": reasons[2] if len(reasons) > 2 else None,
                    "tertiary_variables": variables[2] if len(variables) > 2 else None,
                }
            )

        reason_df = df.apply(reason_builder, axis=1)
        return pd.concat([df.reset_index(drop=True), reason_df.reset_index(drop=True)], axis=1)

    def threshold_table(self, df, score_col, daily_capacity):
        if df.empty or score_col not in df.columns:
            return pd.DataFrame(columns=["threshold", "avg_daily_alerts", "within_capacity", "precision_proxy", "flagged_volume"])

        rows = []
        thresholds = np.arange(0.1, 0.96, 0.05)
        for th in thresholds:
            flagged = df[score_col] >= th
            daily_counts = self._daily_counts(df, score_col, th)
            avg_daily = daily_counts.mean() if len(daily_counts) else 0
            precision_proxy = df.loc[flagged, "label"].isin(CONFIG["risky_labels"]).mean() if flagged.sum() > 0 else 0
            rows.append(
                {
                    "threshold": round(float(th), 2),
                    "avg_daily_alerts": round(float(avg_daily), 2),
                    "within_capacity": int(avg_daily <= daily_capacity),
                    "precision_proxy": round(float(precision_proxy), 4),
                    "flagged_volume": int(flagged.sum()),
                }
            )
        return pd.DataFrame(rows)

    def optimize_threshold_by_capacity(self, df, score_col, daily_capacity):
        if df.empty or score_col not in df.columns:
            return {"selected_threshold": None, "avg_daily_alerts": None}

        thresholds = np.arange(0.1, 0.96, 0.01)
        candidates = []
        for th in thresholds:
            daily_counts = self._daily_counts(df, score_col, th)
            avg_daily = daily_counts.mean() if len(daily_counts) else 0
            if avg_daily <= daily_capacity:
                candidates.append((th, avg_daily))
        if not candidates:
            return {"selected_threshold": None, "avg_daily_alerts": None}
        selected = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
        return {"selected_threshold": round(float(selected[0]), 2), "avg_daily_alerts": round(float(selected[1]), 2)}

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
            thresholds = np.arange(0.1, 0.96, 0.01)
            found = None
            for th in thresholds:
                daily_counts = self._daily_counts(df, col, th)
                avg_daily = daily_counts.mean() if len(daily_counts) else 0
                if avg_daily <= alloc:
                    found = (th, avg_daily)
            rows.append(
                {
                    "probability_column": col,
                    "selected_threshold": round(float(found[0]), 2) if found else None,
                    "avg_daily_alerts": round(float(found[1]), 2) if found else None,
                }
            )
        return pd.DataFrame(rows)

    def model8_alert_pack(self, test_df, test_prob, model6_artifacts):
        print_step("STEP 16: MODEL8 ALERT PACK")
        out = self.build_reasoning(test_df.copy())
        for i, cls in enumerate(model6_artifacts.label_encoder.classes_):
            out[f"prob_{cls}"] = test_prob[:, i] if len(test_prob) else 0.0
        out["channel_exposure"] = out.groupby("customer_id")["channel"].transform(
            lambda x: ",".join(sorted(set(map(str, x.dropna()))))
        )
        out["risk_tier"] = pd.cut(
            out["final_mule_score"],
            bins=[-0.001, 0.55, 0.75, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)
        out["priority_band"] = out["final_mule_score"].apply(assign_priority_band)
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
