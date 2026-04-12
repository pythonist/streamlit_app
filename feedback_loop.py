import numpy as np
import pandas as pd

from config import CONFIG
from utils import print_step


class FeedbackLoop:
    def _outcome_from_score(self, score, proxy_positive, rng):
        if score >= 0.85 or (proxy_positive and score >= 0.72):
            probs = [0.45, 0.10, 0.10, 0.25, 0.10]
        elif score >= 0.70:
            probs = [0.30, 0.20, 0.15, 0.20, 0.15]
        elif score >= 0.55:
            probs = [0.16, 0.38, 0.18, 0.10, 0.18]
        else:
            probs = [0.08, 0.52, 0.20, 0.05, 0.15]
        outcomes = ["confirmed_mule", "false_positive", "insufficient_evidence", "sar_filed", "watchlist_only"]
        return rng.choice(outcomes, p=probs)

    def weak_supervision_and_feedback(self, feature_df, alert_output):
        print_step("STEP 17: WEAK SUPERVISION + FEEDBACK")
        rng = np.random.default_rng(int(CONFIG.get("random_state", 42)))

        weak_df = feature_df.copy()
        weak_df["proxy_positive"] = (
            (weak_df["shared_device_risk"] == 1)
            | (weak_df["shared_ip_risk"] == 1)
            | (weak_df["dormant_activation_flag"] == 1)
            | (weak_df["fanout_flag"] == 1)
            | (weak_df["transition_score"] >= 0.60)
        ).astype(int)
        weak_df["proxy_confidence"] = (
            0.2 * weak_df["shared_device_risk"].fillna(0)
            + 0.2 * weak_df["shared_ip_risk"].fillna(0)
            + 0.2 * weak_df["dormant_activation_flag"].fillna(0)
            + 0.2 * weak_df["fanout_flag"].fillna(0)
            + 0.2 * (weak_df["transition_score"].fillna(0) >= 0.60).astype(int)
        )
        weak_df["feedback_signal"] = np.where(weak_df["proxy_positive"] == 1, "high", "low")

        feedback_alerts = alert_output.copy()
        if feedback_alerts.empty:
            return {
                "weak_df": weak_df,
                "feedback_enriched_alerts": feedback_alerts,
                "relabel_candidate_table": pd.DataFrame(),
                "suppression_rule_candidates": pd.DataFrame(),
                "active_learning_queue": pd.DataFrame(),
                "suppression_rules": pd.DataFrame(),
                "precision_by_reason": pd.DataFrame(),
                "retraining_pack": pd.DataFrame(),
            }

        feedback_alerts["alert_id"] = "ALERT_" + np.arange(len(feedback_alerts)).astype(str)
        feedback_alerts["score_bucket"] = pd.cut(
            feedback_alerts["final_mule_score"],
            bins=[-0.001, 0.55, 0.75, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
        ).astype(str)

        sample_n = min(500, len(feedback_alerts))
        sampled_ids = rng.choice(feedback_alerts["alert_id"], size=sample_n, replace=False)
        sampled_rows = feedback_alerts.set_index("alert_id").loc[sampled_ids].reset_index()

        outcomes = []
        for _, row in sampled_rows.iterrows():
            outcome = self._outcome_from_score(
                float(row.get("final_mule_score", 0.0)),
                int(row.get("proxy_positive", 0)) if "proxy_positive" in row else 0,
                rng,
            )
            outcomes.append(outcome)

        analyst_feedback_master = pd.DataFrame(
            {
                "alert_id": sampled_rows["alert_id"].values,
                "customer_id": sampled_rows["customer_id"].values,
                "account_id": sampled_rows["account_id"].values,
                "analyst_outcome": outcomes,
                "analyst_notes": [
                    "elevated score review" if outcome in {"confirmed_mule", "sar_filed"} else "standard review"
                    for outcome in outcomes
                ],
                "review_ts": pd.Timestamp("2025-01-01") + pd.to_timedelta(np.arange(sample_n), unit="h"),
            }
        )

        feedback_enriched = feedback_alerts.merge(analyst_feedback_master, on=["alert_id"], how="left", suffixes=("", "_fb"))
        relabel_candidate_table = feedback_enriched[feedback_enriched["analyst_outcome"].isin(["confirmed_mule", "sar_filed"])].copy()
        suppression_rule_candidates = feedback_enriched[feedback_enriched["analyst_outcome"] == "false_positive"].copy()
        active_learning_queue = feedback_enriched[feedback_enriched["final_mule_score"].between(0.45, 0.65)].copy()

        suppression_rules = pd.DataFrame()
        if not suppression_rule_candidates.empty:
            cols = [c for c in ["primary_reason", "primary_category", "priority_band"] if c in suppression_rule_candidates.columns]
            suppression_rules = (
                suppression_rule_candidates.groupby(cols)
                .size()
                .reset_index(name="false_positive_count")
                .sort_values("false_positive_count", ascending=False)
            )

        precision_by_reason = pd.DataFrame()
        if "primary_reason" in feedback_enriched.columns:
            tmp = feedback_enriched.copy()
            tmp["is_tp"] = tmp["analyst_outcome"].isin(["confirmed_mule", "sar_filed"]).astype(int)
            precision_by_reason = (
                tmp.groupby("primary_reason")
                .agg(reviewed_alerts=("primary_reason", "count"), true_positives=("is_tp", "sum"))
                .reset_index()
            )
            precision_by_reason["precision"] = precision_by_reason["true_positives"] / precision_by_reason["reviewed_alerts"].replace(0, np.nan)

        retraining_pack = feedback_enriched[feedback_enriched["analyst_outcome"].isin(["confirmed_mule", "sar_filed", "false_positive"])].copy()
        retraining_pack["review_weight"] = np.where(retraining_pack["analyst_outcome"].isin(["confirmed_mule", "sar_filed"]), 1.0, 0.5)

        return {
            "weak_df": weak_df,
            "feedback_enriched_alerts": feedback_enriched,
            "relabel_candidate_table": relabel_candidate_table,
            "suppression_rule_candidates": suppression_rule_candidates,
            "active_learning_queue": active_learning_queue,
            "suppression_rules": suppression_rules,
            "precision_by_reason": precision_by_reason,
            "retraining_pack": retraining_pack,
        }
