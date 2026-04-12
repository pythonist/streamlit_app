from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, top_k_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

from config import CONFIG
from utils import month_backtest_table, print_step

USE_SHAP = False
if CONFIG.get("enable_shap", False):
    try:
        import shap

        USE_SHAP = True
    except Exception:
        pass


@dataclass
class Model6Artifacts:
    preprocessor: any
    label_encoder: any
    base_model: any
    calibrated_model: any
    challenger_model: any
    feature_names: np.ndarray


class MulticlassModel:
    def split_time_based(self, df, train_end, valid_end):
        df = df.sort_values("event_ts").reset_index(drop=True).copy()
        train_df = df[df["event_ts"] <= pd.to_datetime(train_end)].copy()
        valid_df = df[(df["event_ts"] > pd.to_datetime(train_end)) & (df["event_ts"] <= pd.to_datetime(valid_end))].copy()
        test_df = df[df["event_ts"] > pd.to_datetime(valid_end)].copy()
        return train_df, valid_df, test_df

    def _build_champion_model(self):
        model_name = CONFIG.get("classifier_model", "Random Forest")
        estimators = int(CONFIG.get("classifier_estimators", 220))
        max_depth = int(CONFIG.get("classifier_max_depth", 12))
        min_leaf = int(CONFIG.get("classifier_min_samples_leaf", 4))

        if model_name == "Extra Trees":
            return ExtraTreesClassifier(
                n_estimators=estimators,
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                class_weight="balanced",
                n_jobs=-1,
                random_state=CONFIG["random_state"],
            )
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=max(80, estimators // 2),
                learning_rate=0.08,
                max_depth=min(max_depth, 5),
                random_state=CONFIG["random_state"],
            )
        return RandomForestClassifier(
            n_estimators=estimators,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=CONFIG["random_state"],
        )

    def _build_challenger_model(self):
        model_name = CONFIG.get("challenger_model", "Logistic Regression")
        if model_name == "SVM Linear":
            return LinearSVC(class_weight="balanced", max_iter=4000, random_state=CONFIG["random_state"])
        if model_name == "Naive Bayes":
            return GaussianNB()
        return LogisticRegression(max_iter=1200, class_weight="balanced")

    def _select_feature_columns(self, train_df):
        exclude_cols = {
            "transaction_id",
            "event_ts",
            "label",
            "mule_category",
            "prev_event_ts",
            "session_id",
            "transition_path",
            "upi_transaction_id",
            "atm_transaction_id",
            "branch_transaction_id",
            "merchant_transaction_id",
        }
        blocked_tokens = ["_id", "_name", "_address", "_number", "_hash", "_signature", "email", "url", "_node"]

        selected = []
        for col in train_df.columns:
            if col in exclude_cols:
                continue
            series = train_df[col]
            if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
                continue
            if not series.notna().any():
                continue
            if series.nunique(dropna=True) <= 1:
                continue

            col_lower = col.lower()
            if any(token in col_lower for token in blocked_tokens):
                continue

            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                selected.append(col)
                continue

            nunique = int(series.nunique(dropna=True))
            uniqueness_ratio = nunique / max(len(series), 1)
            if nunique <= 25 and uniqueness_ratio <= 0.10:
                selected.append(col)

        return selected

    def model6_multiclass(self, train_df, valid_df, test_df):
        print_step("STEP 14: MODEL6 MULTICLASS")
        feature_cols = self._select_feature_columns(train_df)

        if not feature_cols:
            raise ValueError("No usable features were available for model training.")

        X_train = train_df[feature_cols].copy()
        X_valid = valid_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()

        label_enc = LabelEncoder()
        y_train = label_enc.fit_transform(train_df["label"].astype(str))
        y_valid = label_enc.transform(valid_df["label"].astype(str))
        y_test = label_enc.transform(test_df["label"].astype(str))

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [col for col in X_train.columns if col not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    cat_cols,
                ),
            ]
        )

        X_train_proc = preprocessor.fit_transform(X_train)
        X_valid_proc = preprocessor.transform(X_valid)
        X_test_proc = preprocessor.transform(X_test)

        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = np.array([f"f_{i}" for i in range(X_train_proc.shape[1])])

        champion_model = self._build_champion_model()
        champion_model.fit(X_train_proc, y_train)

        calibrated_model = champion_model
        calibration_method = CONFIG.get("calibration_method", "sigmoid")
        try:
            calibrated_model = CalibratedClassifierCV(self._build_champion_model(), method=calibration_method, cv=3)
            calibrated_model.fit(X_train_proc, y_train)
        except Exception as exc:
            print(f"Calibration fallback activated: {exc}")

        challenger_model = self._build_challenger_model()
        X_train_ch = X_train_proc.toarray() if hasattr(X_train_proc, "toarray") and isinstance(challenger_model, GaussianNB) else X_train_proc
        X_valid_ch = X_valid_proc.toarray() if hasattr(X_valid_proc, "toarray") and isinstance(challenger_model, GaussianNB) else X_valid_proc
        X_test_ch = X_test_proc.toarray() if hasattr(X_test_proc, "toarray") and isinstance(challenger_model, GaussianNB) else X_test_proc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            challenger_model.fit(X_train_ch, y_train)

        valid_prob = calibrated_model.predict_proba(X_valid_proc)
        test_prob = calibrated_model.predict_proba(X_test_proc)
        valid_pred = np.argmax(valid_prob, axis=1)
        test_pred = np.argmax(test_prob, axis=1)

        if hasattr(challenger_model, "predict_proba"):
            valid_prob_ch = challenger_model.predict_proba(X_valid_ch)
            test_prob_ch = challenger_model.predict_proba(X_test_ch)
            valid_pred_ch = np.argmax(valid_prob_ch, axis=1)
            test_pred_ch = np.argmax(test_prob_ch, axis=1)
        else:
            valid_pred_ch = challenger_model.predict(X_valid_ch)
            test_pred_ch = challenger_model.predict(X_test_ch)

        print(classification_report(y_test, test_pred, target_names=label_enc.classes_, zero_division=0))
        print("Accuracy:", accuracy_score(y_test, test_pred))
        print("Macro F1:", f1_score(y_test, test_pred, average="macro"))
        print("Top-2 accuracy:", top_k_accuracy_score(y_test, test_prob, k=min(2, len(label_enc.classes_)), labels=np.arange(len(label_enc.classes_))))
        print("Top-3 accuracy:", top_k_accuracy_score(y_test, test_prob, k=min(3, len(label_enc.classes_)), labels=np.arange(len(label_enc.classes_))))

        if hasattr(champion_model, "feature_importances_"):
            feature_importance = champion_model.feature_importances_
        else:
            feature_importance = np.zeros(len(feature_names))
        fi = pd.DataFrame({"feature": feature_names, "importance": feature_importance}).sort_values("importance", ascending=False)

        print(fi.head(20))
        print(month_backtest_table(test_df["event_ts"], y_test, test_pred))

        if USE_SHAP and hasattr(champion_model, "feature_importances_"):
            try:
                explainer = shap.TreeExplainer(champion_model)
                shap_values = explainer.shap_values(X_test_proc[:400])
                shap.summary_plot(shap_values, X_test_proc[:400], feature_names=feature_names)
            except Exception as exc:
                print("SHAP failed:", exc)

        artifacts = Model6Artifacts(preprocessor, label_enc, champion_model, calibrated_model, challenger_model, feature_names)
        return artifacts, feature_cols, y_valid, y_test, valid_prob, test_prob, valid_pred, test_pred, valid_pred_ch, test_pred_ch, fi

    def plot_confusion_matrix(self, y_true, y_pred, label_encoder):
        return confusion_matrix(y_true, y_pred)
