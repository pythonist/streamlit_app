import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from utils import print_step

USE_HMM = False
USE_TF = False
try:
    from hmmlearn import hmm

    USE_HMM = True
except Exception:
    pass

if CONFIG.get("enable_tensorflow_sequences", False):
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, LSTM, Masking
        from tensorflow.keras.models import Sequential

        USE_TF = True
    except Exception:
        pass


class SequenceModels:
    def _usable_numeric_features(self, df, feature_cols):
        usable = []
        for col in feature_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if not df[col].notna().any():
                continue
            usable.append(col)
        return usable

    def _constant_sequence_outputs(self, valid_ids, test_ids, train_mask, valid_mask, test_mask, model_name="constant"):
        return {
            "lstm_model": None,
            "valid_lstm_prob": np.zeros(len(valid_ids), dtype=float),
            "test_lstm_prob": np.zeros(len(test_ids), dtype=float),
            "valid_ids": valid_ids,
            "test_ids": test_ids,
            "train_attention_mask": train_mask,
            "valid_attention_mask": valid_mask,
            "test_attention_mask": test_mask,
            "sequence_model_type": model_name,
        }

    def _sequence_summary(self, seq_tensor):
        if len(seq_tensor) == 0:
            return np.empty((0, 0))
        last_step = seq_tensor[:, -1, :]
        seq_mean = seq_tensor.mean(axis=1)
        seq_max = seq_tensor.max(axis=1)
        return np.hstack([last_step, seq_mean, seq_max])

    def _surrogate_sequence_model(self, X_train_seq, y_train_seq, X_valid_seq, X_test_seq):
        if len(X_train_seq) == 0:
            return None, np.zeros(len(X_valid_seq)), np.zeros(len(X_test_seq))

        X_train = self._sequence_summary(X_train_seq)
        X_valid = self._sequence_summary(X_valid_seq)
        X_test = self._sequence_summary(X_test_seq)

        if X_train.shape[1] == 0 or len(np.unique(y_train_seq)) < 2:
            baseline = float(np.mean(y_train_seq)) if len(y_train_seq) else 0.0
            return None, np.full(len(X_valid_seq), baseline), np.full(len(X_test_seq), baseline)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid) if len(X_valid) else X_valid
        X_test_scaled = scaler.transform(X_test) if len(X_test) else X_test

        model = LogisticRegression(max_iter=800, class_weight="balanced")
        model.fit(X_train_scaled, y_train_seq)

        valid_prob = model.predict_proba(X_valid_scaled)[:, 1] if len(X_valid_scaled) else np.zeros(0)
        test_prob = model.predict_proba(X_test_scaled)[:, 1] if len(X_test_scaled) else np.zeros(0)
        return model, valid_prob, test_prob

    def model3_hazard(self, train_df, valid_df, test_df):
        print_step("STEP 11: MODEL3 HAZARD")
        hazard_features = [
            "hours_since_prev_txn",
            "first_time_counterparty",
            "first_time_device",
            "first_time_channel",
            "shared_device_risk",
            "shared_ip_risk",
            "fanout_flag",
            "velocity_flag",
            "transition_score",
            "dormant_activation_flag",
        ]
        hazard_features = self._usable_numeric_features(train_df, hazard_features)
        for dataset in [train_df, valid_df, test_df]:
            dataset["emerging_mule_flag"] = dataset["label"].isin(CONFIG["risky_labels"]).astype(int)

        if not hazard_features or train_df["emerging_mule_flag"].nunique() < 2:
            for dataset in [valid_df, test_df]:
                dataset["hazard_score"] = 0.0
            return None, train_df, valid_df, test_df

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train_df[hazard_features])
        X_valid = imputer.transform(valid_df[hazard_features])
        X_test = imputer.transform(test_df[hazard_features])

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train, train_df["emerging_mule_flag"])
        valid_df["hazard_score"] = model.predict_proba(X_valid)[:, 1]
        test_df["hazard_score"] = model.predict_proba(X_test)[:, 1]
        return model, train_df, valid_df, test_df

    def model4_hmm(self, train_df, valid_df, test_df):
        print_step("STEP 12: MODEL4 HMM")
        hmm_features = [
            "amount",
            "event_hour",
            "hours_since_prev_txn",
            "velocity_flag",
            "fanout_flag",
            "transition_score",
            "shared_device_risk",
            "shared_ip_risk",
            "dormant_activation_flag",
        ]
        hmm_features = self._usable_numeric_features(train_df, hmm_features)
        if not USE_HMM or not hmm_features or len(train_df) < 200:
            for dataset in [train_df, valid_df, test_df]:
                dataset["hmm_state"] = 0
                dataset["hmm_sequence_anomaly_score"] = 0.0
            return None, train_df, valid_df, test_df

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[hmm_features].fillna(0))
        X_valid = scaler.transform(valid_df[hmm_features].fillna(0))
        X_test = scaler.transform(test_df[hmm_features].fillna(0))

        hmm_model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=120, random_state=CONFIG["random_state"])
        hmm_model.fit(X_train)

        train_df["hmm_state"] = hmm_model.predict(X_train)
        valid_df["hmm_state"] = hmm_model.predict(X_valid)
        test_df["hmm_state"] = hmm_model.predict(X_test)

        train_df["hmm_sequence_anomaly_score"] = 1.0 - hmm_model.predict_proba(X_train).max(axis=1)
        valid_df["hmm_sequence_anomaly_score"] = 1.0 - hmm_model.predict_proba(X_valid).max(axis=1)
        test_df["hmm_sequence_anomaly_score"] = 1.0 - hmm_model.predict_proba(X_test).max(axis=1)

        return hmm_model, train_df, valid_df, test_df

    def build_sequence_tensor(self, df, feature_cols, entity_col="customer_id", time_col="event_ts", max_len=25):
        seqs, ids = [], []
        df = df.sort_values([entity_col, time_col]).copy()
        for entity_id, part in df.groupby(entity_col):
            values = part[feature_cols].fillna(0).values
            if len(values) > max_len:
                values = values[-max_len:]
            elif len(values) < max_len:
                pad = np.zeros((max_len - len(values), len(feature_cols)))
                values = np.vstack([pad, values])
            seqs.append(values)
            ids.append(entity_id)
        if not seqs:
            return np.zeros((0, max_len, len(feature_cols))), []
        return np.array(seqs), ids

    def build_transformer_inputs(self, df, feature_cols, entity_col="customer_id", time_col="event_ts", max_len=25):
        X, masks, ids = [], [], []
        df = df.sort_values([entity_col, time_col]).copy()
        for entity_id, part in df.groupby(entity_col):
            values = part[feature_cols].fillna(0).values
            length = len(values)
            if length > max_len:
                values = values[-max_len:]
                mask = np.ones(max_len)
            else:
                pad_len = max_len - length
                values = np.vstack([np.zeros((pad_len, len(feature_cols))), values])
                mask = np.concatenate([np.zeros(pad_len), np.ones(length)])
            X.append(values)
            masks.append(mask)
            ids.append(entity_id)
        if not X:
            return np.zeros((0, max_len, len(feature_cols))), np.zeros((0, max_len)), []
        return np.array(X), np.array(masks), ids

    def model5_lstm_and_transformer(self, train_df, valid_df, test_df):
        print_step("STEP 13: MODEL5 LSTM + TRANSFORMER INPUTS")
        feature_cols = [
            "amount",
            "event_hour",
            "hours_since_prev_txn",
            "first_time_counterparty",
            "first_time_device",
            "first_time_channel",
            "shared_device_risk",
            "shared_ip_risk",
            "fanout_flag",
            "velocity_flag",
            "transition_score",
            "dormant_activation_flag",
        ]
        feature_cols = self._usable_numeric_features(train_df, feature_cols)

        X_train_seq, train_ids = self.build_sequence_tensor(train_df, feature_cols, max_len=CONFIG["sequence_max_len"]) if feature_cols else (np.zeros((0, 0, 0)), [])
        X_valid_seq, valid_ids = self.build_sequence_tensor(valid_df, feature_cols, max_len=CONFIG["sequence_max_len"]) if feature_cols else (np.zeros((0, 0, 0)), [])
        X_test_seq, test_ids = self.build_sequence_tensor(test_df, feature_cols, max_len=CONFIG["sequence_max_len"]) if feature_cols else (np.zeros((0, 0, 0)), [])
        X_train_tr, train_mask, _ = self.build_transformer_inputs(train_df, feature_cols, max_len=CONFIG["sequence_max_len"]) if feature_cols else (np.zeros((0, 0, 0)), np.zeros((0, 0)), [])
        X_valid_tr, valid_mask, _ = self.build_transformer_inputs(valid_df, feature_cols, max_len=CONFIG["sequence_max_len"]) if feature_cols else (np.zeros((0, 0, 0)), np.zeros((0, 0)), [])
        X_test_tr, test_mask, _ = self.build_transformer_inputs(test_df, feature_cols, max_len=CONFIG["sequence_max_len"]) if feature_cols else (np.zeros((0, 0, 0)), np.zeros((0, 0)), [])

        train_target_map = train_df.groupby("customer_id")["label"].apply(lambda labels: int(labels.isin(CONFIG["risky_labels"]).any())).to_dict()
        valid_target_map = valid_df.groupby("customer_id")["label"].apply(lambda labels: int(labels.isin(CONFIG["risky_labels"]).any())).to_dict()
        y_train_seq = np.array([train_target_map.get(entity_id, 0) for entity_id in train_ids])
        y_valid_seq = np.array([valid_target_map.get(entity_id, 0) for entity_id in valid_ids])

        if not feature_cols or len(X_train_seq) == 0:
            return self._constant_sequence_outputs(valid_ids, test_ids, train_mask, valid_mask, test_mask)

        if USE_TF and CONFIG.get("enable_tensorflow_sequences", False) and len(np.unique(y_train_seq)) >= 2:
            try:
                tf.keras.backend.clear_session()
                lstm_model = Sequential(
                    [
                        tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                        Masking(mask_value=0.0),
                        LSTM(32),
                        Dense(16, activation="relu"),
                        Dense(1, activation="sigmoid"),
                    ]
                )
                lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
                lstm_model.fit(
                    X_train_seq,
                    y_train_seq,
                    validation_data=(X_valid_seq, y_valid_seq),
                    epochs=int(CONFIG.get("sequence_epochs", 3)),
                    batch_size=int(CONFIG.get("sequence_batch_size", 128)),
                    verbose=0,
                )
                valid_lstm_prob = lstm_model.predict(X_valid_seq, verbose=0).ravel() if len(X_valid_seq) else np.zeros(0)
                test_lstm_prob = lstm_model.predict(X_test_seq, verbose=0).ravel() if len(X_test_seq) else np.zeros(0)
                return {
                    "lstm_model": lstm_model,
                    "valid_lstm_prob": valid_lstm_prob,
                    "test_lstm_prob": test_lstm_prob,
                    "valid_ids": valid_ids,
                    "test_ids": test_ids,
                    "train_attention_mask": train_mask,
                    "valid_attention_mask": valid_mask,
                    "test_attention_mask": test_mask,
                    "sequence_model_type": "lstm",
                }
            except Exception as exc:
                print(f"TensorFlow sequence fallback activated: {exc}")

        surrogate_model, valid_lstm_prob, test_lstm_prob = self._surrogate_sequence_model(
            X_train_seq, y_train_seq, X_valid_seq, X_test_seq
        )
        return {
            "lstm_model": surrogate_model,
            "valid_lstm_prob": valid_lstm_prob,
            "test_lstm_prob": test_lstm_prob,
            "valid_ids": valid_ids,
            "test_ids": test_ids,
            "train_attention_mask": train_mask,
            "valid_attention_mask": valid_mask,
            "test_attention_mask": test_mask,
            "sequence_model_type": "surrogate_logistic",
        }
