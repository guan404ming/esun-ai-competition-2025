"""
Alert Pattern Detection System
Training ONLY on verified alert account patterns for maximum precision

This approach uses unsupervised learning on alert accounts to identify
the most distinctive fraud patterns, then applies strict similarity
matching to predict future alerts.

Author: AI Assistant
Date: 2025-01-21
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


class AlertPatternDetector:
    """
    Ultra-conservative fraud detector that learns ONLY from verified alert patterns.
    Uses pattern matching and anomaly detection for high precision.
    """

    def __init__(self, lookback_days: int = 30, prediction_horizon: int = 30):
        """Initialize the alert pattern detector"""
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon

        # Data storage
        self.df_transaction = None
        self.df_alert = None
        self.df_predict = None
        self.alert_dates = {}

        # Pattern learning
        self.scaler = StandardScaler()
        self.alert_patterns = None  # Learned alert patterns
        self.alert_features_df = None
        self.similarity_threshold = 0.7  # Conservative similarity threshold
        self.feature_names = []

        # Clustering for pattern discovery
        self.dbscan = DBSCAN(eps=0.3, min_samples=3)
        self.core_alert_patterns = None

    def load_data(self, data_dir: str = "data"):
        """Load and optimize data"""
        print(f"Loading data from {data_dir}...")

        # Load transaction data
        transaction_files = ["acct_transaction_transfered.csv", "acct_transaction.csv"]
        for filename in transaction_files:
            transaction_path = os.path.join(data_dir, filename)
            if os.path.exists(transaction_path):
                print(f"Loading {filename}...")
                self.df_transaction = pd.read_csv(transaction_path)
                break

        # Load other datasets
        self.df_alert = pd.read_csv(os.path.join(data_dir, "acct_alert.csv"))
        self.df_predict = pd.read_csv(os.path.join(data_dir, "acct_predict.csv"))

        # Create alert date lookup
        self.alert_dates = dict(zip(self.df_alert["acct"], self.df_alert["event_date"]))

        print(f"Loaded {len(self.df_transaction):,} transactions")
        print(f"Loaded {len(self.df_alert):,} alert accounts")
        print(f"Loaded {len(self.df_predict):,} accounts to predict")

        # Optimize data types
        self._optimize_data_types()

    def _optimize_data_types(self):
        """Optimize data types for memory and performance"""
        print("Optimizing data types...")

        if "txn_date" in self.df_transaction.columns:
            self.df_transaction["txn_date"] = pd.to_numeric(
                self.df_transaction["txn_date"], errors="coerce"
            ).astype("int16")

        if "txn_amt" in self.df_transaction.columns:
            self.df_transaction["txn_amt"] = self.df_transaction["txn_amt"].astype(
                "float32"
            )

    def extract_alert_patterns(self) -> pd.DataFrame:
        """Extract patterns ONLY from verified alert accounts"""
        print("Extracting patterns from verified alert accounts...")

        alert_features = []

        for acct in tqdm(self.alert_dates.keys(), desc="Processing alert accounts"):
            alert_day = self.alert_dates[acct]

            # Extract features from the period BEFORE the alert
            analysis_end_day = max(1, alert_day - self.prediction_horizon)

            if analysis_end_day > self.lookback_days:
                features = self._extract_comprehensive_features(acct, analysis_end_day)

                # Only include accounts with meaningful transaction activity
                if features["total_transactions"] >= 10:
                    features["acct"] = acct
                    features["alert_day"] = alert_day
                    alert_features.append(features)

        self.alert_features_df = pd.DataFrame(alert_features)

        if len(self.alert_features_df) == 0:
            raise ValueError("No valid alert patterns found")

        # Remove account identifiers for pattern analysis
        pattern_features = self.alert_features_df.drop(
            columns=["acct", "alert_day"]
        ).fillna(0)

        # Handle any remaining NaN or infinite values
        pattern_features = pattern_features.replace([np.inf, -np.inf], 0)
        pattern_features = pattern_features.fillna(0)

        self.feature_names = pattern_features.columns.tolist()

        print(f"Extracted {len(self.alert_features_df)} alert patterns")
        print(f"Features per pattern: {len(self.feature_names)}")

        return pattern_features

    def _extract_comprehensive_features(self, acct: str, end_day: int) -> Dict:
        """Extract comprehensive features for pattern learning"""
        start_day = max(1, end_day - self.lookback_days)

        # Get all transactions for this account in the window
        time_mask = (self.df_transaction["txn_date"] >= start_day) & (
            self.df_transaction["txn_date"] <= end_day
        )
        account_mask = (self.df_transaction["from_acct"] == acct) | (
            self.df_transaction["to_acct"] == acct
        )
        acct_txns = self.df_transaction[time_mask & account_mask]

        if len(acct_txns) == 0:
            return self._get_zero_features()

        # Separate incoming and outgoing
        incoming = acct_txns[acct_txns["to_acct"] == acct]
        outgoing = acct_txns[acct_txns["from_acct"] == acct]

        features = {}

        # 1. Basic Transaction Patterns
        features.update(self._extract_basic_patterns(acct_txns, incoming, outgoing))

        # 2. Money Flow Patterns
        features.update(
            self._extract_money_flow_patterns(acct_txns, incoming, outgoing)
        )

        # 3. Network Connectivity Patterns
        features.update(self._extract_network_patterns(acct_txns, incoming, outgoing))

        # 4. Temporal Behavior Patterns
        features.update(self._extract_temporal_patterns(acct_txns))

        # 5. Amount Distribution Patterns
        features.update(self._extract_amount_patterns(acct_txns, incoming, outgoing))

        # 6. Suspicious Activity Indicators
        features.update(
            self._extract_suspicious_indicators(acct_txns, incoming, outgoing)
        )

        return features

    def _extract_basic_patterns(self, acct_txns, incoming, outgoing):
        """Extract basic transaction count and frequency patterns"""
        return {
            "total_transactions": len(acct_txns),
            "incoming_count": len(incoming),
            "outgoing_count": len(outgoing),
            "transaction_frequency": len(acct_txns) / self.lookback_days,
            "in_out_ratio": len(incoming) / max(len(outgoing), 1),
            "activity_days": acct_txns["txn_date"].nunique(),
            "daily_avg_transactions": len(acct_txns)
            / max(acct_txns["txn_date"].nunique(), 1),
        }

    def _extract_money_flow_patterns(self, acct_txns, incoming, outgoing):
        """Extract money movement and flow patterns"""
        features = {}

        # Amount flows
        incoming_total = incoming["txn_amt"].sum() if len(incoming) > 0 else 0
        outgoing_total = outgoing["txn_amt"].sum() if len(outgoing) > 0 else 0

        features.update(
            {
                "total_incoming": incoming_total,
                "total_outgoing": outgoing_total,
                "net_flow": incoming_total - outgoing_total,
                "flow_ratio": incoming_total / max(outgoing_total, 1),
                "turnover_efficiency": min(incoming_total, outgoing_total)
                / max(incoming_total, outgoing_total, 1),
            }
        )

        # Average amounts
        features.update(
            {
                "avg_incoming_amount": incoming["txn_amt"].mean()
                if len(incoming) > 0
                else 0,
                "avg_outgoing_amount": outgoing["txn_amt"].mean()
                if len(outgoing) > 0
                else 0,
                "overall_avg_amount": acct_txns["txn_amt"].mean(),
            }
        )

        # Amount velocity
        features["amount_velocity"] = (
            incoming_total + outgoing_total
        ) / self.lookback_days

        return features

    def _extract_network_patterns(self, acct_txns, incoming, outgoing):
        """Extract network connectivity patterns"""
        features = {}

        # Unique counterparties
        unique_senders = incoming["from_acct"].nunique() if len(incoming) > 0 else 0
        unique_receivers = outgoing["to_acct"].nunique() if len(outgoing) > 0 else 0

        features.update(
            {
                "unique_senders": unique_senders,
                "unique_receivers": unique_receivers,
                "total_unique_counterparties": unique_senders + unique_receivers,
                "counterparty_diversity": (unique_senders + unique_receivers)
                / max(len(acct_txns), 1),
            }
        )

        # Hub characteristics
        if len(incoming) > 0:
            features["incoming_concentration"] = len(incoming) / max(unique_senders, 1)
        else:
            features["incoming_concentration"] = 0

        if len(outgoing) > 0:
            features["outgoing_dispersion"] = unique_receivers / max(len(outgoing), 1)
        else:
            features["outgoing_dispersion"] = 0

        # Collection hub score (need to get in_out_ratio from basic patterns)
        in_out_ratio = len(incoming) / max(len(outgoing), 1)
        hub_score = 0
        if in_out_ratio > 2:
            hub_score += 2
        if features["incoming_concentration"] > 3:
            hub_score += 1
        if unique_senders > 15:
            hub_score += 1
        features["collection_hub_score"] = hub_score

        return features

    def _extract_temporal_patterns(self, acct_txns):
        """Extract temporal behavior patterns"""
        features = {}

        # Daily patterns
        daily_counts = acct_txns["txn_date"].value_counts()
        features.update(
            {
                "max_daily_transactions": daily_counts.max()
                if len(daily_counts) > 0
                else 0,
                "daily_transaction_std": daily_counts.std()
                if len(daily_counts) > 0
                else 0,
                "burst_intensity": daily_counts.max() / max(daily_counts.mean(), 1)
                if len(daily_counts) > 0
                else 0,
            }
        )

        # Activity concentration
        features["activity_concentration"] = (
            acct_txns["txn_date"].nunique() / self.lookback_days
        )

        # Channel usage patterns
        if "channel_type" in acct_txns.columns:
            channel_counts = acct_txns["channel_type"].value_counts(normalize=True)
            features.update(
                {
                    "mobile_banking_usage": channel_counts.get("03", 0),
                    "internet_banking_usage": channel_counts.get("04", 0),
                    "channel_diversity": acct_txns["channel_type"].nunique(),
                }
            )
        else:
            features.update(
                {
                    "mobile_banking_usage": 0,
                    "internet_banking_usage": 0,
                    "channel_diversity": 1,
                }
            )

        return features

    def _extract_amount_patterns(self, acct_txns, incoming, outgoing):
        """Extract amount distribution patterns"""
        amounts = acct_txns["txn_amt"]

        features = {
            "amount_mean": amounts.mean(),
            "amount_median": amounts.median(),
            "amount_std": amounts.std(),
            "amount_max": amounts.max(),
            "amount_min": amounts.min(),
            "amount_range": amounts.max() - amounts.min(),
            "amount_skewness": amounts.skew(),
        }

        # Amount distribution analysis
        features.update(
            {
                "small_amounts_ratio": (amounts <= 5000).mean(),
                "medium_amounts_ratio": ((amounts > 5000) & (amounts <= 100000)).mean(),
                "large_amounts_ratio": (amounts > 100000).mean(),
                "round_amounts_ratio": (amounts % 1000 == 0).mean(),
            }
        )

        # Coefficient of variation
        features["amount_cv"] = amounts.std() / max(amounts.mean(), 1)

        return features

    def _extract_suspicious_indicators(self, acct_txns, incoming, outgoing):
        """Extract specific suspicious activity indicators"""
        features = {}

        # High frequency indicators
        features["high_frequency_flag"] = (
            1 if len(acct_txns) / self.lookback_days > 10 else 0
        )
        features["burst_activity_flag"] = (
            1 if acct_txns["txn_date"].value_counts().max() > 20 else 0
        )

        # Structuring indicators
        amounts = acct_txns["txn_amt"]
        features["potential_structuring"] = (
            (amounts >= 9000) & (amounts <= 11000)
        ).sum()
        features["consistent_amounts"] = (
            1 if amounts.std() / max(amounts.mean(), 1) < 0.1 else 0
        )

        # Rapid turnover
        if len(incoming) > 0 and len(outgoing) > 0:
            in_total = incoming["txn_amt"].sum()
            out_total = outgoing["txn_amt"].sum()
            features["rapid_turnover"] = (
                1 if abs(in_total - out_total) / (in_total + out_total) < 0.1 else 0
            )
        else:
            features["rapid_turnover"] = 0

        # Network anomalies (calculate here since we need it)
        unique_senders = incoming["from_acct"].nunique() if len(incoming) > 0 else 0
        unique_receivers = outgoing["to_acct"].nunique() if len(outgoing) > 0 else 0
        total_counterparties = unique_senders + unique_receivers
        features["high_connectivity"] = 1 if total_counterparties > 25 else 0

        return features

    def _get_zero_features(self):
        """Return zero features for accounts with no transactions"""
        return {
            # Basic patterns
            "total_transactions": 0,
            "incoming_count": 0,
            "outgoing_count": 0,
            "transaction_frequency": 0,
            "in_out_ratio": 0,
            "activity_days": 0,
            "daily_avg_transactions": 0,
            # Money flow patterns
            "total_incoming": 0,
            "total_outgoing": 0,
            "net_flow": 0,
            "flow_ratio": 0,
            "turnover_efficiency": 0,
            "avg_incoming_amount": 0,
            "avg_outgoing_amount": 0,
            "overall_avg_amount": 0,
            "amount_velocity": 0,
            # Network patterns
            "unique_senders": 0,
            "unique_receivers": 0,
            "total_unique_counterparties": 0,
            "counterparty_diversity": 0,
            "incoming_concentration": 0,
            "outgoing_dispersion": 0,
            "collection_hub_score": 0,
            # Temporal patterns
            "max_daily_transactions": 0,
            "daily_transaction_std": 0,
            "burst_intensity": 0,
            "activity_concentration": 0,
            "mobile_banking_usage": 0,
            "internet_banking_usage": 0,
            "channel_diversity": 1,
            # Amount patterns
            "amount_mean": 0,
            "amount_median": 0,
            "amount_std": 0,
            "amount_max": 0,
            "amount_min": 0,
            "amount_range": 0,
            "amount_skewness": 0,
            "small_amounts_ratio": 0,
            "medium_amounts_ratio": 0,
            "large_amounts_ratio": 0,
            "round_amounts_ratio": 0,
            "amount_cv": 0,
            # Suspicious indicators
            "high_frequency_flag": 0,
            "burst_activity_flag": 0,
            "potential_structuring": 0,
            "consistent_amounts": 0,
            "rapid_turnover": 0,
            "high_connectivity": 0,
        }

    def learn_alert_patterns(self):
        """Learn core patterns from alert accounts using clustering"""
        print("Learning core alert patterns...")

        # Extract patterns from alert accounts
        alert_patterns = self.extract_alert_patterns()

        if len(alert_patterns) < 5:
            print("Warning: Very few alert patterns available for learning")
            self.alert_patterns = alert_patterns
            return

        # Scale features for pattern analysis
        scaled_patterns = self.scaler.fit_transform(alert_patterns)

        # Use DBSCAN to find core pattern clusters
        cluster_labels = self.dbscan.fit_predict(scaled_patterns)

        # Analyze clusters
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise cluster

        print(f"Found {len(unique_labels)} core alert pattern clusters")
        print(
            f"Noise patterns: {sum(cluster_labels == -1)} out of {len(cluster_labels)}"
        )

        # Store patterns for similarity matching
        self.alert_patterns = scaled_patterns
        self.cluster_labels = cluster_labels

        # Analyze pattern characteristics
        self._analyze_pattern_characteristics(alert_patterns, cluster_labels)

    def _analyze_pattern_characteristics(self, patterns, labels):
        """Analyze characteristics of learned patterns"""
        print("\n=== Alert Pattern Analysis ===")

        for feature in [
            "total_transactions",
            "in_out_ratio",
            "collection_hub_score",
            "transaction_frequency",
            "total_unique_counterparties",
        ]:
            if feature in patterns.columns:
                values = patterns[feature]
                print(
                    f"{feature}: mean={values.mean():.2f}, std={values.std():.2f}, "
                    f"min={values.min():.2f}, max={values.max():.2f}"
                )

        # Set conservative similarity threshold based on pattern diversity
        pattern_std = np.std(self.alert_patterns, axis=0).mean()
        self.similarity_threshold = max(0.8, 1 - pattern_std)  # Very conservative
        print(f"Set similarity threshold to: {self.similarity_threshold:.3f}")

    def predict_using_pattern_matching(self) -> pd.DataFrame:
        """Predict alerts using pattern similarity matching"""
        print("Making predictions using alert pattern matching...")

        if self.alert_patterns is None:
            raise ValueError(
                "Alert patterns not learned yet. Call learn_alert_patterns() first."
            )

        batch_size = 500
        all_predictions = []

        for i in tqdm(
            range(0, len(self.df_predict), batch_size), desc="Pattern matching"
        ):
            batch_accounts = self.df_predict.iloc[i : i + batch_size]["acct"].tolist()

            batch_predictions = []
            for acct in batch_accounts:
                # Extract features for this account
                features = self._extract_comprehensive_features(acct, 121)

                # Require minimum activity
                if features["total_transactions"] < 10:
                    prediction = 0
                    similarity_score = 0.0
                else:
                    # Convert to feature vector and scale
                    feature_vector = np.array(
                        [features.get(fname, 0) for fname in self.feature_names]
                    ).reshape(1, -1)

                    # Handle NaN and infinite values
                    feature_vector = np.nan_to_num(
                        feature_vector, nan=0.0, posinf=0.0, neginf=0.0
                    )

                    scaled_features = self.scaler.transform(feature_vector)

                    # Calculate similarity to all alert patterns
                    similarities = cosine_similarity(
                        scaled_features, self.alert_patterns
                    )[0]
                    max_similarity = similarities.max()

                    # Conservative prediction based on similarity
                    prediction = 1 if max_similarity >= self.similarity_threshold else 0
                    similarity_score = max_similarity

                batch_predictions.append(
                    {
                        "acct": acct,
                        "label": prediction,
                        "similarity_score": similarity_score,
                    }
                )

            all_predictions.extend(batch_predictions)

        result_df = pd.DataFrame(all_predictions)
        alert_count = result_df["label"].sum()
        alert_rate = result_df["label"].mean()

        print(f"Predicted {alert_count} accounts as future alerts")
        print(f"Alert rate: {alert_rate:.2%}")

        if alert_count > 0:
            avg_similarity = result_df[result_df["label"] == 1][
                "similarity_score"
            ].mean()
            print(f"Average similarity score of alerts: {avg_similarity:.3f}")

        return result_df

    def save_predictions(
        self, predictions: pd.DataFrame, output_path: str = "result.csv"
    ):
        """Save predictions in competition format"""
        submission_df = predictions[["acct", "label"]].copy()
        submission_df.to_csv(output_path, index=False)
        print(f"Pattern-based predictions saved to {output_path}")

    def run_full_pipeline(
        self, data_dir: str = "data", output_path: str = "result.csv"
    ):
        """Run the complete pattern-based detection pipeline"""
        print("=== Alert Pattern Detection Pipeline ===")

        self.load_data(data_dir)
        self.learn_alert_patterns()
        predictions = self.predict_using_pattern_matching()
        self.save_predictions(predictions, output_path)

        return predictions


if __name__ == "__main__":
    detector = AlertPatternDetector(lookback_days=30, prediction_horizon=30)
    predictions = detector.run_full_pipeline()

    print(f"\n=== Final Pattern-Based Results ===")
    print(f"Total predictions: {len(predictions)}")
    print(f"Alert predictions: {predictions['label'].sum()}")
    print(f"Alert rate: {predictions['label'].mean():.2%}")
    print(
        f"High-similarity alerts (>0.9): {(predictions['similarity_score'] > 0.9).sum()}"
    )
