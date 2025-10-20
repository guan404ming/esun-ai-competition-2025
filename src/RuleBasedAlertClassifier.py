import pandas as pd
from typing import Dict
import os
from tqdm import tqdm


class RuleBasedAlertClassifier:
    """
    Time series-based alert account classifier that analyzes transaction patterns
    over configurable time intervals to predict future alert accounts.

    Based on aggregate analysis of 1004 alert accounts:
    - Incoming/Outgoing ratio: 1.66 (collection hub behavior)
    - Average 33.9 transactions per account (3.2 per day)
    - Average 17.4 unique counterparties per account
    - Alert timing: average Day 68.5, median Day 74.0
    - Net positive flow of $5.9M across all alert accounts
    - 45 accounts had connections to other alert accounts
    """

    def __init__(self, interval_days: int = 3):
        """
        Initialize the classifier with configurable time interval.

        Args:
            interval_days: Number of days to look back for pattern analysis
        """
        self.interval_days = interval_days
        self.df_transaction = None
        self.df_alert = None
        self.df_predict = None
        self.alert_dates = {}
        self.account_transactions = {}  # Pre-filtered transactions per account

    def load_data(self, data_dir: str = "data"):
        """Load transaction and alert data"""
        print(f"Loading data from {data_dir}...")

        # Load transaction data
        transaction_path = os.path.join(data_dir, "acct_transaction.csv")
        self.df_transaction = pd.read_csv(transaction_path)

        # Load alert data
        alert_path = os.path.join(data_dir, "acct_alert.csv")
        self.df_alert = pd.read_csv(alert_path)

        # Load prediction targets
        predict_path = os.path.join(data_dir, "acct_predict.csv")
        self.df_predict = pd.read_csv(predict_path)

        # Create alert date lookup
        self.alert_dates = dict(zip(self.df_alert["acct"], self.df_alert["event_date"]))

        print(f"Loaded {len(self.df_transaction)} transactions")
        print(f"Loaded {len(self.df_alert)} alert accounts")
        print(f"Loaded {len(self.df_predict)} accounts to predict")

        # Pre-filter transactions for each prediction account for speed
        self._preprocess_account_transactions()

    def _preprocess_account_transactions(self):
        """
        Pre-filter and store all transactions for each prediction account.
        This dramatically speeds up the time series analysis.
        """
        print("Preprocessing account transactions for faster lookup...")

        predict_accounts = set(self.df_predict["acct"])

        # Filter to only transactions involving prediction accounts (much faster)
        from_mask = self.df_transaction["from_acct"].isin(predict_accounts)
        to_mask = self.df_transaction["to_acct"].isin(predict_accounts)
        relevant_txns = self.df_transaction[from_mask | to_mask].copy()

        print(
            f"Filtered to {len(relevant_txns)} relevant transactions from {len(self.df_transaction)} total"
        )

        # Group by account for fast lookup
        for acct in tqdm(predict_accounts, desc="Processing accounts"):
            # Get all transactions involving this account
            acct_mask = (relevant_txns["from_acct"] == acct) | (
                relevant_txns["to_acct"] == acct
            )

            account_txns = relevant_txns[acct_mask]

            if not account_txns.empty:
                # Sort by transaction date for efficient time-based filtering
                account_txns = account_txns.sort_values("txn_date")
                # Store the filtered transactions
                self.account_transactions[acct] = account_txns
            else:
                self.account_transactions[acct] = pd.DataFrame()

        print(f"Preprocessed transactions for {len(predict_accounts)} accounts")

    def extract_features_for_interval(self, acct: str, end_day: int) -> Dict:
        """
        Extract features for an account during a specific time interval.

        Args:
            acct: Account ID
            end_day: End day of the interval (exclusive)

        Returns:
            Dictionary of features
        """
        start_day = max(1, end_day - self.interval_days)

        # Use pre-filtered transactions for this account (much faster!)
        account_txns = self.account_transactions.get(acct, pd.DataFrame())

        if account_txns.empty:
            return self._get_zero_features()

        # Filter by time window only (account filter already applied)
        time_mask = (account_txns["txn_date"] >= start_day) & (
            account_txns["txn_date"] < end_day
        )
        interval_txns = account_txns[time_mask]

        if len(interval_txns) == 0:
            return self._get_zero_features()

        # Separate incoming and outgoing transactions
        incoming = interval_txns[interval_txns["to_acct"] == acct]
        outgoing = interval_txns[interval_txns["from_acct"] == acct]

        features = {
            # Basic transaction counts
            "total_txns": len(interval_txns),
            "incoming_txns": len(incoming),
            "outgoing_txns": len(outgoing),
            "receive_send_ratio": len(incoming) / max(len(outgoing), 1),
            # Amount features
            "total_amount": interval_txns["txn_amt"].sum(),
            "avg_amount": interval_txns["txn_amt"].mean(),
            "max_amount": interval_txns["txn_amt"].max(),
            "amount_std": interval_txns["txn_amt"].std()
            if len(interval_txns) > 1
            else 0,
            # Incoming amount features
            "incoming_amount": incoming["txn_amt"].sum() if len(incoming) > 0 else 0,
            "avg_incoming_amount": incoming["txn_amt"].mean()
            if len(incoming) > 0
            else 0,
            # Outgoing amount features
            "outgoing_amount": outgoing["txn_amt"].sum() if len(outgoing) > 0 else 0,
            "avg_outgoing_amount": outgoing["txn_amt"].mean()
            if len(outgoing) > 0
            else 0,
            # Network features
            "unique_senders": incoming["from_acct"].nunique()
            if len(incoming) > 0
            else 0,
            "unique_receivers": outgoing["to_acct"].nunique()
            if len(outgoing) > 0
            else 0,
            "network_diversity": (
                incoming["from_acct"].nunique() + outgoing["to_acct"].nunique()
            ),
            # Channel features
            "mobile_banking_ratio": (interval_txns["channel_type"] == "03").mean(),
            "internet_banking_ratio": (interval_txns["channel_type"] == "04").mean(),
            "unknown_channel_ratio": (interval_txns["channel_type"] == "UNK").mean(),
            # High frequency indicators
            "txns_per_day": len(interval_txns) / self.interval_days,
            "high_freq_flag": 1 if len(interval_txns) / self.interval_days > 10 else 0,
            # Small amount indicators (based on analysis showing median ~3-4K)
            "small_txn_ratio": (interval_txns["txn_amt"] <= 5000).mean(),
            "large_txn_count": (interval_txns["txn_amt"] >= 100000).sum(),
            # Time-based features
            "days_active": interval_txns["txn_date"].nunique(),
            "activity_concentration": interval_txns["txn_date"].nunique()
            / self.interval_days,
        }

        return features

    def _get_zero_features(self) -> Dict:
        """Return zero features when no transactions found"""
        return {
            "total_txns": 0,
            "incoming_txns": 0,
            "outgoing_txns": 0,
            "receive_send_ratio": 0,
            "total_amount": 0,
            "avg_amount": 0,
            "max_amount": 0,
            "amount_std": 0,
            "incoming_amount": 0,
            "avg_incoming_amount": 0,
            "outgoing_amount": 0,
            "avg_outgoing_amount": 0,
            "unique_senders": 0,
            "unique_receivers": 0,
            "network_diversity": 0,
            "mobile_banking_ratio": 0,
            "internet_banking_ratio": 0,
            "unknown_channel_ratio": 0,
            "txns_per_day": 0,
            "high_freq_flag": 0,
            "small_txn_ratio": 0,
            "large_txn_count": 0,
            "days_active": 0,
            "activity_concentration": 0,
        }

    def apply_alert_rules(self, features: Dict) -> bool:
        """
        Apply rule-based logic to determine if account should be flagged as alert.

        Based on aggregate analysis of 1004 alert accounts:
        - Incoming/Outgoing ratio: 1.66 (collection hub behavior)
        - Average 33.9 transactions per account (3.2 per day)
        - Average 17.4 unique counterparties
        - Alert timing: average Day 68.5, median Day 74.0
        """

        # Rule 1: Collection hub pattern (based on actual 1.66 ratio)
        collection_hub_score = 0
        if features["receive_send_ratio"] > 1.8:  # Above typical alert pattern
            collection_hub_score += 2
        elif features["receive_send_ratio"] > 1.4:  # Near alert pattern
            collection_hub_score += 1

        # Rule 2: Transaction frequency (based on actual 3.2 txns/day average)
        frequency_score = 0
        if features["txns_per_day"] > 5.0:  # Well above alert average
            frequency_score += 2
        elif features["txns_per_day"] > 3.0:  # Near alert average
            frequency_score += 1

        # Rule 3: Network diversity (based on actual 17.4 average)
        network_score = 0
        if features["network_diversity"] > 25:  # Well above alert average
            network_score += 2
        elif features["network_diversity"] > 15:  # Near alert average
            network_score += 1

        # Rule 4: Transaction volume pattern (based on actual 33.9 avg)
        volume_score = 0
        if features["total_txns"] > 50:  # Well above alert average
            volume_score += 2
        elif features["total_txns"] > 25:  # Near alert average
            volume_score += 1

        # Rule 5: Small amount dominance pattern
        amount_pattern_score = 0
        if features["small_txn_ratio"] > 0.7 and features["avg_amount"] < 10000:
            amount_pattern_score += 1
        if features["large_txn_count"] > 0:  # Has some large transactions
            amount_pattern_score += 1

        # Rule 6: High mobile banking usage (digital channel preference)
        channel_score = 0
        if features["mobile_banking_ratio"] > 0.6:  # High mobile banking usage
            channel_score += 1

        # Rule 7: Activity concentration (burst pattern)
        concentration_score = 0
        if features["activity_concentration"] < 0.5 and features["total_txns"] > 10:
            concentration_score += 1  # High activity in few days

        # Calculate total score
        total_score = (
            collection_hub_score
            + frequency_score
            + network_score
            + volume_score
            + amount_pattern_score
            + channel_score
            + concentration_score
        )

        # Threshold for alert classification (adjusted based on more rules)
        # Require at least 6 points from 9 possible to flag as alert
        return total_score >= 7

    def predict_time_series(self) -> pd.DataFrame:
        """
        Main prediction method that implements time series analysis.

        Returns:
            DataFrame with account predictions
        """
        print(
            f"Starting time series prediction with {self.interval_days}-day intervals..."
        )

        # Initialize prediction list with default 0 (non-alert)
        predictions = []
        total_accounts = len(self.df_predict)

        for idx, (_, row) in enumerate(
            tqdm(
                self.df_predict.iterrows(),
                total=total_accounts,
                desc="Predicting alerts",
            )
        ):
            acct = row["acct"]
            is_alert = 0  # Default to non-alert

            # Quick check: if account has no transactions at all, skip
            if (
                acct not in self.account_transactions
                or self.account_transactions[acct].empty
            ):
                predictions.append({"acct": acct, "label": is_alert})
                continue

            # Loop from day 3 to day 121 (as specified)
            for day in range(self.interval_days, 122):
                # Extract features for current interval
                features = self.extract_features_for_interval(acct, day)

                # Apply rules to determine if should be flagged
                if self.apply_alert_rules(features):
                    is_alert = 1
                    print(f"Account {acct[:12]}... flagged as alert at day {day}")
                    break  # Once flagged, stop checking

            predictions.append({"acct": acct, "label": is_alert})

            # Update progress bar description with current alert count
            if (idx + 1) % 100 == 0:
                alerts_so_far = sum(p["label"] for p in predictions)
                tqdm.write(f"Found {alerts_so_far} alerts so far")

        result_df = pd.DataFrame(predictions)
        print(
            f"Prediction completed. Flagged {result_df['label'].sum()} accounts as alerts"
        )
        return result_df

    def save_predictions(
        self, predictions: pd.DataFrame, output_path: str = "result.csv"
    ):
        """Save predictions to CSV file"""
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    def run_full_pipeline(
        self, data_dir: str = "data", output_path: str = "result.csv"
    ):
        """Run the complete prediction pipeline"""
        self.load_data(data_dir)
        predictions = self.predict_time_series()
        self.save_predictions(predictions, output_path)
        return predictions


if __name__ == "__main__":
    # Example usage - using 3-day intervals based on data analysis
    classifier = RuleBasedAlertClassifier(interval_days=3)
    predictions = classifier.run_full_pipeline()

    print(f"Total predictions: {len(predictions)}")
    print(f"Alert predictions: {predictions['label'].sum()}")
    print(f"Non-alert predictions: {(predictions['label'] == 0).sum()}")
