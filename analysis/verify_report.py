import pandas as pd

def verify_report_data(
    data_path="processed_data/alert_account_history_before_alert_v2.csv",
    alert_path="data/acct_alert.csv",
):
    """
    Verifies the data in the time_series_report.md file.

    Args:
        data_path (str): Path to the filtered transaction data.
        alert_path (str): Path to the alert data.
    """
    df = pd.read_csv(data_path)
    df_alert = pd.read_csv(alert_path)
    alert_accounts = set(df_alert["acct"])

    # 1.1 Overall Statistics
    total_transactions = len(df)
    unique_alert_accounts = df_alert["acct"].nunique()
    avg_transactions_per_account = total_transactions / unique_alert_accounts

    print("--- 1.1 Overall Statistics ---")
    print(f"Total transactions: {total_transactions}")
    print(f"Unique alert accounts: {unique_alert_accounts}")
    print(f"Average transactions per account: {avg_transactions_per_account:.2f}")

    # 1.2 Transaction Amount Analysis
    txn_amt_stats = df["txn_amt"].describe()
    print("\n--- 1.2 Transaction Amount Analysis ---")
    print(txn_amt_stats)

    # 1.3 Transaction Direction Analysis
    as_payer = df[df['from_acct'].isin(alert_accounts)]
    as_receiver = df[df['to_acct'].isin(alert_accounts)]
    print("\n--- 1.3 Transaction Direction Analysis ---")
    print(f"As payer: {len(as_payer)} ({len(as_payer)/total_transactions:.2%})")
    print(f"As receiver: {len(as_receiver)} ({len(as_receiver)/total_transactions:.2%})")

    # 1.4 Transaction Channel Analysis
    channel_distribution = df["channel_type"].value_counts(normalize=True)
    print("\n--- 1.4 Transaction Channel Analysis ---")
    print(channel_distribution)

    # 1.5 Preliminary Time and Object Analysis
    df["days_before_alert"] = df["event_date"] - df["txn_date"]
    avg_days_before_alert = df["days_before_alert"].mean()
    transactions_within_7_days = df[df["days_before_alert"] <= 7]
    percentage_within_7_days = len(transactions_within_7_days) / total_transactions

    print("\n--- 1.5 Preliminary Time and Object Analysis ---")
    print(f"Average transaction time before alert: {avg_days_before_alert:.2f} days")
    print(f"Transactions within 7 days of alert: {percentage_within_7_days:.2%}")

    # 2.1 Time Series Analysis
    print("\n--- 2.1 Time Series Analysis ---")
    df_last_15_days = df[df["days_before_alert"] <= 15]
    daily_summary = df_last_15_days.groupby("days_before_alert").agg(
        daily_txn_count=("txn_amt", "count"),
        daily_txn_volume=("txn_amt", "sum")
    ).sort_index(ascending=False)
    daily_summary["daily_txn_volume"] = daily_summary["daily_txn_volume"] / 10000 # in 10,000s
    daily_summary["7_day_rolling_avg_count"] = daily_summary["daily_txn_count"].rolling(window=7, min_periods=1).mean()
    print(daily_summary)

    # 2.2 Propagation Chain Analysis
    print("\n--- 2.2 Propagation Chain Analysis ---")
    fan_in = df[df['to_acct'].isin(alert_accounts)].groupby('to_acct')['from_acct'].nunique()
    fan_out = df[df['from_acct'].isin(alert_accounts)].groupby('from_acct')['to_acct'].nunique()

    fan_in_dist = pd.cut(fan_in, bins=[0, 1, 5, 10, 50, 100, 1000], labels=["1", "2-5", "6-10", "11-50", "51-100", "100+"])
    fan_out_dist = pd.cut(fan_out, bins=[0, 1, 5, 10, 50, 100, 1000], labels=["1", "2-5", "6-10", "11-50", "51-100", "100+"])

    print("\nFan-in Distribution")
    print(fan_in_dist.value_counts().sort_index())

    print("\nFan-out Distribution")
    print(fan_out_dist.value_counts().sort_index())

if __name__ == "__main__":
    verify_report_data()
