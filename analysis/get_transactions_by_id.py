import pandas as pd
import argparse


transaction_path = "data/acct_transaction.csv"
transaction = pd.read_csv(transaction_path)
alert_acct_path = "data/acct_alert.csv"
alert_acct = pd.read_csv(alert_acct_path)
alert_acct_set = set(alert_acct["acct"])


def get_transactions_by_id(id):
    return transaction[
        (transaction["from_acct"] == id) | (transaction["to_acct"] == id)
    ].copy()


def format_transactions(df: pd.DataFrame) -> str:
    if df.empty:
        return "No records found"
    # sort by txn_date and then txn_time
    df = df.sort_values(by=["txn_date", "txn_time"])
    return df.to_string(index=False)


def add_is_alert_acct(df: pd.DataFrame, id: str) -> pd.DataFrame:
    if df.empty:
        return df
    df["is_alert_acct"] = df.apply(
        lambda row: (row["to_acct"] if row["from_acct"] == id else row["from_acct"])
        in alert_acct_set,
        axis=1,
    )
    return df


def get_max_daily_transactions(df: pd.DataFrame):
    if df.empty:
        return "No transactions to analyze."
    daily_counts = df.groupby("txn_date").size()
    if daily_counts.empty:
        return "No transactions to analyze."
    max_transactions_date = daily_counts.idxmax()
    max_transactions_count = daily_counts.max()
    df_on_max_date = df[df["txn_date"] == max_transactions_date]
    avg_txn_amt = df_on_max_date["txn_amt"].mean()
    return f"Maximum daily transactions: {max_transactions_count} on day {max_transactions_date}, with an average transaction amount of {avg_txn_amt:.2f}"


def get_all_transaction_intervals(df: pd.DataFrame):
    if len(df) < 2:
        return "Not enough transactions to calculate intervals."

    df_sorted = df.sort_values(by=["txn_date", "txn_time"]).copy()
    df_sorted["datetime"] = pd.to_datetime(
        "2024-"
        + df_sorted["txn_date"].astype(str)
        + " "
        + df_sorted["txn_time"].astype(str),
        format="%Y-%j %H:%M:%S",
        errors="coerce",
    )
    time_diffs = df_sorted["datetime"].diff().dt.total_seconds().dropna()
    all_intervals = time_diffs.tolist()

    if not all_intervals:
        return "No transaction intervals to show."

    if len(all_intervals) > 3:
        return f"All transaction intervals (seconds): {sorted(all_intervals)}\nTop 3 avg transaction amount: {sum(sorted(all_intervals)[:3]) / 3:.2f}\n"

    return f"All transaction intervals (seconds): {sorted(all_intervals)}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    args = parser.parse_args()
    id = args.id
    result = get_transactions_by_id(id)

    # 1. 單日最高交易次數, 在哪一天
    max_daily_info = get_max_daily_transactions(result)
    print(max_daily_info)

    # 2. 列出所有交易間隔
    all_intervals_info = get_all_transaction_intervals(result)
    print(all_intervals_info)

    # 3. check 跟 id 的對手 id 是否為 alert account
    result_with_alert_info = add_is_alert_acct(result, id)

    print("\n--- Transactions ---")
    print(format_transactions(result_with_alert_info))
