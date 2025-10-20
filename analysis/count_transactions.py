import pandas as pd
import os
import argparse

os.makedirs("result", exist_ok=True)


def count_transactions(days=7, alert_only=False, latest_date=121):
    predict_path = "data/acct_predict.csv"
    transaction_path = "data/acct_transaction.csv"
    alert_path = "data/acct_alert.csv"
    output_path = f"results/transaction_counts_{days}days_alert_only_{alert_only}.csv"

    # Read the account IDs to predict
    predict_df = pd.read_csv(predict_path)
    accounts_to_track = set(predict_df["acct"])

    # Initialize transaction counts
    transaction_counts = {acct: 0 for acct in accounts_to_track}

    # Find the latest transaction date
    start_date = latest_date - days

    # Get alert accounts if needed
    alert_accounts = set()
    if alert_only:
        alert_df = pd.read_csv(alert_path)
        alert_df = alert_df[
            pd.to_datetime(alert_df["event_date"]) <= pd.to_datetime(latest_date)
        ]["acct"].values
        alert_accounts = set(alert_df)
    print(alert_accounts)

    # Process the large transaction file in chunks
    chunksize = 10**6  # 1 million rows per chunk
    for chunk in pd.read_csv(transaction_path, chunksize=chunksize):
        # Filter for recent transactions
        recent_chunk = chunk[chunk["txn_date"] >= start_date]

        if alert_only:
            # Filter for transactions involving at least one alert account
            recent_chunk = recent_chunk[
                (recent_chunk["from_acct"].isin(alert_accounts))
                | (recent_chunk["to_acct"].isin(alert_accounts))
            ]

        # Check 'from_acct'
        from_counts = (
            recent_chunk[recent_chunk["from_acct"].isin(accounts_to_track)]["from_acct"]
            .value_counts()
            .to_dict()
        )
        for acct, count in from_counts.items():
            transaction_counts[acct] += count

        # Check 'to_acct'
        to_counts = (
            recent_chunk[recent_chunk["to_acct"].isin(accounts_to_track)]["to_acct"]
            .value_counts()
            .to_dict()
        )
        for acct, count in to_counts.items():
            transaction_counts[acct] += count

    # Convert the counts to a DataFrame and save
    result_df = pd.DataFrame(
        list(transaction_counts.items()), columns=["acct", "transaction_count"]
    )
    result_df.to_csv(output_path, index=False)
    print(f"Transaction counts saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of recent days to consider for transactions.",
    )
    parser.add_argument(
        "--alert-only",
        action="store_true",
        help="Count only transactions involving alert accounts.",
    )
    parser.add_argument(
        "--latest-date",
        default=121,
    )
    args = parser.parse_args()
    count_transactions(
        days=args.days, alert_only=args.alert_only, latest_date=args.latest_date
    )
