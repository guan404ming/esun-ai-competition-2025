import pandas as pd
import argparse


transaction_path = "data/acct_transaction.csv"
transaction = pd.read_csv(transaction_path)


def get_transactions_by_id(id):
    return transaction[
        (transaction["from_acct"] == id) | (transaction["to_acct"] == id)
    ]


def format_transactions(df: pd.DataFrame) -> str:
    if df.empty:
        return "No records found"
    # sort by txn_date and then txn_time
    df = df.sort_values(by=["txn_date", "txn_time"])
    return df.to_string(index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    args = parser.parse_args()
    id = args.id
    result = get_transactions_by_id(id)
    print(format_transactions(result))
