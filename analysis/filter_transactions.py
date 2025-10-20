import pandas as pd


def filter_transactions_before_alert(
    alert_path="data/acct_alert.csv",
    transaction_path="data/acct_transaction.csv",
    output_path="processed_data/alert_account_history_before_alert.csv",
):
    """
    Filters transactions that occurred before an account was flagged, considering both from_acct and to_acct.

    Args:
        alert_path (str): Path to the alert data CSV file.
        transaction_path (str): Path to the transaction data CSV file.
        output_path (str): Path to save the filtered data CSV file.
    """
    # Load the datasets
    df_alert = pd.read_csv(alert_path)
    df_txn = pd.read_csv(transaction_path)

    # Prepare alert data for merging with from_acct
    df_alert_from = df_alert.rename(columns={"acct": "from_acct"})

    # Prepare alert data for merging with to_acct
    df_alert_to = df_alert.rename(columns={"acct": "to_acct"})

    # Merge with from_acct
    df_merged_from = pd.merge(df_txn, df_alert_from, on="from_acct", how="inner")
    df_merged_from["alert_acct"] = df_merged_from["from_acct"]

    # Merge with to_acct
    df_merged_to = pd.merge(df_txn, df_alert_to, on="to_acct", how="inner")
    df_merged_to["alert_acct"] = df_merged_to["to_acct"]

    # Concatenate the two merged dataframes
    df_combined = pd.concat([df_merged_from, df_merged_to], ignore_index=True)

    # Filter transactions that occurred before the alert date
    df_filtered = df_combined[df_combined["txn_date"] < df_combined["event_date"]]

    # Remove duplicate transactions
    df_filtered = df_filtered.drop_duplicates()

    # Save the filtered dataframe
    df_filtered.to_csv(output_path, index=False)
    print(f"Filtered data saved to {output_path}")


if __name__ == "__main__":
    filter_transactions_before_alert()
