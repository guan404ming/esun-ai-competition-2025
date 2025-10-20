#!/usr/bin/env python3
"""
Currency Conversion Script for E.SUN AI Competition 2025
Converts transaction amounts from various currencies to TWD (NTD)
"""

import os
import pandas as pd


def get_exchange_rates():
    """
    Define exchange rates to convert various currencies to TWD (NTD)
    Note: These are approximate rates - in production, you'd want to use historical rates
    or rates from the transaction date
    """
    exchange_rates = {
        'TWD': 1.0,  # Base currency
        'USD': 31.5,  # 1 USD = 31.5 TWD
        'CNY': 4.4,   # 1 CNY = 4.4 TWD
        'JPY': 0.21,  # 1 JPY = 0.21 TWD
        'EUR': 34.2,  # 1 EUR = 34.2 TWD
        'HKD': 4.0,   # 1 HKD = 4.0 TWD
        'GBP': 39.8,  # 1 GBP = 39.8 TWD
        'AUD': 20.5,  # 1 AUD = 20.5 TWD
        'CAD': 23.1,  # 1 CAD = 23.1 TWD
        'SGD': 23.4,  # 1 SGD = 23.4 TWD
        'CHF': 35.1,  # 1 CHF = 35.1 TWD
        'NZD': 18.7,  # 1 NZD = 18.7 TWD
        'THB': 0.88,  # 1 THB = 0.88 TWD
        'ZAR': 1.75,  # 1 ZAR = 1.75 TWD
        'SEK': 2.95,  # 1 SEK = 2.95 TWD
        'MXN': 1.85,  # 1 MXN = 1.85 TWD
    }
    return exchange_rates


def show_currency_distribution(df):
    """
    Show the distribution of currencies in the dataset
    """
    print("\n=== Currency Distribution ===")
    currency_counts = df['currency_type'].value_counts()
    total_transactions = len(df)

    for currency, count in currency_counts.items():
        percentage = (count / total_transactions) * 100
        print(f"{currency}: {count:,} transactions ({percentage:.2f}%)")

    print(f"\nTotal transactions: {total_transactions:,}")
    print(f"Number of different currencies: {len(currency_counts)}")


def convert_to_twd(df):
    """
    Convert transaction amounts from various currencies to TWD
    Args:
        df: DataFrame with txn_amt and currency_type columns
    Returns:
        df: DataFrame with converted amounts in TWD
    """
    exchange_rates = get_exchange_rates()

    print("\n=== Exchange Rates Used ===")
    for currency, rate in exchange_rates.items():
        print(f"1 {currency} = {rate} TWD")

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Show original amounts for some sample currencies
    print("\n=== Sample Conversions ===")
    for currency in ['USD', 'CNY', 'JPY', 'EUR']:
        if currency in df['currency_type'].values:
            sample_row = df[df['currency_type'] == currency].iloc[0]
            original_amt = sample_row['txn_amt']
            converted_amt = original_amt * exchange_rates.get(currency, 1.0)
            print(f"{original_amt:,.2f} {currency} → {converted_amt:,.2f} TWD")

    # Convert amounts to TWD
    df['txn_amt_original'] = df['txn_amt'].copy()  # Keep original for reference
    df['original_currency'] = df['currency_type'].copy()

    df['txn_amt'] = df.apply(
        lambda row: row['txn_amt'] * exchange_rates.get(row['currency_type'], 1.0),
        axis=1
    )

    # Update currency type to TWD for all records
    df['currency_type'] = 'TWD'

    print("\n✅ Currency conversion completed!")
    return df


def main():
    """
    Main function to convert transaction data currencies to TWD
    """
    print("🔄 Starting Currency Conversion Process...")

    # File paths
    input_file = "data/acct_transaction.csv"
    output_file = "data/acct_transaction_transfered.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Please make sure the transaction data is in the data/ directory.")
        return

    print(f"📂 Loading data from: {input_file}")

    # Load the transaction data
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df):,} transactions")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Show currency distribution before conversion
    show_currency_distribution(df)

    # Convert currencies to TWD
    df_converted = convert_to_twd(df)

    # Show statistics after conversion
    print(f"\n=== Conversion Summary ===")
    total_original = df_converted['txn_amt_original'].sum()
    total_converted = df_converted['txn_amt'].sum()
    print(f"Total amount before conversion: {total_original:,.2f} (mixed currencies)")
    print(f"Total amount after conversion:  {total_converted:,.2f} TWD")

    # Save the converted data
    print(f"\n💾 Saving converted data to: {output_file}")
    try:
        df_converted.to_csv(output_file, index=False)
        print(f"✅ Successfully saved {len(df_converted):,} converted transactions")

        # Show file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"📊 Output file size: {file_size:.1f} MB")

    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return

    print("\n🎉 Currency conversion completed successfully!")
    print(f"📄 Converted data saved to: {output_file}")
    print("\nThe converted file includes:")
    print("- txn_amt: Converted amount in TWD")
    print("- currency_type: All set to 'TWD'")
    print("- txn_amt_original: Original amount (for reference)")
    print("- original_currency: Original currency type (for reference)")


if __name__ == "__main__":
    main()