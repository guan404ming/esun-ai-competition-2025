# Exploratory Data Analysis (EDA)

## 2025 E.SUN AI Challenge - Fraudulent Alert Account Detection

**Competition Link:** https://tbrain.trendmicro.com.tw/Competitions/Details/40

**Date:** 2025-10-20

---

## 1. Dataset Overview

### 1.1 Data Schema

Based on `資料欄位說明_V2.xlsx`:

**acct_transaction.csv (Transaction Records)**
| Field | Chinese Name | Data Type | Description |
|-------|--------------|-----------|-------------|
| from_acct | 匯款帳戶 | Categorical | Sender account (hashed) |
| from_acct_type | 匯款帳戶是否為玉山帳戶 | Categorical | 01: E.SUN; 02: Other financial institution |
| to_acct | 收款帳戶 | Categorical | Receiver account (hashed) |
| to_acct_type | 收款帳戶是否為玉山帳戶 | Categorical | 01: E.SUN; 02: Other financial institution |
| is_self_txn | 交易雙方是否為同一人 | Categorical | Y: Same person; N: Different people; UNK: Unknown/null |
| txn_amt | 交易金額 | Numeric | Transaction amount (original currency, not converted to TWD) |
| txn_date | 交易日期 | Categorical | Transaction date (normalized, day 1 onwards) |
| txn_time | 交易時間 | Numeric | Transaction time |
| currency_type | 幣別 | Categorical | Currency type (TWD, USD, etc.) |
| channel_type | 交易通路 | Categorical | Transaction channel (see channel codes below) |

**Channel Type Codes:**
- 01: ATM
- 02: Branch counter (臨櫃)
- 03: Mobile banking (行動銀行)
- 04: Internet banking (網路銀行)
- 05: Voice
- 06: eATM
- 07: E-payment
- 99: System scheduled transaction
- UNK: Original data channel is null

**acct_alert.csv (Alert Account List)**
| Field | Chinese Name | Data Type | Description |
|-------|--------------|-----------|-------------|
| acct | 玉山帳戶 | Categorical | E.SUN account (all are alert accounts, label=1) |
| event_date | 警示日期 | Categorical | Alert date (normalized, day 1 onwards) |

**acct_predict.csv (Prediction Target)**
| Field | Chinese Name | Data Type | Description |
|-------|--------------|-----------|-------------|
| acct | 玉山帳戶 | Categorical | E.SUN account to predict |
| label | 預測結果 | Categorical | 1: Predicted as alert account; 0: Predicted as normal |

### 1.2 Dataset Statistics

| Dataset | Rows | Memory | Description |
|---------|------|--------|-------------|
| acct_transaction.csv | 4,435,890 | 1,987.83 MB | Complete transaction history |
| acct_alert.csv | 1,004 | < 1 MB | Known alert accounts |
| acct_predict.csv | 4,780 | < 1 MB | Accounts to classify |

**Key Observations:**
- All accounts in `acct_predict.csv` exist in transaction data (100% coverage)
- All alert accounts exist in transaction data (100% coverage)
- All alert accounts are E.SUN accounts (100%)
- All prediction accounts are E.SUN accounts (100%)
- No overlap between alert and prediction accounts (0 common accounts)

---

## 2. Transaction Data Analysis

### 2.1 Basic Statistics

**Temporal Coverage:**
- Date range: Day 1 to Day 121 (121 days ≈ 4 months)
- Total transactions: 4,435,890
- Average transactions per day: 36,660
- Transaction patterns show variability across days (ranging from ~20K to ~82K per day)

**Account Statistics:**
- Unique sender accounts (from_acct): 819,399
- Unique receiver accounts (to_acct): 1,169,482
- Total unique accounts: 1,800,106
- E.SUN accounts in transactions: 333,768
  - As senders: 231,092
  - As receivers: 201,472

### 2.2 Transaction Amount Distribution

| Statistic | Value (TWD) |
|-----------|-------------|
| Mean | 40,819.73 |
| Std Dev | 253,975.30 |
| Min | 5.00 |
| 25th percentile | 1,350.00 |
| 50th percentile (Median) | 4,850.00 |
| 75th percentile | 18,500.00 |
| 90th percentile | 50,500.00 |
| 95th percentile | 105,000.00 |
| 99th percentile | 645,000.00 |
| Max | 5,000,000.00 |

**Key Insights:**
- **Highly right-skewed distribution:** Mean (40,820) >> Median (4,850)
- **Large transactions:** 32,501 transactions ≥ 1M TWD (0.73% of all transactions)
  - These represent 81.15 billion TWD in total
- **Small transactions:** 91,004 transactions ≤ 100 TWD (2.05%)

### 2.3 Account Type Distribution

**Sender Account Types (from_acct_type):**
| Type | Count | Percentage |
|------|-------|------------|
| 01 (E.SUN) | 2,988,819 | 67.38% |
| 02 (Other) | 1,447,071 | 32.62% |

**Receiver Account Types (to_acct_type):**
| Type | Count | Percentage |
|------|-------|------------|
| 01 (E.SUN) | 2,404,671 | 54.21% |
| 02 (Other) | 2,031,219 | 45.79% |

**Interpretation:**
- More transactions originate from E.SUN accounts (67%) than are received by them (54%)
- This suggests E.SUN customers frequently transfer to external banks

### 2.4 Self-Transaction Analysis

| Category | Count | Percentage |
|----------|-------|------------|
| UNK (Unknown) | 3,346,272 | 75.44% |
| N (Different people) | 943,713 | 21.27% |
| Y (Same person) | 145,905 | 3.29% |

**Key Insight:**
- **75% of transactions have unknown ownership relationship** - major data quality issue
- Only 3.3% are confirmed self-transfers
- This feature may have limited predictive power due to high UNK rate

### 2.5 Currency Distribution

| Currency | Count | Percentage |
|----------|-------|------------|
| TWD (New Taiwan Dollar) | 4,382,571 | 98.80% |
| USD | 23,458 | 0.53% |
| JPY | 21,545 | 0.49% |
| CNY | 2,888 | 0.07% |
| EUR | 2,233 | 0.05% |
| Others (HKD, AUD, GBP, CAD, NZD, THB, ZAR, SGD, CHF, SEK, MXN) | 3,195 | 0.07% |

**Insight:** TWD dominates (98.8%), but there are 15+ different currencies represented

### 2.6 Channel Distribution

| Channel | Code | Count | Percentage |
|---------|------|-------|------------|
| Mobile banking | 03 | 2,008,166 | 45.27% |
| Internet banking | 04 | 595,522 | 13.43% |
| ATM | 01 | 51,364 | 1.16% |
| eATM | 06 | 58,445 | 1.32% |
| Branch counter | 02 | 15,458 | 0.35% |
| Voice | 05 | 3,505 | 0.08% |
| E-payment | 07 | 410 | 0.01% |
| System scheduled | 99 | 1,281 | 0.03% |
| **UNK (Unknown)** | UNK | **1,701,739** | **38.37%** |

**Key Insights:**
- **Mobile banking is the most popular channel** (45%)
- **38% of transactions have unknown channel** - another significant data quality issue
- Digital channels (mobile + internet) represent 59% of known-channel transactions

### 2.7 Transaction Network Patterns

**Accounts as Senders (All Accounts):**
| Metric | Value |
|--------|-------|
| Mean transactions per account | 5.41 |
| Median transactions per account | 1 |
| Max transactions by single account | 1,462 |
| Mean unique recipients per sender | 2.72 |
| Max unique recipients | 497 |
| Mean total sent | 220,981 TWD |
| Max total sent | 693.7 million TWD |

**Interpretation:**
- **Highly concentrated activity:** Median is only 1 transaction, but some accounts have 1,400+ transactions
- **Power law distribution:** Most accounts have minimal activity, few accounts are extremely active
- Top account sent 693.7M TWD across 733 transactions to multiple recipients

---

## 3. Alert Account Analysis

### 3.1 Alert Account Characteristics

**Basic Statistics:**
- Total alert accounts: 1,004
- All are E.SUN accounts (100%)
- Alert dates range: Day 1 to Day 121
- Mean alert date: Day 68.5
- Median alert date: Day 74

**Alert Date Distribution:**
- Alerts are spread throughout the 121-day period
- Slight concentration in middle to later period (median at day 74)
- Weekly distribution shows alerts occur consistently over time

### 3.2 Alert Account Transaction Patterns

**As Senders:**
- Total transactions: 12,797 (0.29% of all transactions)
- Mean amount: 25,003.58 TWD
- Median amount: 3,050 TWD
- Max amount: 3,050,000 TWD

**As Receivers:**
- Total transactions: 21,218 (0.48% of all transactions)
- Mean amount: 15,358.20 TWD
- Median amount: 4,050 TWD
- Max amount: 2,050,000 TWD

**Channel Distribution (Alert Accounts as Senders):**
| Channel | Count | Percentage |
|---------|-------|------------|
| Mobile banking (03) | 10,106 | 78.97% |
| UNK | 1,762 | 13.77% |
| Internet banking (04) | 710 | 5.55% |
| ATM (01) | 202 | 1.58% |
| Branch counter (02) | 13 | 0.10% |
| System scheduled (99) | 4 | 0.03% |

**Self-Transaction Pattern (Alert Accounts as Senders):**
| Category | Count | Percentage |
|----------|-------|------------|
| UNK | 11,372 | 88.87% |
| N (Different) | 1,196 | 9.35% |
| Y (Same) | 229 | 1.79% |

**Currency Distribution (Alert Accounts as Senders):**
- TWD: 12,725 (99.44%)
- USD: 38
- JPY: 22
- Others: 12

**Key Insights:**
- Alert accounts **receive more than they send** (21K vs 13K transactions)
- Alert accounts primarily use **mobile banking** (79%)
- Transaction amounts are **lower than overall average** (median 3K-4K vs 4.85K)
- **Predominantly TWD transactions** (99.4%)
- **High UNK rate for self-transaction** (89%)

---

## 4. Prediction Account Analysis

### 4.1 Prediction Set Characteristics

**Basic Statistics:**
- Total accounts to predict: 4,780
- All are E.SUN accounts (100%)
- All appear in transaction data (100% coverage)
- Accounts with sending transactions: 4,541 (95.0%)
- Accounts with receiving transactions: 4,624 (96.7%)
- Accounts with no transactions: 0

### 4.2 Prediction Account Transaction Patterns

**As Senders:**
- Total transactions: 195,502 (4.41% of all transactions)
- Mean amount: 49,044.16 TWD
- Median amount: 5,050 TWD
- Max amount: 5,000,000 TWD
- **Transactions per account:** ~43 on average

**As Receivers:**
- Total transactions: 176,277 (3.97% of all transactions)
- Mean amount: 44,104.89 TWD
- Median amount: 4,050 TWD
- Max amount: 5,000,000 TWD
- **Transactions per account:** ~37 on average

**Channel Distribution (Prediction Accounts as Senders):**
| Channel | Count | Percentage |
|---------|-------|------------|
| Mobile banking (03) | 136,282 | 69.71% |
| Internet banking (04) | 35,807 | 18.32% |
| UNK | 20,799 | 10.64% |
| ATM (01) | 1,536 | 0.79% |
| Branch counter (02) | 853 | 0.44% |
| System scheduled (99) | 114 | 0.06% |
| Voice (05) | 111 | 0.06% |

**Channel Distribution (Prediction Accounts as Receivers):**
| Channel | Count | Percentage |
|---------|-------|------------|
| UNK | 128,537 | 72.92% |
| Mobile banking (03) | 32,176 | 18.25% |
| Internet banking (04) | 12,715 | 7.21% |
| ATM (01) | 1,367 | 0.78% |
| Branch counter (02) | 989 | 0.56% |
| E-payment (07) | 364 | 0.21% |
| Voice (05) | 92 | 0.05% |
| System scheduled (99) | 37 | 0.02% |

**Key Differences from Alert Accounts:**
| Metric | Alert Accounts | Prediction Accounts | Observation |
|--------|----------------|---------------------|-------------|
| Transactions as sender | 12,797 (avg: 12.7 per acct) | 195,502 (avg: 40.9 per acct) | Prediction accounts **3.2x more active** |
| Transactions as receiver | 21,218 (avg: 21.1 per acct) | 176,277 (avg: 36.9 per acct) | Prediction accounts **1.7x more active** |
| Send/Receive ratio | 0.60 (receive > send) | 1.11 (send > receive) | **Opposite pattern** |
| Median send amount | 3,050 TWD | 5,050 TWD | Prediction accounts send **larger amounts** |
| UNK channel (as sender) | 13.77% | 10.64% | Prediction set has **better data quality** |

---

## 5. Key Findings and Implications

### 5.1 Data Quality Issues

1. **High UNK rates:**
   - 38% of transactions have unknown channel
   - 75% have unknown self-transaction status
   - **Impact:** These features may have limited discriminative power

2. **Complete coverage:**
   - All alert and prediction accounts exist in transaction data
   - **Impact:** No cold-start problem for prediction

### 5.2 Class Imbalance

- Alert accounts: 1,004
- Prediction accounts: 4,780
- E.SUN accounts in transactions: 333,768
- **Estimated positive class rate:** ~0.3-0.5% (severe imbalance)
- **Implication:** Need careful handling of class imbalance in modeling

### 5.3 Temporal Considerations

- Data spans 121 days (≈4 months)
- Alert dates are spread throughout this period
- **Critical question:** Are we predicting future alerts based on past behavior, or current alerts based on concurrent behavior?
- **Recommendation:** Need to understand the prediction time horizon to properly construct train/validation splits

### 5.4 Behavioral Patterns

**Alert Account Signature:**
- Lower transaction volumes (13K send, 21K receive total)
- **Receive more than send** (ratio 0.60)
- Smaller transaction amounts (median 3-4K vs 4.85K overall)
- Higher mobile banking usage (79% vs 45% overall)
- Minimal foreign currency usage (99.4% TWD)

**Prediction Account Profile:**
- Higher transaction volumes (196K send, 176K receive total)
- **Send more than receive** (ratio 1.11)
- Larger transaction amounts (median 5K)
- Moderate mobile banking usage (70%)

**Hypothesis:** Alert accounts may exhibit **receiving behavior** characteristic of fraud targets, while prediction accounts show more **normal sending behavior**

### 5.5 Feature Engineering Opportunities

**Transaction Volume Features:**
1. Total send/receive counts
2. Unique sender/receiver counts
3. Send/receive ratio
4. Transaction frequency over time windows

**Transaction Amount Features:**
5. Sum, mean, median, std, min, max of amounts (send/receive separately)
6. Large transaction counts (>1M, >100K thresholds)
7. Small transaction counts (<100 threshold)
8. Amount volatility

**Temporal Features:**
9. Active days count
10. Days since first/last transaction
11. Transaction velocity (transactions per day)
12. Time-of-day patterns
13. Day-of-week patterns (if applicable)
14. Temporal clustering (bursty vs. regular)

**Network Features:**
15. Recipient diversity (unique recipients / total sends)
16. Sender diversity (unique senders / total receives)
17. Self-transaction rate
18. E.SUN vs. external bank ratios

**Channel & Currency Features:**
19. Channel diversity
20. Mobile banking ratio
21. Foreign currency usage
22. UNK rate (as a data quality indicator)

**Behavioral Ratios:**
23. Send/receive transaction count ratio
24. Send/receive amount ratio
25. Inbound/outbound network complexity
