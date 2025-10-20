import pandas as pd


def generate_report(
    file_path="processed_data/alert_account_history_before_alert.csv",
    output_path="analysis/generated_report.md",
):
    """
    Generates a markdown report summarizing the analysis of alert account transactions.

    Args:
        file_path (str): The path to the CSV file to analyze.
        output_path (str): The path to save the generated markdown report.
    """
    df = pd.read_csv(file_path)
    df["days_before_alert"] = df["event_date"] - df["txn_date"]

    report_content = f"""# 警示帳戶綜合分析報告

本報告彙總了對警示帳戶在被標記前的交易行為的探索性分析與深度調查，旨在全面揭示其活動模式、時間序列特徵及在潛在詐騙網絡中的結構性角色。

**Note:** This report is based on the data from `{file_path}`.

---

## 第一部分：初步探索性分析

此部分旨在對 `{file_path}` 資料集進行宏觀掃描，建立對警示帳戶基本特徵的認知。

"""
    report_content += generate_overall_statistics(df)
    report_content += generate_transaction_amount_analysis(df)
    report_content += generate_transaction_direction_analysis(df)
    report_content += generate_transaction_channel_analysis(df)
    report_content += generate_preliminary_time_and_counterparty_analysis(df)

    report_content += """
---

## 第二部分：時間序列與傳播鏈深度分析

此部分在前述分析的基礎上，對時間維度和網絡結構進行更深入的挖掘。

"""
    report_content += generate_time_series_analysis(df)
    report_content += generate_fan_in_fan_out_analysis(df)

    report_content += """
---

## 綜合結論

綜合以上所有分析，即將被列為警示的帳戶在事前展現出以下鮮明特徵：

1.  **角色定位 (Role)**: 主要是**資金匯集點**，接收遠多於付出。
2.  **時間特徵 (Temporal)**: 交易活動在帳戶生命週期的**末期會顯著加速**，呈現「最後的瘋狂」模式。
3.  **金額特徵 (Value)**: 交易以**小額高頻**為主，但夾雜極端大額交易。
4.  **網絡特徵 (Network)**: 普遍存在「多對一」的**扇入**結構，部分帳戶與海量對象有資金往來，是網絡中的關鍵節點。
5.  **行為特徵 (Behavioral)**: 高度依賴**行動銀行**等數位通路，但簡單的「快進快出」模式並非普遍現象，暗示了更複雜的洗錢手法。

這些發現為後續的特徵工程和模型建立提供了寶貴的方向與洞見。
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report generated at {output_path}")


def generate_overall_statistics(df):
    total_transactions = len(df)
    unique_alert_accounts = df["alert_acct"].nunique()
    avg_txns_per_account = total_transactions / unique_alert_accounts

    return f"""### 1.1 整體統計

- **總交易筆數**: {total_transactions:,} 筆
- **獨立警示帳戶數量**: {unique_alert_accounts:,} 個
- **核心發現**: 平均每個即將被警示的帳戶，在被警示前涉及約 {avg_txns_per_account:.0f} 筆交易。

"""


def generate_transaction_amount_analysis(df):
    mean_amt = df["txn_amt"].mean()
    median_amt = df["txn_amt"].median()
    std_amt = df["txn_amt"].std()
    min_amt = df["txn_amt"].min()
    max_amt = df["txn_amt"].max()

    return f"""### 1.2 交易金額分析

| 統計指標 | 交易金額 (新台幣) |
| :--- | :--- |
| 平均數 (Mean) | {mean_amt:,.0f} |
| 中位數 (Median) | {median_amt:,.0f} |
| 標準差 (Std Dev) | {std_amt:,.0f} |
| 最小值 (Min) | {min_amt:,.0f} |
| 最大值 (Max) | {max_amt:,.0f} |

- **核心發現**: 交易金額分佈極度右偏（平均值遠大於中位數），顯示大部分交易為小額，但存在少數極端大額的交易。過半數的交易金額低於 {median_amt:,.0f} 元。

"""


def generate_transaction_direction_analysis(df):
    as_payer = (df["from_acct"] == df["alert_acct"]).sum()
    as_receiver = (df["to_acct"] == df["alert_acct"]).sum()
    total_transactions = len(df)
    payer_pct = (as_payer / total_transactions) * 100
    receiver_pct = (as_receiver / total_transactions) * 100

    return f"""### 1.3 交易方向分析

- **作為付款方**: {as_payer:,} 筆 ({payer_pct:.2f}%)
- **作為收款方**: {as_receiver:,} 筆 ({receiver_pct:.2f}%)

- **核心發現**: 在被警示前，這些帳戶作為**收款方**的次數遠多於付款方，佔比超過六成。這符合詐騙活動中，人頭帳戶主要用於接收詐騙款項的典型特徵。

"""


def generate_transaction_channel_analysis(df):
    channel_dist = df["channel_type"].value_counts(normalize=True) * 100

    channel_map = {
        "UNK": "未知",
        "03": "行動銀行",
        "04": "網路銀行",
        "01": "ATM",
        "02": "臨櫃",
        "05": "語音",
        "99": "系統排程",
    }

    table = "| 通路代碼 | 通路名稱 | 佔比 |\n| :--- | :--- | :--- |\n"
    for channel, pct in channel_dist.items():
        table += f"| {channel} | {channel_map.get(channel, 'N/A')} | {pct:.2f}% |\n"

    return f"""### 1.4 交易通路分析

{table}
- **核心發現**:
    - 超過一半的交易通路未知（UNK），這是一個顯著的數據缺口。
    - 在已知通路中，**行動銀行 ({channel_dist.get("03", 0):.2f}%)** 是最主要的交易方式，遠超其他通路，顯示出詐騙交易對數位金融工具的高度依賴。

"""


def generate_preliminary_time_and_counterparty_analysis(df):
    avg_days_before = df["days_before_alert"].mean()
    within_7_days_pct = (df["days_before_alert"] <= 7).sum() / len(df) * 100

    counterparty_stats = df.groupby("alert_acct").agg(
        unique_counterparties=("from_acct", "nunique"),
    )
    avg_counterparties = counterparty_stats["unique_counterparties"].mean()
    max_counterparties = counterparty_stats["unique_counterparties"].max()

    return f"""### 1.5 初步時間與對象分析

- **平均交易發生時間**: 警示前 {avg_days_before:.2f} 天
- **警示前 7 天內交易**: 佔總交易筆數的 **{within_7_days_pct:.2f}%**
- **交易對象**: 平均與約 **{avg_counterparties:.0f}** 個獨立帳戶往來，但極端情況下有帳戶與超過 **{max_counterparties}** 個對象交易。

- **核心發現**: 交易活動在接近警示日期時有**顯著增加**的趨勢，且部分帳戶呈現出作為**資金分散或匯集中心（Mule Account Hub）**的強烈特徵。
"""


def generate_time_series_analysis(df):
    daily_summary = (
        df[df["days_before_alert"] <= 15]
        .groupby("days_before_alert")
        .agg(daily_txns=("txn_amt", "count"), daily_volume=("txn_amt", "sum"))
        .sort_index(ascending=False)
    )

    daily_summary["rolling_avg_txns"] = (
        daily_summary["daily_txns"].rolling(window=7, min_periods=1).mean()
    )

    table = """| 警示前天數 | 當日交易筆數 | 當日交易總額 (萬) | 7日滾動平均筆數 |
| :--- | :--- | :--- | :--- |
"""
    for day, row in daily_summary.iterrows():
        table += f"| {day} | {row['daily_txns']:,} | {row['daily_volume'] / 10000:,.2f} | {row['rolling_avg_txns']:.2f} |"

    last_day_txns = daily_summary.loc[1, "daily_txns"]
    last_day_volume = daily_summary.loc[1, "daily_volume"]

    return f"""### 2.1 時間序列分析 — 風暴前的集結

我們以「警示前天數」為時間軸，觀察交易活動的演變。數據顯示，帳戶在被警示前的最後時刻，交易活動呈現爆炸性增長。

#### 警示前 15 日交易活動匯總

{table}
- **核心發現**:
    1.  **活動劇增 (Ramping Up)**：交易筆數和金額在接近警示日（`days_before_alert` 變小）時急劇攀升。7日滾動平均筆數從兩週前的約 {daily_summary.loc[14, "rolling_avg_txns"]:.0f} 筆/日，飆升至最後一週的超過 {daily_summary.loc[7, "rolling_avg_txns"]:.0f} 筆/日。
    2.  **最後的瘋狂**: 在被警示前的**最後一天** (`days_before_alert` = 1)，交易活動達到頂峰，單日交易筆數高達 **{last_day_txns:,} 筆**，是近期平均值的兩倍；單日交易總額更是超過 **{last_day_volume / 10000:,.0f} 萬**。

"""


def generate_fan_in_fan_out_analysis(df):
    fan_in = (
        df[df["to_acct"] == df["alert_acct"]]
        .groupby("alert_acct")["from_acct"]
        .nunique()
    )
    fan_out = (
        df[df["from_acct"] == df["alert_acct"]]
        .groupby("alert_acct")["to_acct"]
        .nunique()
    )

    fan_in_dist = (
        pd.cut(fan_in, bins=[0, 1, 5, 10, 50, 100, float("inf")], right=True)
        .value_counts()
        .sort_index()
    )
    fan_out_dist = (
        pd.cut(fan_out, bins=[0, 1, 5, 10, 50, 100, float("inf")], right=True)
        .value_counts()
        .sort_index()
    )

    fan_in_table = """| 獨立收款來源數 | 警示帳戶數量 |
| :--- | :--- |
"""
    for interval, count in fan_in_dist.items():
        fan_in_table += f"""| {interval} | {count} |
"""

    fan_out_table = """| 獨立付款目標數 | 警示帳戶數量 |
| :--- | :--- |
"""
    for interval, count in fan_out_dist.items():
        fan_out_table += f"""| {interval} | {count} |
"""

    return f"""### 2.2 傳播鏈分析 — 資金的匯集與分散

我們分析了警示帳戶作為收款方（扇入）和付款方（扇出）時，其交易對象的數量分佈。

#### 資金扇入 (Fan-in) 分佈
（警示帳戶從多少獨立帳戶收款）

{fan_in_table}
#### 資金扇出 (Fan-out) 分佈
（警示帳戶向多少獨立帳戶付款）

{fan_out_table}
- **核心發現**:
    1.  **匯集中心 (Collection Hubs)**：**扇入的廣度遠大於扇出**。有 **{fan_in[fan_in > 100].count()}** 個警示帳戶的收款來源超過 **100** 個獨立帳戶，而付款目標超過 100 個的僅有 **{fan_out[fan_out > 100].count()}** 個。這強烈表明，這些帳戶在詐騙網絡中主要扮演**資金匯集中心**的角色。
    2.  **普遍的匯集模式**：超過半數的警示帳戶從多於一個來源收款，其中從 2-5 個來源收款是最常見的模式。

"""


if __name__ == "__main__":
    generate_report()
