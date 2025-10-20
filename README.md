# AI CUP 2025 玉山人工智慧公開挑戰賽－AI偵探出任務，精準揪出警示帳戶！

## 介紹
本專案為「2025 玉山人工智慧公開挑戰賽」的實作範本，主要展示如何利用交易資料進行二元分類。程式 `TransactionAlertClassifier.py` 示範了如何前處理資料、訓練模型及產生帳戶警示的預測結果。有興趣的參賽者可於2025/09/17 - 2025/11/05至 https://tbrain.trendmicro.com.tw/Competitions/Details/40 報名，即可取得資料集及相關說明。

## 資料說明
本次比賽提供過去N個月之交易資料，參賽者需預測交易資料中的玉山帳戶在未來是否轉為警示戶 (舉例：提供7月至10月的交易資料，預測交易資料中的帳戶在11月是否轉為警示帳戶)。

以下僅大略說明資料集，請參賽隊伍在報名後於初賽時程內至T-Brain 平臺Dataset Download區下載訓練資料集。
- 交易資料集：約400萬筆，每個row即為一筆交易，含匯款帳戶、收款帳戶，以及交易時間、日期、金額、幣別等交易數據
- 警示帳戶註記：約1000筆，每個row為警示帳戶及對應的警示日
- 警示帳戶待預測清單：約4000筆，需預測這些帳戶在未來一個月內是否轉為警示戶

## 主要功能
- 讀取交易、警示及預測帳戶的CSV資料檔
- 資料前處理，針對每個帳戶萃取特徵
- 分割訓練與測試資料集
- 使用 scikit-learn 的決策樹模型進行訓練與預測
- 輸出預測結果至 result.csv

## 使用方式
1. **安裝套件**
   - 請先安裝 Python 3.12或以上
   - 本專案使用 `uv` 進行套件管理，安裝指令如下：
     ```bash
     uv sync
     ```
   - 啟用虛擬環境：
     ```bash
     source .venv/bin/activate
     ```
2. **準備資料**
   - 執行以下指令自動下載並解壓縮資料集：
     ```bash
     bash get_dataset.sh
     ```
   - 資料將會自動放置在 `data/` 資料夾中，包含以下三個CSV檔案：
     - acct_transaction.csv
     - acct_alert.csv
     - acct_predict.csv
3. **執行程式**
   - 執行預測程式：
     ```bash
     python TransactionAlertClassifier.py
     ```
   - 預測結果將會儲存於 result.csv。

## TransactionAlertClassifier.py

這個 Python 程式展示了一個利用交易資料進行二元分類的工作流程。它載入交易、警示和預測帳戶的資料集，對資料進行前處理以萃取特徵，訓練決策樹分類器，並輸出帳戶警示的預測結果。

### 主要函式
- `LoadCSV(dir_path)`: 讀取三個資料集
- `PreProcessing(df)`: 帳戶特徵萃取
- `TrainTestSplit(df, df_alert, df_test)`: 準備訓練與測試資料
- `Modeling(X_train, y_train, X_test)`: 決策樹訓練與預測
- `OutputCSV(path, df_test, X_test, y_pred)`: 輸出預測結果

您可以依需求調整資料前處理及建模流程。

## 注意事項
- 本範本僅以玉山帳戶作為訓練資料，旨在使參賽者了解基本資料處理流程，參賽者可依需求調整資料前處理及建模方法提升成效。
