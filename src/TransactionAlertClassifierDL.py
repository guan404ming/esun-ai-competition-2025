"""
2025玉山人工智慧挑戰賽 - Deep Learning 版本
使用 PyTorch 實作神經網路進行警示帳戶預測
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from datetime import datetime


def LoadCSV(dir_path):
    """
    讀取挑戰賽提供的3個資料集：交易資料、警示帳戶註記、待預測帳戶清單
    Args:
        dir_path (str): 資料夾，請把上述3個檔案放在同一個資料夾

    Returns:
        df_txn: 交易資料 DataFrame
        df_alert: 警示帳戶註記 DataFrame
        df_test: 待預測帳戶清單 DataFrame
    """
    df_txn = pd.read_csv(os.path.join(dir_path, "acct_transaction.csv"))
    df_alert = pd.read_csv(os.path.join(dir_path, "acct_alert.csv"))
    df_test = pd.read_csv(os.path.join(dir_path, "acct_predict.csv"))

    print("(Finish) Load Dataset.")
    return df_txn, df_alert, df_test


def EnhancedFeatureEngineering(df):
    """
    增強版特徵工程，為深度學習模型提供更豐富的特徵
    包含：
    - 基本統計量 (總額、最大/最小/平均值)
    - 交易次數統計
    - 交易時間特徵
    - 帳戶類型特徵
    - 交易行為模式特徵
    """
    features_list = []

    # 1. 發送交易統計
    send_stats = df.groupby("from_acct").agg(
        {"txn_amt": ["sum", "max", "min", "mean", "std", "count"]}
    )
    send_stats.columns = [
        "total_send_amt",
        "max_send_amt",
        "min_send_amt",
        "avg_send_amt",
        "std_send_amt",
        "send_count",
    ]
    send_stats = send_stats.reset_index().rename(columns={"from_acct": "acct"})

    # 2. 接收交易統計
    recv_stats = df.groupby("to_acct").agg(
        {"txn_amt": ["sum", "max", "min", "mean", "std", "count"]}
    )
    recv_stats.columns = [
        "total_recv_amt",
        "max_recv_amt",
        "min_recv_amt",
        "avg_recv_amt",
        "std_recv_amt",
        "recv_count",
    ]
    recv_stats = recv_stats.reset_index().rename(columns={"to_acct": "acct"})

    # 3. 合併發送和接收統計
    df_result = pd.merge(send_stats, recv_stats, on="acct", how="outer").fillna(0)

    # 4. 計算衍生特徵
    df_result["total_txn_count"] = df_result["send_count"] + df_result["recv_count"]
    df_result["net_flow"] = df_result["total_recv_amt"] - df_result["total_send_amt"]
    df_result["send_recv_ratio"] = df_result["total_send_amt"] / (
        df_result["total_recv_amt"] + 1
    )
    df_result["avg_txn_amt"] = (
        df_result["total_send_amt"] + df_result["total_recv_amt"]
    ) / (df_result["total_txn_count"] + 1)

    # 5. 交易波動性特徵
    df_result["txn_volatility"] = (
        df_result["std_send_amt"] + df_result["std_recv_amt"]
    ) / 2

    # 6. 獲取帳戶類型 (is_esun)
    df_from = df[["from_acct", "from_acct_type"]].rename(
        columns={"from_acct": "acct", "from_acct_type": "is_esun"}
    )
    df_to = df[["to_acct", "to_acct_type"]].rename(
        columns={"to_acct": "acct", "to_acct_type": "is_esun"}
    )
    df_acc = (
        pd.concat([df_from, df_to], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 7. 合併帳戶類型
    df_result = pd.merge(df_result, df_acc, on="acct", how="left")

    # 8. 處理缺失值
    df_result = df_result.fillna(0)

    print(
        f"(Finish) Enhanced Feature Engineering. Total features: {len(df_result.columns) - 1}"
    )
    return df_result


class TransactionDataset(Dataset):
    """
    PyTorch Dataset 類別，用於處理交易資料
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AlertClassifierNet(nn.Module):
    """
    深度神經網路模型，用於警示帳戶分類
    架構：
    - 輸入層
    - 3個隱藏層 (256 -> 128 -> 64)
    - 使用 BatchNorm 和 Dropout 防止過擬合
    - ReLU 激活函數
    - 輸出層不使用 Sigmoid (配合 BCEWithLogitsLoss)
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(AlertClassifierNet, self).__init__()

        layers = []
        prev_dim = input_dim

        # 隱藏層
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 輸出層 (不使用 Sigmoid，因為 BCEWithLogitsLoss 內建)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def TrainTestSplit(
    df, df_alert, df_test, val_size=0.2, use_smote=True, random_state=42
):
    """
    切分訓練集、驗證集及測試集
    Args:
        df: 特徵資料
        df_alert: 警示帳戶標籤
        df_test: 待預測帳戶清單
        val_size: 驗證集比例
        use_smote: 是否使用 SMOTE 處理類別不平衡
        random_state: 隨機種子

    Returns:
        處理後的訓練/驗證/測試資料
    """
    # 準備訓練資料 (僅使用玉山帳戶)
    X_train_full = (
        df[(~df["acct"].isin(df_test["acct"])) & (df["is_esun"] == 1)]
        .drop(columns=["is_esun"])
        .copy()
    )
    y_train_full = X_train_full["acct"].isin(df_alert["acct"]).astype(int)

    # 準備測試資料
    X_test = df[df["acct"].isin(df_test["acct"])].drop(columns=["is_esun"]).copy()

    # 保存帳戶ID
    train_accts = X_train_full["acct"].values
    test_accts = X_test["acct"].values

    # 移除帳戶ID列
    X_train_features = X_train_full.drop(columns=["acct"]).values
    X_test_features = X_test.drop(columns=["acct"]).values
    y_train_values = y_train_full.values

    # 切分訓練集和驗證集
    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X_train_features,
        y_train_values,
        np.arange(len(y_train_values)),
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_values,
    )

    # 特徵標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_features)

    # 使用 SMOTE 處理類別不平衡
    if use_smote:
        print(f"Original class distribution: {np.bincount(y_train.astype(int))}")
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE class distribution: {np.bincount(y_train.astype(int))}")

    print(f"(Finish) Train-Val-Test Split")
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Validation samples: {len(X_val_scaled)}")
    print(f"  Test samples: {len(X_test_scaled)}")

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        test_accts,
        scaler,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device="cpu",
    patience=10,
    use_amp=True,
):
    """
    訓練深度學習模型，使用 F1 score 作為評估指標
    Args:
        model: PyTorch 模型
        train_loader: 訓練資料 DataLoader
        val_loader: 驗證資料 DataLoader
        criterion: 損失函數
        optimizer: 優化器
        num_epochs: 訓練輪數
        device: 運算裝置 (cpu/cuda)
        patience: Early stopping 的耐心值
        use_amp: 是否使用混合精度訓練 (僅 CUDA)

    Returns:
        訓練好的模型
    """
    model.to(device)
    best_val_f1 = 0.0
    patience_counter = 0

    # 混合精度訓練 (僅 CUDA)
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster training")

    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for X_batch, y_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 混合精度訓練
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            # 使用 sigmoid 將 logits 轉為機率，再進行閾值判斷
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_preds.extend(predictions.detach().cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                # 使用 sigmoid 將 logits 轉為機率，再進行閾值判斷
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        # 計算評估指標
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        train_precision = precision_score(train_labels, train_preds, zero_division=0)
        train_recall = recall_score(train_labels, train_preds, zero_division=0)

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(
            f"  Train Loss: {avg_train_loss:.4f}, F1: {train_f1:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}"
        )
        print(
            f"  Val Loss: {avg_val_loss:.4f}, F1: {val_f1:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}"
        )

        # Early stopping based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # 保存最佳模型到 results 資料夾
            model_path = (
                f"results/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            )
            torch.save(model.state_dict(), model_path)
            print(f"  -> Best model saved to {model_path}! (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation F1 score: {best_val_f1:.4f}")
                break

    # 載入最佳模型
    model.load_state_dict(torch.load(model_path))
    print("(Finish) Model Training")
    return model


def predict(model, X_test, device="cpu", batch_size=256):
    """
    使用訓練好的模型進行預測
    """
    model.eval()
    model.to(device)

    X_test_tensor = torch.FloatTensor(X_test)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            # 使用 sigmoid 將 logits 轉為機率，再進行閾值判斷
            preds = (torch.sigmoid(outputs) > 0.5).float()
            predictions.extend(preds.cpu().numpy())

    return np.array(predictions, dtype=int)


def OutputCSV(path, df_test, test_accts, y_pred):
    """
    根據測試資料集及預測結果，產出預測結果之CSV，該CSV可直接上傳於TBrain
    """
    df_pred = pd.DataFrame({"acct": test_accts, "label": y_pred})
    df_out = df_test[["acct"]].merge(df_pred, on="acct", how="left")
    df_out.to_csv(path, index=False)

    print(f"(Finish) Output saved to {path}")
    print(f"Prediction distribution: {np.bincount(y_pred)}")


if __name__ == "__main__":
    # 建立 results 資料夾
    os.makedirs("results", exist_ok=True)

    # 設定隨機種子以確保可重現性
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 檢查是否有 GPU 可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # 超參數設定 - 優化 GPU 使用率
    # GPU 優化策略:
    # 1. 增加 BATCH_SIZE (256->1024) - 提高 GPU 並行計算量
    # 2. 增加模型容量 HIDDEN_DIMS ([256,128,64]->[512,256,128]) - 增加計算密度
    # 3. 啟用 AMP (混合精度訓練) - 減少記憶體使用，提高吞吐量
    # 4. pin_memory + num_workers - 加速 CPU->GPU 資料傳輸
    # 5. non_blocking transfers - CPU/GPU 非同步操作
    BATCH_SIZE = 1024  # 增加批次大小以提高 GPU 利用率
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    HIDDEN_DIMS = [512, 256, 128]  # 增加模型容量
    DROPOUT_RATE = 0.3
    PATIENCE = 15
    USE_SMOTE = True
    USE_AMP = True  # 啟用混合精度訓練 (Float16/32 混合)
    NUM_WORKERS = 4  # DataLoader 多線程 (根據 CPU 核心數調整)

    # 1. 載入資料
    dir_path = "data/"
    df_txn, df_alert, df_test = LoadCSV(dir_path)

    # 2. 特徵工程
    df_X = EnhancedFeatureEngineering(df_txn)

    # 3. 切分資料集
    X_train, X_val, X_test, y_train, y_val, test_accts, scaler = TrainTestSplit(
        df_X, df_alert, df_test, use_smote=USE_SMOTE, random_state=RANDOM_SEED
    )

    # 4. 建立 DataLoader - 優化 GPU 傳輸
    train_dataset = TransactionDataset(X_train, y_train)
    val_dataset = TransactionDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    # 5. 建立模型
    input_dim = X_train.shape[1]
    model = AlertClassifierNet(
        input_dim, hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT_RATE
    )

    # 6. 設定損失函數和優化器
    # 使用 BCEWithLogitsLoss (結合 Sigmoid + BCE，對 AMP 安全)
    # 如果使用 SMOTE，類別已經平衡，不需要 pos_weight
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 訓練模型
    model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=NUM_EPOCHS,
        device=device,
        patience=PATIENCE,
        use_amp=USE_AMP,
    )

    # 8. 預測測試集
    y_pred = predict(model, X_test, device=device)

    # 9. 輸出結果到 results 資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/{timestamp}_prediction_dl.csv"
    OutputCSV(out_path, df_test, test_accts, y_pred)

    print("\n=== Training Complete ===")
    print(f"Model saved to: results/best_model_{timestamp}.pth")
    print(f"Predictions saved to: {out_path}")
