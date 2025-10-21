import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class TransactionSequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        if self.labels is not None:
            label = torch.LongTensor([self.labels[idx]])
            return sequence, label
        return sequence


class TransactionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(TransactionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 2),
        )

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.norm1(lstm_out + attn_out)

        # Global average pooling over sequence dimension
        pooled = torch.mean(attn_out, dim=1)
        pooled = self.norm2(pooled)

        output = self.classifier(pooled)
        return output


class TimeSeriesAlertClassifier:
    def __init__(self, sequence_length=50, hidden_size=128, num_layers=2):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        self.categorical_mappings = {}  # Store mappings instead of LabelEncoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        self.model = None

    def load_data(self, data_dir="data/"):
        """Load and preprocess the competition data using Polars"""
        print("Loading data with Polars...")

        # Load datasets with Polars
        df_transaction = pl.read_csv(f"{data_dir}/acct_transaction_transfered.csv")
        df_alert = pl.read_csv(f"{data_dir}/acct_alert.csv")
        df_predict = pl.read_csv(f"{data_dir}/acct_predict.csv")

        print(
            f"Loaded {len(df_transaction)} transactions, {len(df_alert)} alerts, {len(df_predict)} predictions"
        )

        return df_transaction, df_alert, df_predict

    def extract_temporal_features(self, df_transaction):
        """Extract temporal and sequence features from transaction data using Polars"""
        print("Extracting temporal features with Polars...")

        # Extract temporal features
        df_pl = df_transaction.with_columns(
            [
                pl.col("txn_date").alias("day_of_year"),
                pl.col("txn_time")
                .str.strptime(pl.Time, "%H:%M:%S")
                .dt.hour()
                .alias("hour"),
                pl.col("txn_time")
                .str.strptime(pl.Time, "%H:%M:%S")
                .dt.minute()
                .alias("minute"),
            ]
        )

        # Create minute timestamp for interval calculation
        df_pl = df_pl.with_columns(
            [(pl.col("hour") * 60 + pl.col("minute")).alias("minute_timestamp")]
        )

        # Encode categorical features using Polars
        categorical_cols = [
            "from_acct_type",
            "to_acct_type",
            "is_self_txn",
            "currency_type",
            "channel_type",
        ]

        for col in categorical_cols:
            # Fill nulls with "UNK"
            df_pl = df_pl.with_columns(pl.col(col).fill_null("UNK"))

            if col not in self.categorical_mappings:
                # Create mapping from unique values to integers
                unique_vals = df_pl[col].unique().sort()
                self.categorical_mappings[col] = {
                    val: idx for idx, val in enumerate(unique_vals.to_list())
                }

            # Map categorical values to integers
            mapping = self.categorical_mappings[col]
            df_pl = df_pl.with_columns(
                pl.col(col).map_elements(
                    lambda x: mapping.get(x, 0), return_dtype=pl.Int32
                )
            )

        # Create transaction direction features
        df_pl = df_pl.with_columns(pl.lit(1).alias("is_outgoing"))

        # Get all column names
        all_cols = df_pl.columns

        # Create incoming transactions view - swap from_acct and to_acct, keep same column order
        incoming_cols = []
        for col in all_cols:
            if col == "from_acct":
                incoming_cols.append(pl.col("to_acct").alias("from_acct"))
            elif col == "to_acct":
                incoming_cols.append(pl.col("from_acct").alias("to_acct"))
            elif col == "is_outgoing":
                incoming_cols.append(pl.lit(0).alias("is_outgoing"))
            elif col == "txn_amt":
                incoming_cols.append((-pl.col("txn_amt")).alias("txn_amt"))
            else:
                incoming_cols.append(pl.col(col))

        df_incoming = df_pl.select(incoming_cols)

        # Combine and sort using Polars
        print("Combining outgoing and incoming transactions...")
        df_combined = pl.concat([df_pl, df_incoming], how="vertical")
        df_combined = df_combined.sort(["from_acct", "day_of_year", "hour", "minute"])

        print(f"Created combined dataset with {len(df_combined)} rows")
        return df_combined

    def calculate_daily_features(self, df_combined):
        """Calculate enhanced daily transaction features for each account using Polars"""
        print("Calculating daily transaction features with Polars...")

        # Sort data for interval calculations
        df_pl = df_combined.sort(["from_acct", "day_of_year", "minute_timestamp"])

        # Calculate transaction intervals within each account-day group
        df_pl = df_pl.with_columns(
            [
                pl.col("minute_timestamp")
                .diff()
                .over(["from_acct", "day_of_year"])
                .alias("interval"),
                pl.col("txn_amt").abs().alias("abs_amt"),
            ]
        )

        # Aggregate daily features using Polars expressions
        print("Aggregating daily features...")
        daily_df = (
            df_pl.group_by(["from_acct", "day_of_year"])
            .agg(
                [
                    # Transaction counts
                    pl.len().alias("daily_txn_count"),
                    # Volume features
                    pl.col("abs_amt").max().alias("daily_max_volume"),
                    pl.col("abs_amt").sum().alias("daily_total_volume"),
                    # Interval features
                    pl.col("interval").min().fill_null(0).alias("daily_min_interval"),
                    pl.col("interval").mean().fill_null(0).alias("daily_avg_interval"),
                    # Transaction direction counts
                    pl.col("is_outgoing").sum().alias("daily_outgoing_count"),
                    pl.col("is_self_txn").sum().alias("daily_self_txn_count"),
                    # Time distribution features
                    pl.col("hour").min().alias("hour_min"),
                    pl.col("hour").max().alias("hour_max"),
                    pl.col("hour").n_unique().alias("unique_hours"),
                ]
            )
            .with_columns(
                [
                    # Calculate derived features
                    (pl.col("daily_txn_count") - pl.col("daily_outgoing_count")).alias(
                        "daily_incoming_count"
                    ),
                    (pl.col("hour_max") - pl.col("hour_min")).alias("hour_spread"),
                ]
            )
            .drop(["hour_min", "hour_max"])
        )

        print(f"Calculated features for {len(daily_df)} account-day combinations")
        return daily_df

    def create_sequences(self, df_combined, df_alert=None, df_predict=None):
        """Create time series sequences for each account using Polars"""
        print("Creating sequences...")

        # Calculate daily features
        daily_df = self.calculate_daily_features(df_combined)

        # Merge daily features back to transaction data
        df_combined = df_combined.join(
            daily_df, on=["from_acct", "day_of_year"], how="left"
        )

        feature_cols = [
            "txn_amt",
            "day_of_year",
            "hour",
            "minute",
            "is_outgoing",
            "from_acct_type",
            "to_acct_type",
            "is_self_txn",
            "currency_type",
            "channel_type",
            "daily_txn_count",
            "daily_max_volume",
            "daily_total_volume",
            "daily_min_interval",
            "daily_avg_interval",
            "daily_outgoing_count",
            "daily_incoming_count",
            "daily_self_txn_count",
            "hour_spread",
            "unique_hours",
        ]

        sequences = []
        labels = []
        account_ids = []

        # Get unique accounts
        if df_alert is not None:
            alert_accounts = set(df_alert["acct"].to_list())
        else:
            alert_accounts = set()

        if df_predict is not None:
            predict_accounts = set(df_predict["acct"].to_list())
        else:
            predict_accounts = set()

        all_accounts = df_combined["from_acct"].unique().to_list()

        for account in tqdm(all_accounts, desc="Creating sequences"):
            account_data = df_combined.filter(pl.col("from_acct") == account)

            if len(account_data) < 5:  # Skip accounts with too few transactions
                continue

            # Fill missing values with 0 for daily features
            account_data = account_data.fill_null(0)

            # Create features for this account - convert to numpy
            features = account_data.select(feature_cols).to_numpy()

            # Create sequences of fixed length
            if len(features) >= self.sequence_length:
                # Use the most recent transactions
                sequence = features[-self.sequence_length :]
                sequences.append(sequence)
                account_ids.append(account)

                # Determine label
                if account in alert_accounts:
                    labels.append(1)
                elif account in predict_accounts:
                    labels.append(-1)  # Unknown label for prediction
                else:
                    labels.append(0)  # Normal account
            else:
                # Pad sequences that are too short
                padded_sequence = np.zeros((self.sequence_length, len(feature_cols)))
                padded_sequence[-len(features) :] = features
                sequences.append(padded_sequence)
                account_ids.append(account)

                if account in alert_accounts:
                    labels.append(1)
                elif account in predict_accounts:
                    labels.append(-1)
                else:
                    labels.append(0)

        sequences = np.array(sequences)
        labels = np.array(labels)

        print(f"Created {len(sequences)} sequences with {len(feature_cols)} features")
        print(
            f"Label distribution: Normal={np.sum(labels == 0)}, Alert={np.sum(labels == 1)}, Unknown={np.sum(labels == -1)}"
        )

        return sequences, labels, account_ids

    def prepare_training_data(self, sequences, labels, account_ids):
        """Prepare training and validation data"""
        print("Preparing training data...")

        # Only use labeled data for training (exclude prediction accounts)
        train_mask = labels != -1
        train_sequences = sequences[train_mask]
        train_labels = labels[train_mask]

        # Reshape for scaling
        original_shape = train_sequences.shape
        train_sequences_scaled = self.scaler.fit_transform(
            train_sequences.reshape(-1, original_shape[-1])
        ).reshape(original_shape)

        # Split training data temporally (earlier data for training, later for validation)
        X_train, X_val, y_train, y_val = train_test_split(
            train_sequences_scaled,
            train_labels,
            test_size=0.2,
            random_state=42,
            stratify=train_labels,
        )

        print(f"Training set: {len(X_train)} sequences")
        print(f"Validation set: {len(X_val)} sequences")

        return X_train, X_val, y_train, y_val, train_sequences_scaled, train_labels

    def train_model(
        self, X_train, X_val, y_train, y_val, epochs=100, batch_size=32, lr=0.001
    ):
        """Train the LSTM model"""
        print("Training model...")

        input_size = X_train.shape[2]
        self.model = TransactionLSTM(input_size, self.hidden_size, self.num_layers).to(
            self.device
        )

        # Create datasets and dataloaders
        train_dataset = TransactionSequenceDataset(X_train, y_train)
        val_dataset = TransactionSequenceDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in tqdm(range(epochs), desc="Training epochs"):
            # Training
            self.model.train()
            train_loss = 0
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False)
            for sequences, labels in train_progress:
                sequences, labels = (
                    sequences.to(self.device),
                    labels.squeeze().to(self.device),
                )

                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_progress.set_postfix({"loss": loss.item()})

            # Validation
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False)
                for sequences, labels in val_progress:
                    sequences, labels = (
                        sequences.to(self.device),
                        labels.squeeze().to(self.device),
                    )
                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    predicted = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predicted.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
                    val_progress.set_postfix({"loss": loss.item()})

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                auc_score = (
                    roc_auc_score(val_targets, val_predictions)
                    if len(set(val_targets)) > 1
                    else 0
                )
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {auc_score:.4f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load("best_model.pth"))
        print("Training completed!")

        return val_predictions, val_targets

    def predict(self, sequences, account_ids):
        """Make predictions on new data"""
        print("Making predictions...")

        # Scale the sequences
        original_shape = sequences.shape
        sequences_scaled = self.scaler.transform(
            sequences.reshape(-1, original_shape[-1])
        ).reshape(original_shape)

        # Create dataset and dataloader
        predict_dataset = TransactionSequenceDataset(sequences_scaled)
        predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

        predictions = []
        prediction_probs = []

        self.model.eval()
        with torch.no_grad():
            predict_progress = tqdm(predict_loader, desc="Predicting")
            for sequences_batch in predict_progress:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)

                predictions.extend(predicted.cpu().numpy())
                prediction_probs.extend(
                    probs[:, 1].cpu().numpy()
                )  # Probability of being alert

        return predictions, prediction_probs

    def run_pipeline(self, data_dir="data/"):
        """Run the complete pipeline"""
        # Load data
        df_transaction, df_alert, df_predict = self.load_data(data_dir)

        # Extract features
        df_combined = self.extract_temporal_features(df_transaction)

        # Create sequences
        sequences, labels, account_ids = self.create_sequences(
            df_combined, df_alert, df_predict
        )

        # Prepare training data
        X_train, X_val, y_train, y_val, all_train_sequences, all_train_labels = (
            self.prepare_training_data(sequences, labels, account_ids)
        )

        # Train model
        val_predictions, val_targets = self.train_model(X_train, X_val, y_train, y_val)

        # Print validation results
        print("\nValidation Results:")
        print(classification_report(val_targets, val_predictions))
        if len(set(val_targets)) > 1:
            auc_score = roc_auc_score(val_targets, val_predictions)
            print(f"Validation AUC: {auc_score:.4f}")

        # Make predictions on test set
        predict_mask = labels == -1
        if np.any(predict_mask):
            predict_sequences = sequences[predict_mask]
            predict_account_ids = [
                account_ids[i] for i in range(len(account_ids)) if predict_mask[i]
            ]

            predictions, prediction_probs = self.predict(
                predict_sequences, predict_account_ids
            )

            # Create submission file using Polars
            submission_df = pl.DataFrame(
                {"acct": predict_account_ids, "label": predictions}
            )

            submission_df.write_csv("time_series_result.csv")
            print("\nPredictions saved to time_series_result.csv")
            print(f"Predicted alert accounts: {np.sum(predictions)}/{len(predictions)}")

            return submission_df

        return None


if __name__ == "__main__":
    classifier = TimeSeriesAlertClassifier(
        sequence_length=30, hidden_size=128, num_layers=2
    )
    result = classifier.run_pipeline()
