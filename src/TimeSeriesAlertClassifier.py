import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

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

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8,
                                             dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 2)
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
        self.label_encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        self.model = None

    def load_data(self, data_dir='data/'):
        """Load and preprocess the competition data"""
        print("Loading data...")

        # Load datasets
        df_transaction = pd.read_csv(f'{data_dir}/acct_transaction.csv')
        df_alert = pd.read_csv(f'{data_dir}/acct_alert.csv')
        df_predict = pd.read_csv(f'{data_dir}/acct_predict.csv')

        print(f"Loaded {len(df_transaction)} transactions, {len(df_alert)} alerts, {len(df_predict)} predictions")

        return df_transaction, df_alert, df_predict

    def extract_temporal_features(self, df_transaction):
        """Extract temporal and sequence features from transaction data"""
        print("Extracting temporal features...")

        # Convert txn_date to proper datetime representation
        df_transaction['txn_datetime'] = pd.to_datetime(
            df_transaction['txn_date'].astype(str) + ' ' + df_transaction['txn_time'].astype(str),
            format='%j %H:%M:%S', errors='coerce'
        )

        # Extract temporal features
        df_transaction['day_of_year'] = df_transaction['txn_date']
        df_transaction['hour'] = pd.to_datetime(df_transaction['txn_time'], format='%H:%M:%S').dt.hour
        df_transaction['minute'] = pd.to_datetime(df_transaction['txn_time'], format='%H:%M:%S').dt.minute

        # Create minute timestamp for interval calculation
        df_transaction['minute_timestamp'] = df_transaction['hour'] * 60 + df_transaction['minute']

        # Encode categorical features
        categorical_cols = ['from_acct_type', 'to_acct_type', 'is_self_txn', 'currency_type', 'channel_type']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_transaction[col] = self.label_encoders[col].fit_transform(df_transaction[col].fillna('UNK'))
            else:
                df_transaction[col] = self.label_encoders[col].transform(df_transaction[col].fillna('UNK'))

        # Create transaction direction features
        df_transaction['is_outgoing'] = 1
        df_incoming = df_transaction.copy()
        df_incoming['from_acct'] = df_transaction['to_acct']
        df_incoming['is_outgoing'] = 0
        df_incoming['txn_amt'] = -df_transaction['txn_amt']  # Negative for incoming

        # Combine outgoing and incoming transactions
        df_combined = pd.concat([df_transaction, df_incoming], ignore_index=True)
        df_combined = df_combined.sort_values(['from_acct', 'day_of_year', 'hour', 'minute'])

        return df_combined

    def calculate_daily_features(self, df_combined):
        """Calculate enhanced daily transaction features for each account"""
        print("Calculating daily transaction features...")

        daily_features = []

        for account in df_combined['from_acct'].unique():
            account_data = df_combined[df_combined['from_acct'] == account]

            # Group by day
            daily_groups = account_data.groupby('day_of_year')

            for day, day_data in daily_groups:
                # 單日最高交易次數 (Daily max transaction count)
                daily_txn_count = len(day_data)

                # 單日最高交易量 (Daily max transaction volume)
                daily_max_volume = day_data['txn_amt'].abs().max()
                daily_total_volume = day_data['txn_amt'].abs().sum()

                # 單日最短交易間隔 (Daily minimum transaction interval in minutes)
                if len(day_data) > 1:
                    day_data_sorted = day_data.sort_values('minute_timestamp')
                    intervals = day_data_sorted['minute_timestamp'].diff().dropna()
                    daily_min_interval = intervals.min() if len(intervals) > 0 else 0
                    daily_avg_interval = intervals.mean() if len(intervals) > 0 else 0
                else:
                    daily_min_interval = 0
                    daily_avg_interval = 0

                # Additional daily features
                daily_outgoing_count = len(day_data[day_data['is_outgoing'] == 1])
                daily_incoming_count = len(day_data[day_data['is_outgoing'] == 0])
                daily_self_txn_count = len(day_data[day_data['is_self_txn'] == 1])

                # Time distribution features
                hour_spread = day_data['hour'].max() - day_data['hour'].min()
                unique_hours = day_data['hour'].nunique()

                daily_features.append({
                    'from_acct': account,
                    'day_of_year': day,
                    'daily_txn_count': daily_txn_count,
                    'daily_max_volume': daily_max_volume,
                    'daily_total_volume': daily_total_volume,
                    'daily_min_interval': daily_min_interval,
                    'daily_avg_interval': daily_avg_interval,
                    'daily_outgoing_count': daily_outgoing_count,
                    'daily_incoming_count': daily_incoming_count,
                    'daily_self_txn_count': daily_self_txn_count,
                    'hour_spread': hour_spread,
                    'unique_hours': unique_hours
                })

        daily_df = pd.DataFrame(daily_features)
        return daily_df

    def create_sequences(self, df_combined, df_alert=None, df_predict=None):
        """Create time series sequences for each account"""
        print("Creating sequences...")

        # Calculate daily features
        daily_df = self.calculate_daily_features(df_combined)

        # Merge daily features back to transaction data
        df_combined = df_combined.merge(daily_df, on=['from_acct', 'day_of_year'], how='left')

        feature_cols = ['txn_amt', 'day_of_year', 'hour', 'minute', 'is_outgoing',
                       'from_acct_type', 'to_acct_type', 'is_self_txn',
                       'currency_type', 'channel_type', 'daily_txn_count',
                       'daily_max_volume', 'daily_total_volume', 'daily_min_interval',
                       'daily_avg_interval', 'daily_outgoing_count', 'daily_incoming_count',
                       'daily_self_txn_count', 'hour_spread', 'unique_hours']

        sequences = []
        labels = []
        account_ids = []

        # Get unique accounts
        if df_alert is not None:
            alert_accounts = set(df_alert['acct'].values)
        else:
            alert_accounts = set()

        if df_predict is not None:
            predict_accounts = set(df_predict['acct'].values)
        else:
            predict_accounts = set()

        all_accounts = df_combined['from_acct'].unique()

        for account in all_accounts:
            account_data = df_combined[df_combined['from_acct'] == account]

            if len(account_data) < 5:  # Skip accounts with too few transactions
                continue

            # Fill missing values with 0 for daily features
            account_data = account_data.fillna(0)

            # Create features for this account
            features = account_data[feature_cols].values

            # Create sequences of fixed length
            if len(features) >= self.sequence_length:
                # Use the most recent transactions
                sequence = features[-self.sequence_length:]
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
                padded_sequence[-len(features):] = features
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
        print(f"Label distribution: Normal={np.sum(labels==0)}, Alert={np.sum(labels==1)}, Unknown={np.sum(labels==-1)}")

        return sequences, labels, account_ids

    def prepare_training_data(self, sequences, labels, account_ids):
        """Prepare training and validation data"""
        print("Preparing training data...")

        # Only use labeled data for training (exclude prediction accounts)
        train_mask = labels != -1
        train_sequences = sequences[train_mask]
        train_labels = labels[train_mask]
        train_account_ids = [account_ids[i] for i in range(len(account_ids)) if train_mask[i]]

        # Reshape for scaling
        original_shape = train_sequences.shape
        train_sequences_scaled = self.scaler.fit_transform(
            train_sequences.reshape(-1, original_shape[-1])
        ).reshape(original_shape)

        # Split training data temporally (earlier data for training, later for validation)
        X_train, X_val, y_train, y_val = train_test_split(
            train_sequences_scaled, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )

        print(f"Training set: {len(X_train)} sequences")
        print(f"Validation set: {len(X_val)} sequences")

        return X_train, X_val, y_train, y_val, train_sequences_scaled, train_labels

    def train_model(self, X_train, X_val, y_train, y_val, epochs=100, batch_size=32, lr=0.001):
        """Train the LSTM model"""
        print("Training model...")

        input_size = X_train.shape[2]
        self.model = TransactionLSTM(input_size, self.hidden_size, self.num_layers).to(self.device)

        # Create datasets and dataloaders
        train_dataset = TransactionSequenceDataset(X_train, y_train)
        val_dataset = TransactionSequenceDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.squeeze().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(self.device), labels.squeeze().to(self.device)
                    outputs = self.model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    predicted = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predicted.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                auc_score = roc_auc_score(val_targets, val_predictions) if len(set(val_targets)) > 1 else 0
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {auc_score:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
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
            for sequences_batch in predict_loader:
                sequences_batch = sequences_batch.to(self.device)
                outputs = self.model(sequences_batch)
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)

                predictions.extend(predicted.cpu().numpy())
                prediction_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being alert

        return predictions, prediction_probs

    def run_pipeline(self, data_dir='data/'):
        """Run the complete pipeline"""
        # Load data
        df_transaction, df_alert, df_predict = self.load_data(data_dir)

        # Extract features
        df_combined = self.extract_temporal_features(df_transaction)

        # Create sequences
        sequences, labels, account_ids = self.create_sequences(df_combined, df_alert, df_predict)

        # Prepare training data
        X_train, X_val, y_train, y_val, all_train_sequences, all_train_labels = self.prepare_training_data(
            sequences, labels, account_ids
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
            predict_account_ids = [account_ids[i] for i in range(len(account_ids)) if predict_mask[i]]

            predictions, prediction_probs = self.predict(predict_sequences, predict_account_ids)

            # Create submission file
            submission_df = pd.DataFrame({
                'acct': predict_account_ids,
                'label': predictions
            })

            submission_df.to_csv('time_series_result.csv', index=False)
            print(f"\nPredictions saved to time_series_result.csv")
            print(f"Predicted alert accounts: {np.sum(predictions)}/{len(predictions)}")

            return submission_df

        return None

if __name__ == "__main__":
    classifier = TimeSeriesAlertClassifier(sequence_length=30, hidden_size=128, num_layers=2)
    result = classifier.run_pipeline()