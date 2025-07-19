# -- coding: utf-8 --
"""
Loan Prediction HFL Simulation (Colab, 10k Sample, 12 Clients/10 Sampled, Uneven Non-IID, 10 Rounds - Corrected Syntax 3)
"""

# --- Essential Imports ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR # Using constant LR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import copy
import time
import io
import warnings
import matplotlib.pyplot as plt
import random

# Ignore specific warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print(f"PyTorch Version: {torch._version_}")

# --- Configuration Constants ---
print("\n--- Configuration ---")
DATASET_FILENAME = 'Loan_default.csv'
SAMPLE_SIZE = 10000
NUM_CLIENTS = 12
NUM_MIDDLE_SERVERS = 3
COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_SAMPLED_CLIENTS_PER_ROUND = 10
BATCH_SIZE = 32
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 'Default'

print(f"Total Clients: {NUM_CLIENTS}, Middle Servers: {NUM_MIDDLE_SERVERS}")
print(f"Sampled Clients per Round: {NUM_SAMPLED_CLIENTS_PER_ROUND}")
print(f"Rounds: {COMMUNICATION_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE} (Constant)")
print(f"Target Column: '{TARGET_COLUMN}'")
print(f"Dataset Sample Size: {SAMPLE_SIZE}")

# --- 1. Data Loading, Sampling, and Preprocessing ---
print("\n--- 1. Data Loading, Sampling, and Preprocessing ---")
try:
    df_full = pd.read_csv(DATASET_FILENAME)
    print(f"Full dataset '{DATASET_FILENAME}' loaded. Shape: {df_full.shape}")
except FileNotFoundError: raise FileNotFoundError(f"ERROR: Dataset file '{DATASET_FILENAME}' not found.")
except Exception as e: raise RuntimeError(f"Error reading dataset file: {e}") from e

if len(df_full) >= SAMPLE_SIZE:
    print(f"Sampling {SAMPLE_SIZE} entries...")
    df = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
else:
    print(f"Dataset smaller than {SAMPLE_SIZE}, using full data.")
    df = df_full.reset_index(drop=True)
print(f"Using dataset shape: {df.shape}")

def preprocess_data(df_input, target_col_name):
    print(f"\nPreprocessing DataFrame of shape: {df_input.shape}")
    if 'LoanID' in df_input.columns: df_processed = df_input.drop('LoanID', axis=1)
    else: df_processed = df_input.copy()
    if target_col_name not in df_processed.columns: raise ValueError(f"Target '{target_col_name}' not found.")
    y = df_processed[target_col_name].copy(); X_df = df_processed.drop(target_col_name, axis=1)
    if not pd.api.types.is_numeric_dtype(y): y = pd.to_numeric(y, errors='coerce')
    if y.isnull().any(): raise ValueError("Target has NaNs after coercion.")
    y = y.astype(int); print(f"Target counts:\n{y.value_counts(normalize=True)}")
    numerical_cols = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_df.select_dtypes(include='object').columns.tolist()
    print(f"Processing {len(numerical_cols)} numerical, {len(categorical_cols)} categorical features.")
    missing_before = X_df.isnull().sum().sum()
    if numerical_cols and X_df[numerical_cols].isnull().sum().sum() > 0:
        num_imputer = SimpleImputer(strategy='median'); X_df[numerical_cols] = num_imputer.fit_transform(X_df[numerical_cols])
    if categorical_cols and X_df[categorical_cols].isnull().sum().sum() > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent'); X_df[categorical_cols] = cat_imputer.fit_transform(X_df[categorical_cols])
    missing_after = X_df.isnull().sum().sum(); print(f"Imputation complete ({missing_before - missing_after} imputed).")
    if categorical_cols: X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True, dtype=float)
    else: X_encoded = X_df.copy()
    print(f"Shape after encoding: {X_encoded.shape}")
    final_numeric_cols = []
    for col in X_encoded.columns:
        if not pd.api.types.is_numeric_dtype(X_encoded[col]): X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
        final_numeric_cols.append(col)
    if final_numeric_cols: scaler = StandardScaler(); X_encoded[final_numeric_cols] = scaler.fit_transform(X_encoded[final_numeric_cols]); print("Features standardized.")
    X_processed_np = X_encoded.values.astype(np.float32); y_processed_np = y.values
    print("Preprocessing complete."); return X_processed_np, y_processed_np, X_encoded.shape[1]

try:
    X_processed, y_processed, INPUT_DIM = preprocess_data(df, TARGET_COLUMN)
    print(f"Final Input Dimension: {INPUT_DIM}")
except Exception as e: print(f"Preprocessing error: {e}"); raise

print("\n--- Splitting Sampled Data into Train/Test ---")
stratify_option = y_processed if np.unique(y_processed).size > 1 else None
try:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_processed, y_processed, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=stratify_option)
    print(f"Split: Train ({X_train_full.shape[0]}), Test ({X_test.shape[0]})")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32); y_test_tensor = torch.tensor(y_test, dtype=torch.long)
except Exception as e: print(f"Split/Tensor error: {e}"); raise

# --- 2. Define Neural Network ---
print("\n--- 2. Defining Neural Network ---")
class LoanPredictorNN(nn.Module):
    def _init_(self, input_dim):
        super(LoanPredictorNN, self)._init_()
        self.layer_1 = nn.Linear(input_dim, 256); self.relu_1 = nn.ReLU(); self.batchnorm_1 = nn.BatchNorm1d(256); self.dropout_1 = nn.Dropout(0.5)
        self.layer_2 = nn.Linear(256, 128); self.relu_2 = nn.ReLU(); self.batchnorm_2 = nn.BatchNorm1d(128); self.dropout_2 = nn.Dropout(0.4)
        self.layer_3 = nn.Linear(128, 64); self.relu_3 = nn.ReLU(); self.batchnorm_3 = nn.BatchNorm1d(64); self.dropout_3 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, 2)
    def forward(self, x):
        is_eval_batch_one = not self.training and x.shape[0] <= 1
        x = self.layer_1(x); x = self.relu_1(x) if is_eval_batch_one else self.relu_1(self.batchnorm_1(x)); x = self.dropout_1(x)
        x = self.layer_2(x); x = self.relu_2(x) if is_eval_batch_one else self.relu_2(self.batchnorm_2(x)); x = self.dropout_2(x)
        x = self.layer_3(x); x = self.relu_3(x) if is_eval_batch_one else self.relu_3(self.batchnorm_3(x)); x = self.dropout_3(x)
        x = self.output_layer(x); return x
print("Neural Network defined.")

# --- 3. Client Data Splitting (Uneven Non-IID) ---
print("\n--- 3. Splitting Training Data for Clients (Uneven Non-IID Distribution) ---")
def split_data_non_iid_uneven(X, y, num_clients, min_samples=20, power_law_factor=1.6):
    n_samples = len(X); y_np = y if isinstance(y, np.ndarray) else np.array(y)
    if n_samples < num_clients * min_samples: min_samples = max(1, n_samples // num_clients // 2)
    proportions = np.random.power(power_law_factor, num_clients); proportions = proportions / proportions.sum()
    client_sizes = (proportions * n_samples).astype(int)
    below_min_indices = np.where(client_sizes < min_samples)[0]; needed = (min_samples - client_sizes[below_min_indices]).sum()
    client_sizes[below_min_indices] = min_samples
    potential_donor_indices = np.where(client_sizes > min_samples + 1)[0]; available_to_donate = (client_sizes[potential_donor_indices] - (min_samples + 1)).sum()
    if needed > available_to_donate and available_to_donate > 0:
        donate_proportions = client_sizes[potential_donor_indices] - (min_samples + 1); donate_amounts = (donate_proportions / available_to_donate * needed).astype(int)
        client_sizes[potential_donor_indices] -= donate_amounts; needed -= donate_amounts.sum()
    donated_total = 0
    while donated_total < needed:
        eligible_donors = np.where(client_sizes > min_samples)[0];
        if not eligible_donors.size: break
        donor_idx = np.random.choice(eligible_donors); client_sizes[donor_idx] -= 1; donated_total += 1
    size_diff = n_samples - client_sizes.sum(); adjustment_indices = np.random.choice(num_clients, abs(size_diff)); client_sizes[adjustment_indices] += np.sign(size_diff)
    client_sizes = np.maximum(0, client_sizes); client_sizes = (client_sizes / client_sizes.sum() * n_samples).astype(int); final_diff = n_samples - client_sizes.sum()
    if final_diff != 0: client_sizes[np.random.choice(num_clients, abs(final_diff))] += np.sign(final_diff)
    if client_sizes.sum() != n_samples: client_sizes[0] += (n_samples - client_sizes.sum()) # Final ensure
    assert client_sizes.sum() == n_samples, f"Final size mismatch: {client_sizes.sum()} vs {n_samples}"
    indices = np.arange(n_samples); np.random.seed(RANDOM_STATE); np.random.shuffle(indices)
    client_data = {}; current_idx = 0
    print(f"Splitting {n_samples} shuffled samples unevenly among {num_clients} clients.")
    for i in range(num_clients):
        client_size = client_sizes[i]; client_indices = indices[current_idx : current_idx + client_size]
        if client_size == 0: client_data[f'client_{i+1}'] = (torch.empty(0, X.shape[1], dtype=torch.float32), torch.empty(0, dtype=torch.long)); print(f"  - Client {i+1}: 0 samples.")
        else:
            client_X = X[client_indices]; client_y = y_np[client_indices]
            client_data[f'client_{i+1}'] = (torch.tensor(client_X, dtype=torch.float32), torch.tensor(client_y, dtype=torch.long))
            unique, counts = np.unique(client_y, return_counts=True); dist_str = ", ".join([f"Cls {cls}:{cnt}" for cls, cnt in zip(unique, counts)])
            print(f"  - Client {i+1}: {client_size} samples. Labels:({dist_str})")
        current_idx += client_size
    print("Uneven Non-IID client data split complete.")
    return client_data

clients_data = split_data_non_iid_uneven(X_train_full, y_train_full, num_clients=NUM_CLIENTS, min_samples=20)

# --- 4. Client Class ---
print("\n--- 4. Defining Client Class ---")
class Client:
    def _init_(self, client_id, data, input_dim, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
        self.client_id = client_id; self.batch_size = batch_size; self.X, self.y = data; self.has_data = len(self.X) > 0; self.data_loader = None
        if self.has_data: dataset = torch.utils.data.TensorDataset(self.X, self.y); self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.model = LoanPredictorNN(input_dim); self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    def train(self, epochs=LOCAL_EPOCHS):
        if not self.has_data or not self.data_loader: return 0.0
        self.model.train(); epoch_loss = 0.0
        for epoch in range(epochs):
            current_epoch_loss = 0.0; num_batches = 0
            for batch_X, batch_y in self.data_loader:
                self.optimizer.zero_grad(); outputs = self.model(batch_X); loss = self.criterion(outputs, batch_y)
                loss.backward(); self.optimizer.step(); current_epoch_loss += loss.item(); num_batches += 1
            epoch_loss = current_epoch_loss / max(1, num_batches)
        return epoch_loss
    def get_model_params(self): return copy.deepcopy(self.model.state_dict()) if self.has_data else None
    def set_model_params(self, params):
         if params is not None:
            try: self.model.load_state_dict(params)
            except Exception as e: print(f"Err loading state Client {self.client_id}: {e}")
    def evaluate_on_global_test(self, X_test_tensor, y_test_tensor, global_batch_size):
        if not self.has_data: return 0.0, 0.0
        self.model.eval(); all_preds, all_labels = [], []
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        is_bn = any(isinstance(m, nn.BatchNorm1d) for m in self.model.modules()); test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=global_batch_size, drop_last=(is_bn and global_batch_size > 1 and len(X_test_tensor) % global_batch_size == 1))
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X); _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy()); all_labels.extend(batch_y.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds) * 100; f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        return accuracy, f1
print("Client class defined.")

# --- 5. Middle Server Class ---
print("\n--- 5. Defining Middle Server Class ---")
class MiddleServer:
    def _init_(self, server_id, input_dim):
        self.server_id = server_id; self.clients = []; self.has_clients = False; self.model = LoanPredictorNN(input_dim)
    def assign_clients_for_round(self, clients): self.clients = [c for c in clients if c.has_data]; self.has_clients = len(self.clients) > 0
    def aggregate_client_models(self, client_params_list):
        if not self.has_clients or not client_params_list: return None
        valid_params = [p for p in client_params_list if p is not None] # Filter out None
        if not valid_params: # <<<--- Check if the filtered list is empty
            # print(f"Warn MidServ {self.server_id}: No valid client models to aggregate.")
            return None # No valid models to aggregate from
        agg_params = copy.deepcopy(valid_params[0]) # Now this is safe
        for key in agg_params:
            tensors = [p[key].float().cpu() for p in valid_params]
            if tensors: agg_params[key] = torch.stack(tensors, dim=0).mean(dim=0)
        try: self.model.load_state_dict(agg_params); return agg_params
        except Exception as e: print(f"Err load agg MidServ {self.server_id}: {e}"); return None
    def distribute_model_to_clients(self, params):
        if not self.has_clients or params is None: return; [c.set_model_params(copy.deepcopy(params)) for c in self.clients]
    def get_model_params(self): return copy.deepcopy(self.model.state_dict()) if self.has_clients else None
    def set_model_params(self, params):
        if params is not None:
            try: self.model.load_state_dict(params)
            except Exception as e: print(f"Err load state MidServ {self.server_id}: {e}")
print("Middle Server class defined.")

# --- 6. Global Server Class ---
print("\n--- 6. Defining Global Server Class ---")
class GlobalServer:
    def _init_(self, middle_servers, input_dim):
        self.middle_servers = middle_servers; self.model = LoanPredictorNN(input_dim); print(f"Global Server init managing {len(self.middle_servers)} middle servers.")
    def aggregate_middle_server_models(self, middle_params_list):
        if not self.middle_servers or not middle_params_list: return None
        valid_params = [p for p in middle_params_list if p is not None] # Filter out None
        if not valid_params: # <<<--- Check if the filtered list is empty
            # print(f"Warn Global Server: No valid middle server models to aggregate.")
            return None # No valid models to aggregate
        global_params = copy.deepcopy(valid_params[0]) # Now safe
        for key in global_params:
            tensors = [p[key].float().cpu() for p in valid_params]
            if tensors: global_params[key] = torch.stack(tensors, dim=0).mean(dim=0)
        try: self.model.load_state_dict(global_params); return global_params
        except Exception as e: print(f"Err load agg Global Server: {e}"); return None
    def distribute_model_to_middle_servers(self, params):
        if params is None: return; [s.set_model_params(copy.deepcopy(params)) for s in self.middle_servers]
    def get_model_params(self): return copy.deepcopy(self.model.state_dict())
    def evaluate_global_model(self, X_test_tensor, y_test_tensor):
        self.model.eval(); criterion = nn.CrossEntropyLoss(); all_preds, all_labels, all_probs_class1 = [], [], []; test_loss = 0.0; num_batches = 0
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor); test_batch_size = min(BATCH_SIZE * 4, len(X_test_tensor))
        is_bn = any(isinstance(m, nn.BatchNorm1d) for m in self.model.modules()); test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, drop_last=(is_bn and test_batch_size > 1 and len(X_test_tensor) % test_batch_size == 1))
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X); loss = criterion(outputs, batch_y); test_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1); _, predicted = torch.max(probabilities, 1)
                all_preds.extend(predicted.cpu().numpy()); all_labels.extend(batch_y.cpu().numpy()); all_probs_class1.extend(probabilities[:, 1].cpu().numpy())
                num_batches += 1
        avg_loss = test_loss / max(1, num_batches); all_labels_np, all_preds_np = np.array(all_labels), np.array(all_preds)
        accuracy = accuracy_score(all_labels_np, all_preds_np) * 100; f1 = f1_score(all_labels_np, all_preds_np, average='binary', zero_division=0); auc = 0.0
        if len(np.unique(all_labels_np)) > 1:
             try: auc = roc_auc_score(all_labels_np, np.array(all_probs_class1))
             except Exception as e: print(f"  - AUC calc error: {e}")
        return accuracy, avg_loss, f1, auc
print("Global Server class defined.")

# --- 7. System Setup ---
print("\n--- 7. Setting up Federated System ---")
start_time_setup = time.time()
all_clients = []; [all_clients.append(Client(client_id=i+1, data=clients_data.get(f'client_{i+1}', (torch.empty(0, INPUT_DIM, dtype=torch.float32), torch.empty(0, dtype=torch.long))), input_dim=INPUT_DIM, batch_size=BATCH_SIZE, lr=LEARNING_RATE)) for i in range(NUM_CLIENTS)]
NUM_ACTIVE_CLIENTS_INIT = sum(1 for c in all_clients if c.has_data); print(f"Initialized {len(all_clients)} clients ({NUM_ACTIVE_CLIENTS_INIT} with data).")
middle_servers = []; [middle_servers.append(MiddleServer(server_id=i+1, input_dim=INPUT_DIM)) for i in range(NUM_MIDDLE_SERVERS)]
print(f"Initialized {len(middle_servers)} middle servers.")
if not middle_servers: raise SystemExit("No middle servers initialized.")
global_server = GlobalServer(middle_servers=middle_servers, input_dim=INPUT_DIM)
end_time_setup = time.time(); print(f"System setup took: {end_time_setup - start_time_setup:.2f}s")

# --- 8. Hierarchical Federated Training Loop ---
print("\n--- 8. Starting Hierarchical Federated Training ---")
history = {'round': [], 'acc': [], 'loss': [], 'f1': [], 'auc': [], 'time': [], 'avg_client_loss': []}
total_training_start_time = time.time()
current_global_params = global_server.get_model_params()

for r in range(COMMUNICATION_ROUNDS): # Runs for 10 rounds
    round_start_time = time.time(); print(f"\n--- Round {r+1}/{COMMUNICATION_ROUNDS} ---")
    # Sample clients
    clients_with_data = [c for c in all_clients if c.has_data]
    if len(clients_with_data) < NUM_SAMPLED_CLIENTS_PER_ROUND: active_clients_this_round = clients_with_data
    else: active_clients_this_round = random.sample(clients_with_data, NUM_SAMPLED_CLIENTS_PER_ROUND)
    num_active_this_round = len(active_clients_this_round); print(f"  Selected {num_active_this_round} clients.")
    if not active_clients_this_round: print("  No clients active. Skipping."); history['time'].append(time.time() - round_start_time); continue
    # Assign clients to servers
    num_middle_servers_active = len(middle_servers); [s.assign_clients_for_round([]) for s in middle_servers] # Clear
    client_assignment_indices = list(range(num_active_this_round)); random.shuffle(client_assignment_indices); cursor = 0
    for i in range(num_middle_servers_active):
        server = middle_servers[i]; num_for_this = num_active_this_round // num_middle_servers_active + (1 if i < num_active_this_round % num_middle_servers_active else 0)
        end_cursor = min(cursor + num_for_this, num_active_this_round); assigned_objects = [active_clients_this_round[k] for k in client_assignment_indices[cursor:end_cursor]]
        server.assign_clients_for_round(assigned_objects); cursor = end_cursor
    # Distribute model
    global_server.distribute_model_to_middle_servers(current_global_params)
    for server in middle_servers:
        if server.has_clients: server.distribute_model_to_clients(server.get_model_params())
    # Clients train
    client_params_map = {}; client_losses = []
    for client in active_clients_this_round:
        if client.has_data: loss = client.train(epochs=LOCAL_EPOCHS); client_params_map[client.client_id] = client.get_model_params(); client_losses.append(loss)
    avg_client_loss = np.mean([l for l in client_losses if l is not None and not np.isnan(l)]) if client_losses else 0; print(f"  Avg Loss: {avg_client_loss:.4f}")
    # Middle aggregate
    middle_aggregated_params_list = []
    for server in middle_servers:
         if server.has_clients:
             params_for_this_server = [client_params_map.get(c.client_id) for c in server.clients]
             aggregated_params = server.aggregate_client_models(params_for_this_server)
             if aggregated_params is not None: middle_aggregated_params_list.append(aggregated_params)
    # Global aggregate
    current_global_params = global_server.aggregate_middle_server_models(middle_aggregated_params_list)
    if current_global_params is None: print("  Warn: Global agg failed. Using previous."); current_global_params = global_server.get_model_params()
    # Evaluate Global Model
    global_accuracy, global_loss, global_f1, global_auc = global_server.evaluate_global_model(X_test_tensor, y_test_tensor)
    history['round'].append(r+1); history['acc'].append(global_accuracy); history['loss'].append(global_loss);
    history['f1'].append(global_f1); history['auc'].append(global_auc); history['time'].append(time.time() - round_start_time); history['avg_client_loss'].append(avg_client_loss)
    print(f"  Round {r+1} done in {history['time'][-1]:.2f}s."); print(f"  Global Test --> Acc:{global_accuracy:.2f}% | Loss:{global_loss:.4f} | F1:{global_f1:.4f} | AUC:{global_auc:.4f}")
    # Reset middle server lists
    for server in middle_servers: server.assign_clients_for_round([])

# --- Post-Training ---
total_training_end_time = time.time(); total_training_duration = total_training_end_time - total_training_start_time
avg_round_time = np.mean(history['time']) if history['time'] else 0
print("\n--- Federated Training Finished ---"); print(f"Total time: {total_training_duration:.2f}s | Avg round: {avg_round_time:.2f}s")
# Final Eval Global
final_accuracy, final_loss, final_f1, final_auc = global_server.evaluate_global_model(X_test_tensor, y_test_tensor)
print(f"\n--- Final Global Model Evaluation ---"); print(f"Acc: {final_accuracy:.2f}% | Loss: {final_loss:.4f} | F1: {final_f1:.4f} | AUC: {final_auc:.4f}")
# Final Eval Clients
print("\n--- Evaluating Final Client Models on Global Test Set ---")
client_test_accuracies = []; client_test_f1s = []
for client in all_clients:
    if client.has_data: acc, f1 = client.evaluate_on_global_test(X_test_tensor, y_test_tensor, BATCH_SIZE * 4); client_test_accuracies.append(acc); client_test_f1s.append(f1); print(f"  Client {client.client_id}: Acc={acc:.2f}%, F1={f1:.4f}")
if client_test_accuracies: print(f"\nAvg Client Acc (Global Test): {np.mean(client_test_accuracies):.2f}% (StdDev: {np.std(client_test_accuracies):.2f}%)"); print(f"Avg Client F1 (Global Test): {np.mean(client_test_f1s):.4f} (StdDev: {np.std(client_test_f1s):.4f})")

# --- Plotting ---
print("\n--- 11. Plotting Results ---")
try:
    rounds_range = history['round']; plt.figure(figsize=(24, 12))
    if not rounds_range: raise ValueError("No rounds.")
    plt.subplot(2,4,1); plt.plot(rounds_range, history['acc'], 'bo-', label='Accuracy'); plt.axhline(90, c='grey', ls='--', label='90% Target'); plt.title('Global Accuracy'); plt.xlabel('Round'); plt.ylabel('Acc %'); plt.grid(True,ls=':'); plt.legend(); plt.xticks(np.arange(0,max(rounds_range)+1,step=max(1,len(rounds_range)//5)))
    plt.subplot(2,4,2); plt.plot(rounds_range, history['loss'],'rs-',label='Loss'); plt.title('Global Loss'); plt.xlabel('Round'); plt.ylabel('Loss'); plt.grid(True,ls=':'); plt.legend(); plt.xticks(np.arange(0,max(rounds_range)+1,step=max(1,len(rounds_range)//5)))
    plt.subplot(2,4,3); plt.plot(rounds_range, history['f1'],'g^-',label='F1'); plt.title('Global F1'); plt.xlabel('Round'); plt.ylabel('F1 Score'); plt.grid(True,ls=':'); plt.legend(); plt.xticks(np.arange(0,max(rounds_range)+1,step=max(1,len(rounds_range)//5))); plt.ylim(0,1)
    plt.subplot(2,4,4); plt.plot(rounds_range, history['auc'],'p-',c='purple',label='AUC'); plt.title('Global AUC'); plt.xlabel('Round'); plt.ylabel('AUC'); plt.grid(True,ls=':'); plt.legend(); plt.xticks(np.arange(0,max(rounds_range)+1,step=max(1,len(rounds_range)//5))); plt.ylim(0,1)
    plt.subplot(2,4,5); plt.plot(rounds_range, history['avg_client_loss'],'x-',c='orange',label='Avg Client Loss'); plt.title('Avg Client Loss'); plt.xlabel('Round'); plt.ylabel('Loss'); plt.grid(True,ls=':'); plt.legend(); plt.xticks(np.arange(0,max(rounds_range)+1,step=max(1,len(rounds_range)//5)))
    if client_test_accuracies: plt.subplot(2,4,6); plt.hist(client_test_accuracies,bins=max(5,NUM_CLIENTS//2),edgecolor='k'); plt.title('Final Client Acc Distr.'); plt.xlabel('Acc %'); plt.ylabel('# Clients'); plt.grid(True,axis='y',ls=':'); mean_acc = np.mean(client_test_accuracies); plt.axvline(mean_acc, c='r',ls='--',lw=1,label=f'Mean:{mean_acc:.2f}%'); plt.legend()
    if client_test_f1s: plt.subplot(2,4,7); plt.hist(client_test_f1s,bins=max(5,NUM_CLIENTS//2),edgecolor='k',color='green'); plt.title('Final Client F1 Distr.'); plt.xlabel('F1 Score'); plt.ylabel('# Clients'); plt.grid(True,axis='y',ls=':'); mean_f1 = np.mean(client_test_f1s); plt.axvline(mean_f1, c='r',ls='--',lw=1,label=f'Mean:{mean_f1:.4f}'); plt.legend()
    plt.tight_layout(pad=2.5); plt.suptitle('HFL Perf (10k Sample, Uneven Non-IID, 10 Rnds, High Local Epochs)',fontsize=16,y=1.03); plt.show(); print("Plots generated.")
except Exception as e: print(f"Plotting error: {e}")

# --- END OF FILE ---
