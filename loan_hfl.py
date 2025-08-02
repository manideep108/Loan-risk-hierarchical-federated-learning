# -- coding: utf-8 --
"""
Loan Prediction HFL Simulation (Colab, 12 Clients, ALL Sampled, Uneven Non-IID, 10 Rounds)
WITH ENHANCED ACCURACY STRATEGIES (FINAL, BUG-FIXED LOGIC)
"""

# --- Essential Imports ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import copy
import time
import warnings
import matplotlib.pyplot as plt
import random

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
print(f"PyTorch Version: {torch._version_}")

# --- Configuration Constants ---
print("\n--- Configuration ---")
DATASET_FILENAME = 'Loan_default.csv'; SAMPLE_SIZE = 10000; NUM_CLIENTS = 12; NUM_MIDDLE_SERVERS = 3; COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 40; LEARNING_RATE = 0.001; NUM_SAMPLED_CLIENTS_PER_ROUND = NUM_CLIENTS; BATCH_SIZE = 32; TEST_SPLIT_RATIO = 0.2; RANDOM_STATE = 42; TARGET_COLUMN = 'Default'
MIDDLE_SERVER_ACCURACY_THRESHOLD = 80.0
print(f"Total Clients: {NUM_CLIENTS}, Middle Servers: {NUM_MIDDLE_SERVERS}"); print(f"Clients Trained per Round: ALL ({NUM_SAMPLED_CLIENTS_PER_ROUND})")
print(f"Local Epochs: {LOCAL_EPOCHS} (Increased for better convergence)"); print(f"Federated Strategy: Weighted Averaging (FedAvg)")
print(f"Middle Server Exclusion Threshold: {MIDDLE_SERVER_ACCURACY_THRESHOLD}% Avg Client Accuracy (per round)")

# --- 1. Data Loading, Sampling, and Preprocessing ---
print("\n--- 1. Data Loading, Sampling, and Preprocessing ---")
try:
    df_full = pd.read_csv(DATASET_FILENAME)
except FileNotFoundError: raise FileNotFoundError(f"ERROR: Dataset file '{DATASET_FILENAME}' not found.")
if len(df_full) >= SAMPLE_SIZE: df = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
else: df = df_full.reset_index(drop=True)
def preprocess_data(df_input, target_col_name):
    if 'LoanID' in df_input.columns: df_processed = df_input.drop('LoanID', axis=1)
    else: df_processed = df_input.copy()
    y = df_processed[target_col_name].copy(); X_df = df_processed.drop(target_col_name, axis=1); y = pd.to_numeric(y, errors='coerce').astype(int)
    numerical_cols = X_df.select_dtypes(include=np.number).columns.tolist(); categorical_cols = X_df.select_dtypes(include='object').columns.tolist()
    if numerical_cols and X_df[numerical_cols].isnull().sum().sum() > 0: num_imputer = SimpleImputer(strategy='median'); X_df[numerical_cols] = num_imputer.fit_transform(X_df[numerical_cols])
    if categorical_cols and X_df[categorical_cols].isnull().sum().sum() > 0: cat_imputer = SimpleImputer(strategy='most_frequent'); X_df[categorical_cols] = cat_imputer.fit_transform(X_df[categorical_cols])
    if categorical_cols: X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True, dtype=float)
    else: X_encoded = X_df.copy()
    scaler = StandardScaler(); X_encoded[X_encoded.columns] = scaler.fit_transform(X_encoded[X_encoded.columns])
    return X_encoded.values.astype(np.float32), y.values, X_encoded.shape[1]
X_processed, y_processed, INPUT_DIM = preprocess_data(df, TARGET_COLUMN)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_processed, y_processed, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y_processed)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32); y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- 2. Define ENHANCED Neural Network ---
print("\n--- 2. Defining ENHANCED (Deeper) Neural Network ---")
class LoanPredictorNN(nn.Module):
    def _init_(self, input_dim):
        super(LoanPredictorNN, self)._init_()
        self.layer_1 = nn.Linear(input_dim, 256); self.relu_1 = nn.ReLU(); self.batchnorm_1 = nn.BatchNorm1d(256); self.dropout_1 = nn.Dropout(0.5)
        self.layer_2 = nn.Linear(256, 128); self.relu_2 = nn.ReLU(); self.batchnorm_2 = nn.BatchNorm1d(128); self.dropout_2 = nn.Dropout(0.4)
        self.layer_3 = nn.Linear(128, 64); self.relu_3 = nn.ReLU(); self.batchnorm_3 = nn.BatchNorm1d(64); self.dropout_3 = nn.Dropout(0.3)
        self.layer_4 = nn.Linear(64, 32); self.relu_4 = nn.ReLU(); self.batchnorm_4 = nn.BatchNorm1d(32); self.dropout_4 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(32, 2)
    def forward(self, x):
        x = self.layer_1(x); x = self.relu_1(self.batchnorm_1(x)); x = self.dropout_1(x)
        x = self.layer_2(x); x = self.relu_2(self.batchnorm_2(x)); x = self.dropout_2(x)
        x = self.layer_3(x); x = self.relu_3(self.batchnorm_3(x)); x = self.dropout_3(x)
        x = self.layer_4(x); x = self.relu_4(self.batchnorm_4(x)); x = self.dropout_4(x)
        x = self.output_layer(x); return x
print("Neural Network defined.")

# --- 3. Client Data Splitting ---
print("\n--- 3. Splitting Training Data for Clients ---")
def split_data_non_iid_uneven(X, y, num_clients, min_samples=20, power_law_factor=1.6):
    n_samples = len(X); proportions = np.random.power(power_law_factor, num_clients); proportions /= proportions.sum()
    client_sizes = (proportions * n_samples).astype(int); client_sizes = np.maximum(min_samples, client_sizes); diff = n_samples - client_sizes.sum()
    if diff > 0: client_sizes[:diff] += 1
    elif diff < 0:
      eligible_indices = np.where(client_sizes > min_samples)[0]
      for i in range(abs(diff)): client_sizes[random.choice(eligible_indices)] -= 1
    client_sizes[-1] += (n_samples-client_sizes.sum()); assert client_sizes.sum() == n_samples
    indices = np.arange(n_samples); np.random.seed(RANDOM_STATE); np.random.shuffle(indices); client_data = {}; current_idx = 0
    for i in range(num_clients):
        client_size = client_sizes[i]; client_indices = indices[current_idx : current_idx + client_size]; client_X = X[client_indices]; client_y = np.array(y)[client_indices]
        client_data[f'client_{i+1}'] = (torch.tensor(client_X, dtype=torch.float32), torch.tensor(client_y, dtype=torch.long)); current_idx += client_size
    return client_data
clients_data = split_data_non_iid_uneven(X_train_full, y_train_full, num_clients=NUM_CLIENTS)
print("Data splitting complete.")

# --- 4. Client Class ---
print("\n--- 4. Defining Client Class ---")
class Client:
    def _init_(self, client_id, data, input_dim, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
        self.client_id = client_id; self.batch_size = batch_size; self.X, self.y = data; self.has_data = len(self.X) > 0; self.data_loader = None
        if self.has_data: self.data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.X, self.y), batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.model = LoanPredictorNN(input_dim); self.optimizer = optim.Adam(self.model.parameters(), lr=lr); self.criterion = nn.CrossEntropyLoss()
    def train(self, epochs=LOCAL_EPOCHS):
        if not self.has_data or not self.data_loader: return 0.0
        self.model.train(); total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0; num_batches = 0
            for batch_X, batch_y in self.data_loader:
                self.optimizer.zero_grad(); outputs = self.model(batch_X); loss = self.criterion(outputs, batch_y); loss.backward(); self.optimizer.step()
                epoch_loss += loss.item(); num_batches += 1
            total_loss += epoch_loss
        return total_loss / (epochs * max(1, num_batches))
    def get_model_params(self): return copy.deepcopy(self.model.state_dict()) if self.has_data else None
    def set_model_params(self, params):
        if params is not None:
            try: self.model.load_state_dict(params)
            except Exception as e: print(f"Err loading state Client {self.client_id}: {e}")
    def evaluate_on_global_test(self, X_test_tensor, y_test_tensor, global_batch_size):
        if not self.has_data: return 0.0, 0.0
        self.model.eval(); all_preds, all_labels = [], []
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor); drop_last = len(X_test_tensor) % global_batch_size == 1 and global_batch_size > 1
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=global_batch_size, drop_last=drop_last)
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X); _, predicted = torch.max(outputs.data, 1); all_preds.extend(predicted.cpu().numpy()); all_labels.extend(batch_y.cpu().numpy())
        return accuracy_score(all_labels, all_preds) * 100, f1_score(all_labels, all_preds, average='binary', zero_division=0)
print("Client Class defined.")

# --- 5. Middle Server Class ---
print("\n--- 5. Defining Middle Server Class ---")
class MiddleServer:
    def _init_(self, server_id, input_dim):
        self.server_id = server_id; self.clients = []; self.has_clients = False
        self.model = LoanPredictorNN(input_dim); self.data_size = 0
    def assign_clients(self, clients):
        self.clients = [c for c in clients if c.has_data]; self.has_clients = len(self.clients) > 0
        self.data_size = sum(len(c.X) for c in self.clients)
        if self.has_clients: print(f"  Middle Server {self.server_id} assigned clients: {[c.client_id for c in self.clients]} (Total data: {self.data_size})")
    def aggregate_client_models(self, client_params_list):
        if not self.has_clients: return None
        valid_params = [p for p in client_params_list if p is not None]
        if not valid_params: return None
        agg_params = copy.deepcopy(valid_params[0])
        for key in agg_params:
            if agg_params[key] is not None:
              tensors = [p[key].float().cpu() for p in valid_params]
              if tensors: agg_params[key] = torch.stack(tensors, dim=0).mean(dim=0)
        return agg_params
    def distribute_model_to_clients(self, params):
        if self.has_clients and params is not None: [c.set_model_params(copy.deepcopy(params)) for c in self.clients]
    def get_model_params(self): return copy.deepcopy(self.model.state_dict()) if self.has_clients else None
    def set_model_params(self, params):
        if params is not None:
            try: self.model.load_state_dict(params)
            except Exception as e: print(f"Err load state MidServ {self.server_id}: {e}")
print("Middle Server Class defined.")

# --- 6. Global Server Class ---
print("\n--- 6. Defining Global Server Class (with Weighted Averaging) ---")
class GlobalServer:
    def _init_(self, middle_servers, input_dim): self.middle_servers = middle_servers; self.model = LoanPredictorNN(input_dim)
    def aggregate_middle_server_models(self, middle_params_and_sizes):
        valid_models = [item for item in middle_params_and_sizes if item[0] is not None]
        if not valid_models: return None
        valid_params = [item[0] for item in valid_models]
        data_sizes = np.array([item[1] for item in valid_models], dtype=np.float32)
        total_data = np.sum(data_sizes)
        if total_data == 0: return None
        weights = data_sizes / total_data
        print(f"  Global Aggregation Weights: {np.round(weights, 3)}")
        global_params = copy.deepcopy(valid_params[0])
        for key in global_params.keys():
            weighted_sum = torch.stack([params[key].float() * weight for params, weight in zip(valid_params, weights)], dim=0).sum(dim=0)
            global_params[key] = weighted_sum
        return global_params
    def set_model_params(self, params):
        if params is not None:
            try: self.model.load_state_dict(params)
            except Exception as e: print(f"Err load state Global Server: {e}")
    def distribute_model_to_middle_servers(self, params):
        if params is not None: [s.set_model_params(copy.deepcopy(params)) for s in self.middle_servers]
    def get_model_params(self): return copy.deepcopy(self.model.state_dict())
    def evaluate_global_model(self, X_test_tensor, y_test_tensor):
        self.model.eval(); criterion = nn.CrossEntropyLoss(); all_preds, all_labels, all_probs_class1 = [], [], []; test_loss = 0.0; num_batches = 0
        test_batch_size = min(BATCH_SIZE * 4, len(X_test_tensor)); drop_last = len(X_test_tensor) % test_batch_size == 1 and test_batch_size > 1
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), batch_size=test_batch_size, drop_last=drop_last)
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X); loss = criterion(outputs, batch_y); test_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1); _, predicted = torch.max(probabilities, 1)
                all_preds.extend(predicted.cpu().numpy()); all_labels.extend(batch_y.cpu().numpy()); all_probs_class1.extend(probabilities[:, 1].cpu().numpy())
                num_batches += 1
        avg_loss = test_loss / max(1, num_batches); accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0); auc = 0.0
        if len(np.unique(all_labels)) > 1:
             try: auc = roc_auc_score(all_labels, np.array(all_probs_class1))
             except Exception as e: print(f"  - AUC calc error: {e}")
        return accuracy, avg_loss, f1, auc
print("Global Server Class defined.")

# --- 7. System Setup ---
print("\n--- 7. Setting up Federated System with Static Client Assignment ---")
start_time_setup = time.time(); client_init_data = [clients_data.get(f'client_{i+1}', (torch.empty(0, INPUT_DIM), torch.empty(0, dtype=torch.long))) for i in range(NUM_CLIENTS)]
all_clients = [Client(client_id=i+1, data=client_init_data[i], input_dim=INPUT_DIM) for i in range(NUM_CLIENTS)]
middle_servers = [MiddleServer(server_id=i+1, input_dim=INPUT_DIM) for i in range(NUM_MIDDLE_SERVERS)]
clients_per_server = NUM_CLIENTS // NUM_MIDDLE_SERVERS
for i, server in enumerate(middle_servers): server.assign_clients(all_clients[i*clients_per_server:(i+1)*clients_per_server])
global_server = GlobalServer(middle_servers=middle_servers, input_dim=INPUT_DIM); print(f"System setup took: {time.time() - start_time_setup:.2f}s")

# --- 8. Hierarchical Federated Training Loop ---
print("\n--- 8. Starting Hierarchical Federated Training ---")
history = {'round': [], 'acc': [], 'loss': [], 'f1': [], 'auc': [], 'time': [], 'avg_client_loss': []}
total_training_start_time = time.time()
current_global_params = global_server.get_model_params()

for r in range(COMMUNICATION_ROUNDS):
    round_start_time = time.time(); print(f"\n--- Round {r+1}/{COMMUNICATION_ROUNDS} ---")
    
    all_eligible_clients = [c for s in middle_servers for c in s.clients if c.has_data]
    if not all_eligible_clients: print("No clients available. Stopping."); break
    print(f"  Selected ALL {len(all_eligible_clients)} clients for training.")
    
    # Distribute the "current best" global model parameters to the middle servers
    global_server.distribute_model_to_middle_servers(current_global_params)
    
    # <<< BUG FIX: Distribute the model from middle servers down to their clients >>>
    for server in middle_servers:
        server.distribute_model_to_clients(server.get_model_params())

    client_params_map = {}; client_losses = [client.train(epochs=LOCAL_EPOCHS) for client in all_eligible_clients]
    for client in all_eligible_clients: client_params_map[client.client_id] = client.get_model_params()
    avg_client_loss = np.mean([l for l in client_losses if l]) if client_losses else 0.0
    print(f"  Avg Client Training Loss: {avg_client_loss:.4f}")

    print("  EVALUATING ALL CLIENT MODELS (on global test set):")
    round_client_accuracies_map = {}
    for client in all_clients:
        if client.has_data:
            acc, f1 = client.evaluate_on_global_test(X_test_tensor, y_test_tensor, BATCH_SIZE * 4)
            round_client_accuracies_map[client.client_id] = acc; print(f"    - Client {client.client_id}: Acc={acc:.2f}%, F1={f1:.4f}")

    print("  CHECKING MIDDLE SERVER PERFORMANCE FOR THIS ROUND'S AGGREGATION:")
    params_and_sizes_for_global_aggregation = []
    for server in middle_servers:
        client_ids_for_server = [c.client_id for c in server.clients]
        accuracies_for_server = [round_client_accuracies_map.get(cid, 0) for cid in client_ids_for_server]
        ms_avg_client_acc = np.mean(accuracies_for_server) if accuracies_for_server else 0
        
        if ms_avg_client_acc < MIDDLE_SERVER_ACCURACY_THRESHOLD:
            print(f"  - Middle Server {server.server_id} Avg Client Accuracy: {ms_avg_client_acc:.2f}%. EXCLUDING from this round's global aggregation.")
        else:
            print(f"  - Middle Server {server.server_id} Avg Client Accuracy: {ms_avg_client_acc:.2f}%. INCLUDING in this round's global aggregation.")
            params_for_server = [client_params_map.get(c.client_id) for c in server.clients]
            agg_params = server.aggregate_client_models(params_for_server)
            if agg_params is not None:
                params_and_sizes_for_global_aggregation.append((agg_params, server.data_size))
    
    new_global_params = global_server.aggregate_middle_server_models(params_and_sizes_for_global_aggregation)
    if new_global_params is not None:
        print("  Global model was UPDATED.")
        current_global_params = new_global_params
        global_server.set_model_params(current_global_params)
    else:
        print("  Warn: No servers met the criteria. Global model was NOT updated. Using model from previous round.")
        
    global_accuracy, global_loss, global_f1, global_auc = global_server.evaluate_global_model(X_test_tensor, y_test_tensor)
    round_time = time.time() - round_start_time
    history['round'].append(r+1); history['acc'].append(global_accuracy); history['loss'].append(global_loss);
    history['f1'].append(global_f1); history['auc'].append(global_auc); history['time'].append(round_time)
    history['avg_client_loss'].append(avg_client_loss)
    print(f"  Round {r+1} done in {round_time:.2f}s.")
    print(f"  GLOBAL MODEL -> Acc:{global_accuracy:.2f}% | Loss:{global_loss:.4f} | F1:{global_f1:.4f} | AUC:{global_auc:.4f}")

# --- Post-Training and Plotting ---
print("\n--- Federated Training Finished ---")
total_training_duration = time.time() - total_training_start_time
avg_round_time = np.mean(history['time']) if history['time'] else 0
print(f"Total time: {total_training_duration:.2f}s | Avg round: {avg_round_time:.2f}s")
final_accuracy, final_loss, final_f1, final_auc = global_server.evaluate_global_model(X_test_tensor, y_test_tensor)
print(f"\n--- Final Global Model Evaluation ---\nAcc: {final_accuracy:.2f}% | Loss: {final_loss:.4f} | F1: {final_f1:.4f} | AUC: {final_auc:.4f}")
print("\n--- Evaluating Final Client Models on Global Test Set ---")
client_test_accuracies = []; client_test_f1s = []
if all_clients:
    for client in all_clients:
        acc, f1 = client.evaluate_on_global_test(X_test_tensor, y_test_tensor, BATCH_SIZE * 4)
        client_test_accuracies.append(acc); client_test_f1s.append(f1)
        print(f"  Client {client.client_id}: Acc={acc:.2f}%, F1={f1:.4f}")
    print(f"\nAvg Client Acc (Global Test): {np.mean(client_test_accuracies):.2f}% (StdDev: {np.std(client_test_accuracies):.2f}%)")
    print(f"Avg Client F1 (Global Test): {np.mean(client_test_f1s):.4f} (StdDev: {np.std(client_test_f1s):.4f})")
print("\n--- 11. Plotting Results ---")
try:
    if not history['round']: raise ValueError("No rounds were completed to plot.")
    rounds_range = history['round']; fig, axes = plt.subplots(2, 4, figsize=(24, 12)); ax = axes.ravel()
    step = max(1, len(rounds_range)//5)
    ax[0].plot(rounds_range, history['acc'], 'bo-'); ax[0].set_title('Global Accuracy (%)'); ax[0].axhline(y=90, color='grey', linestyle='--');
    ax[1].plot(rounds_range, history['loss'],'rs-'); ax[1].set_title('Global Loss')
    ax[2].plot(rounds_range, history['f1'],'g^-'); ax[2].set_title('Global F1-Score'); ax[2].set_ylim(0, 1)
    ax[3].plot(rounds_range, history['auc'],'p-',c='purple'); ax[3].set_title('Global AUC'); ax[3].set_ylim(0, 1)
    ax[4].plot(rounds_range, history['avg_client_loss'],'x-',c='orange'); ax[4].set_title('Average Client Training Loss')
    if client_test_accuracies: ax[5].hist(client_test_accuracies,bins=max(5,NUM_CLIENTS//2),edgecolor='k'); mean_acc = np.mean(client_test_accuracies); ax[5].axvline(mean_acc, c='r',ls='--',label=f'Mean:{mean_acc:.2f}'); ax[5].set_title('Final Client Accuracy Dist.'); ax[5].legend()
    if client_test_f1s: ax[6].hist(client_test_f1s,bins=max(5,NUM_CLIENTS//2),edgecolor='k',color='g'); mean_f1 = np.mean(client_test_f1s); ax[6].axvline(mean_f1,c='r',ls='--',label=f'Mean:{mean_f1:.4f}'); ax[6].set_title('Final Client F1-Score Dist.'); ax[6].legend()
    ax[7].axis('off')
    for i in range(7): ax[i].grid(True, linestyle=':'); ax[i].set_xlabel('Communication Round'); ax[i].set_xticks(np.arange(min(rounds_range), max(rounds_range)+1, step=step))
    plt.tight_layout(pad=2.0); plt.suptitle('HFL Performance', fontsize=16, y=1.02); plt.show()
    print("Plots generated.")
except Exception as e: print(f"Plotting error: {e}")
