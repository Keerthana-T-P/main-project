import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

print("Loading preprocessed data...")
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    
with open('val_data.pkl', 'rb') as f:
    val_data = pickle.load(f)

# (We'll load test_data.pkl later for the final evaluation)

print("Data loaded successfully.")
print(train_data)

# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Training on device: {device} ---")
train_data = train_data.to(device)
val_data = val_data.to(device)

class HNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()
        
        # We use a simple Linear layer to project the initial random features
        # into the hidden_channels dimension
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # The HGTConv layer automatically handles all node and edge types
            # by using the graph's metadata
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads)  # <-- FIXED!
            self.convs.append(conv)

        # A final linear layer for the output embeddings
        self.out_lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 1. Project initial features
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        
        # 2. Run through HGT layers (the "multi-view" message passing)
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
        # 3. Get final embeddings
        x_dict = {
            node_type: self.out_lin(x)
            for node_type, x in x_dict.items()
        }
        
        return x_dict

# This is a 'Link Predictor' model that *uses* the HNN
class LinkPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We'll just use a simple dot product to score the links
        
    def forward(self, x_dict, edge_label_index):
        # Get the 'chemical' and 'disease' embeddings
        chem_emb = x_dict['chemical']
        dis_emb = x_dict['disease']
        
        # Get the embeddings for the links we want to predict
        chem_src = chem_emb[edge_label_index[0]]
        dis_dst = dis_emb[edge_label_index[1]]
        
        # Calculate the dot product (our prediction score)
        pred = (chem_src * dis_dst).sum(dim=-1)
        return pred

# --- Model Initialization ---
# These are hyperparameters you can tune
HIDDEN_DIM = 64
OUT_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 2
# --- Find this section in train.py ---
print("Data loaded successfully.")
print(train_data)

# --- ADD THIS FIX: Add self-loops for 'chemical' nodes ---
# This ensures HGTConv calculates and returns embeddings for 'chemical' nodes
num_chem_nodes = train_data['chemical'].x.size(0)
chem_self_loop_edge_index = torch.arange(num_chem_nodes).view(1, -1).repeat(2, 1)

# Add to train_data
train_data['chemical', 'self_loop', 'chemical'].edge_index = chem_self_loop_edge_index
# Add to val_data
val_data['chemical', 'self_loop', 'chemical'].edge_index = chem_self_loop_edge_index
print("Added self-loops for 'chemical' nodes.")
# --------------------------------------------------------

# Move data to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Training on device: {device} ---")
train_data = train_data.to(device)
val_data = val_data.to(device)

# ... (rest of your script) ... 

# Initialize the HNN encoder and the link predictor
encoder = HNN(HIDDEN_DIM, OUT_DIM, NUM_HEADS, NUM_LAYERS, train_data).to(device)
predictor = LinkPredictor().to(device)

# --- Optimizer and Loss ---
# Combine parameters from both models
parameters = list(encoder.parameters()) + list(predictor.parameters())
optimizer = Adam(parameters, lr=0.001)

# Binary Cross-Entropy with Logits Loss (good for 1/0 classification)
loss_fn = torch.nn.BCEWithLogitsLoss()

# --- Training Function ---
def train():
    encoder.train()
    predictor.train()
    
    optimizer.zero_grad()
    
    # 1. Get embeddings from the HNN
    x_dict = encoder(train_data.x_dict, train_data.edge_index_dict)
    
    # 2. Get predictions for the training links
    # This is the (chemical, treats, disease) edge_label_index
    edge_label_index = train_data['chemical', 'treats', 'disease'].edge_label_index
    pred = predictor(x_dict, edge_label_index)
    
    # 3. Calculate loss
    # We use .float() to match the prediction data type
    edge_label = train_data['chemical', 'treats', 'disease'].edge_label.float()
    loss = loss_fn(pred, edge_label)
    
    # 4. Backpropagate
    loss.backward()
    optimizer.step()
    
    return float(loss)

# --- Evaluation Function ---
@torch.no_grad()  # We don't need to calculate gradients during evaluation
def test(data):
    encoder.eval()
    predictor.eval()
    
    # Get embeddings
    x_dict = encoder(data.x_dict, data.edge_index_dict)
    
    # Get predictions for the validation/test links
    edge_label_index = data['chemical', 'treats', 'disease'].edge_label_index
    pred = predictor(x_dict, edge_label_index)
    
    edge_label = data['chemical', 'treats', 'disease'].edge_label
    
    # Calculate the AUC score
    # We use .sigmoid() to convert raw scores (logits) to probabilities
    pred_probs = pred.sigmoid().cpu().numpy()
    edge_label = edge_label.cpu().numpy()
    
    return roc_auc_score(edge_label, pred_probs)

print("Starting training...")
for epoch in range(1, 101):  # Run for 100 epochs (you can change this)
    loss = train()
    
    # Check performance on the validation set every 5 epochs
    if epoch % 5 == 0:
        val_auc = test(val_data)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

print("Training finished!")

# --- Final Test ---
# After training, load the test set and get the final score
with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
test_data = test_data.to(device)

test_auc = test(test_data)
print(f"Final Test AUC: {test_auc:.4f}")