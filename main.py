
"""
Quick Run Script — Train & Evaluate Drug–Disease Classifier
(using saved graph + embeddings)
"""

import pickle
import numpy as np
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import random

# -----------------------------
# Load processed Hetionet data
# -----------------------------
print("📂 Loading processed Hetionet graph...")
with open("processed_hetionet.pkl", "rb") as f:
    node2id, node_types, edge_list, rel_types = pickle.load(f)

print("🔁 Loading Node2Vec embeddings...")
model = KeyedVectors.load("hetionet_node2vec.kv", mmap='r')

# -----------------------------
# Prepare Drug–Disease dataset
# -----------------------------
# Positive edges: known treatments
pos_edges = [(src, dst) for (src, rel, dst) in edge_list if rel == "treats"]

# Identify drugs and diseases
drugs = [nid for nid, typ in node_types.items() if typ == "Compound"]
diseases = [nid for nid, typ in node_types.items() if typ == "Disease"]

# Negative edges: random drug–disease pairs not in known edges
neg_edges = set()
while len(neg_edges) < len(pos_edges):
    d = random.choice(drugs)
    di = random.choice(diseases)
    if (node2id[d], "treats", node2id[di]) not in edge_list:
        neg_edges.add((node2id[d], node2id[di]))
neg_edges = list(neg_edges)

edges = pos_edges + neg_edges
labels = [1]*len(pos_edges) + [0]*len(neg_edges)

print(f"✅ Dataset: {len(pos_edges)} positive, {len(neg_edges)} negative edges.")

# -----------------------------
# Helper: get node embedding
# -----------------------------
def get_embedding(node_id, model):
    node_str = str(node_id)
    if hasattr(model, "key_to_index"):
        in_vocab = node_str in model.key_to_index
    else:
        in_vocab = node_str in model.vocab
    return model[node_str] if in_vocab else np.zeros(model.vector_size)

# -----------------------------
# Build edge feature matrix
# -----------------------------
def build_features(edges, model):
    X = []
    for src, dst in edges:
        emb_src = get_embedding(src, model)
        emb_dst = get_embedding(dst, model)
        # Feature options: concatenate + Hadamard product
        feat = np.concatenate([emb_src, emb_dst, emb_src * emb_dst])
        X.append(feat)
    return np.array(X)

X = build_features(edges, model)
y = np.array(labels)
print("🧮 Feature matrix shape:", X.shape)

# -----------------------------
# Train and evaluate classifier
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"📈 ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("✅ Classifier training and evaluation complete.")
