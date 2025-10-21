import pandas as pd
import numpy as np
import torch
from gensim.models import KeyedVectors
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
import gzip
import bz2
import warnings
import pickle  # <-- ADD THIS LINE
warnings.filterwarnings('ignore') # To hide any warnings from pandas/gensim
# --- 1. Define file paths ---
DATA_DIR = 'data/'
CTD_CHEM_GENE_FILE = DATA_DIR + 'CTD_chem_gene_ixns (1).csv.gz'
CTD_CHEM_DISEASE_FILE = DATA_DIR + 'CTD_curated_chemicals_diseases.csv.gz'
CTD_GENE_DISEASE_FILE = DATA_DIR + 'CTD_curated_genes_diseases.csv.gz'
# ... add all your other file paths here

# --- 2. Define column names (since files have no headers) ---
# Based on official CTD data dictionaries
chem_gene_cols = [
    "ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
    "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"
]

# NEW LIST (correct for your file)
chem_disease_cols = [
    "ChemicalName", "ChemicalID", "CasRN", "DiseaseName", "DiseaseID",
    "DirectEvidence", "PubMedIDs"
]

gene_disease_cols = [
    "GeneSymbol", "GeneID", "DiseaseName", "DiseaseID",
    "DirectEvidence", "InferenceNetwork", "PubMedIDs"
]
# ... define column lists for your other files (BindingDB, MedDRA, etc.)

# --- 3. Load and extract unique node IDs ---

print("Loading chemical IDs...")
chem_ids_1 = pd.read_csv(
    CTD_CHEM_GENE_FILE, compression='gzip', header=None,
    names=chem_gene_cols, usecols=['ChemicalID'], comment='#'
)['ChemicalID']
chem_ids_2 = pd.read_csv(
    CTD_CHEM_DISEASE_FILE, compression='gzip', header=None,
    names=chem_disease_cols, usecols=['ChemicalID'], comment='#'
)['ChemicalID']
all_unique_chemicals = pd.concat([chem_ids_1, chem_ids_2]).unique()

print("Loading gene IDs...")
# Note: CTD GeneID is an integer, so we cast to string to be consistent
gene_ids_1 = pd.read_csv(
    CTD_CHEM_GENE_FILE, compression='gzip', header=None,
    names=chem_gene_cols, usecols=['GeneID'], comment='#'
)['GeneID'].astype(str)
gene_ids_2 = pd.read_csv(
    CTD_GENE_DISEASE_FILE, compression='gzip', header=None,
    names=gene_disease_cols, usecols=['GeneID'], comment='#'
)['GeneID'].astype(str)
all_unique_genes = pd.concat([gene_ids_1, gene_ids_2]).unique()

print("Loading disease IDs...")
disease_ids_1 = pd.read_csv(
    CTD_CHEM_DISEASE_FILE, compression='gzip', header=None,
    names=chem_disease_cols, usecols=['DiseaseID'], comment='#'
)['DiseaseID']
disease_ids_2 = pd.read_csv(
    CTD_GENE_DISEASE_FILE, compression='gzip', header=None,
    names=gene_disease_cols, usecols=['DiseaseID'], comment='#'
)['DiseaseID']
all_unique_diseases = pd.concat([disease_ids_1, disease_ids_2]).unique()

# ... Repeat this process for Pathways and Side Effects from their respective files

# --- 4. Create the integer mappings ---
chem_to_int_id = {mesh_id: i for i, mesh_id in enumerate(all_unique_chemicals)}
gene_to_int_id = {gene_id: i for i, gene_id in enumerate(all_unique_genes)}
disease_to_int_id = {disease_id: i for i, disease_id in enumerate(all_unique_diseases)}
# ... and so on for other node types

print(f"Total Chemicals: {len(chem_to_int_id)}")
print(f"Total Genes: {len(gene_to_int_id)}")
print(f"Total Diseases: {len(disease_to_int_id)}")

# We will build a PyG HeteroData object
data = HeteroData()

# --- 1. Process Gene-Disease links ---
print("Processing Gene-Disease links...")
df_gene_disease = pd.read_csv(
    CTD_GENE_DISEASE_FILE, compression='gzip', header=None,
    names=gene_disease_cols, comment='#'
)
# Map string IDs to our new integer IDs
# Ensure GeneID is string type for mapping
source_nodes = df_gene_disease['GeneID'].astype(str).map(gene_to_int_id)
target_nodes = df_gene_disease['DiseaseID'].map(disease_to_int_id)
# Drop any rows that had IDs we aren't tracking (NaNs)
valid_links = source_nodes.notna() & target_nodes.notna()
gene_disease_edge_index = torch.tensor(
    [source_nodes[valid_links].values, target_nodes[valid_links].values],
    dtype=torch.long
)
data['gene', 'associates_with', 'disease'].edge_index = gene_disease_edge_index

# --- 2. Process Chem-Gene links ---
print("Processing Chem-Gene links...")
df_chem_gene = pd.read_csv(
    CTD_CHEM_GENE_FILE, compression='gzip', header=None,
    names=chem_gene_cols, comment='#'
)
source_nodes = df_chem_gene['ChemicalID'].map(chem_to_int_id)
target_nodes = df_chem_gene['GeneID'].astype(str).map(gene_to_int_id)
# Drop any rows that had IDs we aren't tracking (NaNs)
valid_links = source_nodes.notna() & target_nodes.notna()
chem_gene_edge_index = torch.tensor(
    [source_nodes[valid_links].values, target_nodes[valid_links].values],
    dtype=torch.long
)
data['chemical', 'interacts_with', 'gene'].edge_index = chem_gene_edge_index

# --- 3. Process Chem-Disease links (Our Target) ---
print("Processing Chem-Disease links...")
df_chem_disease = pd.read_csv(
    CTD_CHEM_DISEASE_FILE, compression='gzip', header=None,
    names=chem_disease_cols, comment='#'
)
# Keep only 'therapeutic' links for repositioning
df_chem_disease = df_chem_disease[df_chem_disease['DirectEvidence'] == 'therapeutic']
source_nodes = df_chem_disease['ChemicalID'].map(chem_to_int_id)
target_nodes = df_chem_disease['DiseaseID'].map(disease_to_int_id)
# Drop any rows that had IDs we aren't tracking (NaNs)
valid_links = source_nodes.notna() & target_nodes.notna()
chem_disease_edge_index = torch.tensor(
    [source_nodes[valid_links].values, target_nodes[valid_links].values],
    dtype=torch.long
)
data['chemical', 'treats', 'disease'].edge_index = chem_disease_edge_index

# ... Repeat this for ALL your files (BindingDB, pathways, side effects)
# Make sure to define their column names and use header=None

print("\nInitial Graph Structure:")
print(data)


# --- PASTE THIS CODE AT THE END OF YOUR try.py FILE ---
# --- PASTE THIS CODE AT THE END OF YOUR try.py FILE ---

print("\nLoading Hetionet embeddings...")
try:
    hetionet_model = KeyedVectors.load('hetionet_node2vec.kv')
    EMBEDDING_DIM = hetionet_model.vector_size
    print(f"Embedding dimension: {EMBEDDING_DIM}")
except Exception as e:
    print(f"Error loading 'hetionet_node2vec.kv': {e}")
    print("Please ensure the file is in the same directory as try.py")
    exit()

# --- 1. Create Chemical features ---
# Initialize with small random numbers
X_chem = np.random.rand(len(chem_to_int_id), EMBEDDING_DIM)
print(f"Initialized {len(chem_to_int_id)} chemical features with random values (no embeddings found).")
data['chemical'].x = torch.from_numpy(X_chem).float()

# --- 2. Create Gene features ---
# Initialize with small random numbers (for nodes NOT in Hetionet)
X_gene = np.random.rand(len(gene_to_int_id), EMBEDDING_DIM)
found_gene = 0
for ctd_id, int_id in gene_to_int_id.items():
    # The ctd_id is the key itself (e.g., "367")
    hetionet_key = ctd_id 
    
    if hetionet_key in hetionet_model:
        X_gene[int_id] = hetionet_model[hetionet_key]
        found_gene += 1

print(f"Populated {found_gene} / {len(gene_to_int_id)} gene features from Hetionet.")
data['gene'].x = torch.from_numpy(X_gene).float()

# --- 3. Create Disease features ---
# Initialize with small random numbers
X_disease = np.random.rand(len(disease_to_int_id), EMBEDDING_DIM)
print(f"Initialized {len(disease_to_int_id)} disease features with random values (no embeddings found).")
data['disease'].x = torch.from_numpy(X_disease).float()

# ... You would repeat this for other node types like Pathways ...
# (They will also be randomly initialized unless you have embeddings for them)

print("\nGraph with Node Features:")
print(data)

# --- STEP 5: Add this to the VERY END of try.py ---

print("\nCreating Train/Val/Test splits for link prediction...")

# We want to predict the 'treats' edge type
target_edge = ('chemical', 'treats', 'disease')

# This transform will:
# 1. Split the 'treats' links into train/val/test
# 2. Add *negative samples* (false links) for each split
# 3. Remove the val/test links from the main graph to prevent data leakage
try:
    transform = RandomLinkSplit(
        num_val=0.1,  # 10% for validation
        num_test=0.1, # 10% for testing
        is_undirected=False, 
        edge_types=[target_edge],
        # If you add a reverse edge, uncomment the line below
        # rev_edge_types=[('disease', 'treated_by', 'chemical')] 
    )

    # This is where the splitting happens
    train_data, val_data, test_data = transform(data)

    print("\n--- Training Data ---")
    print(train_data)
    print("\n--- Validation Data ---")
    print(val_data)

    # --- 6. Save your processed data ---
    print("\nSaving processed data as .pkl files...")
    
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        
    with open('val_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
        
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    print("Done! Your data is preprocessed and ready for an HNN model.")

except Exception as e:
    print(f"\nError during data splitting: {e}")
    print("This can happen if your 'target_edge' is not in the graph or has too few edges.")