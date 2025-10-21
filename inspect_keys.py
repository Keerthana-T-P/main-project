import sys
from gensim.models import KeyedVectors

try:
    # Load the model
    model = KeyedVectors.load('hetionet_node2vec.kv')
    
    # Get the list of all keys (node names)
    all_keys = list(model.index_to_key)
    
    print(f"Successfully loaded model.")
    print(f"Total nodes in .kv file: {len(all_keys)}")
    
    # --- Print the first 10 keys ---
    print("\n--- First 10 Keys in 'hetionet_node2vec.kv' ---")
    for key in all_keys[:10]:
        print(key)
        
    print("--------------------------------------------------")

except Exception as e:
    print(f"Error: {e}")
    print("Could not load 'hetionet_node2vec.kv'. Make sure gensim is installed.")