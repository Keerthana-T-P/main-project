import pandas as pd

def read_gz(path, sep="\t", usecols=None):
    return pd.read_csv(path, sep=sep, compression='gzip', usecols=usecols, comment="#", dtype=str)

# Load CTD relations
drug_gene = read_gz("CTD_chem_gene_ixns.csv.gz", ",", ["ChemicalName", "GeneSymbol"])
drug_disease = read_gz("CTD_curated_chemicals_diseases.csv.gz", ",", ["ChemicalName", "DiseaseName"])
gene_disease = read_gz("CTD_curated_genes_diseases.csv.gz", ",", ["GeneSymbol", "DiseaseName"])
gene_pathway = read_gz("CTD_genes_pathways.csv.gz", ",", ["GeneSymbol", "PathwayName"])
disease_pathway = read_gz("CTD_diseases_pathways.csv.gz", ",", ["DiseaseName", "PathwayName"])

# Add relation labels
drug_gene["relation"] = "drug-gene"
drug_disease["relation"] = "drug-disease"
gene_disease["relation"] = "gene-disease"
gene_pathway["relation"] = "gene-pathway"
disease_pathway["relation"] = "disease-pathway"
