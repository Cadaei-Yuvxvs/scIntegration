# Run python sca.models.scVI
# runs different parameters on one assay+featureset

# Import relevant modules
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import igraph
import os
import sys
import json
import scvi
import gc
import torch
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import scipy.sparse as sp

print(scvi.__version__)

torch.set_float32_matmul_precision('medium')
sc.settings.figdir = "/scratch/user/username/transcriptomics/plots/"

print("Read parameters")
# get path to parameters and path json
#json_file = open(sys.argv[1])
json_file = open('/scratch/user/username/transcriptomics/slurm/integration_scvi_params_scarches.json')
# read json into dictionary
json_str = json_file.read()
parameter_dict = json.loads(json_str)

# general params from dict (could also save these lines, but I think this way it is easier to digest)
job_id=parameter_dict["job_id"]
batch_var=parameter_dict["batch_var"]
data_filepath_full = parameter_dict["data_filepath_full"]
feature_set_file = parameter_dict["feature_set_file"]
results_path = parameter_dict["output_folder"]
global_seed = parameter_dict["global_seed"]
hvgs_set_name = parameter_dict["hvgs_set_name"]
param_path = parameter_dict["param_path"]
use_cuda = parameter_dict["use_cuda"]
categorical_covariates = parameter_dict["categorical_covariates"]
continuous_covariates = parameter_dict["continuous_covariates"]

# Check if results_path exists
os.makedirs(results_path, exist_ok=True)

scvi.settings.seed = global_seed

# read parameters for scVI
param_df = pd.read_csv(param_path,sep=",")

# read RNA assay (RNA assay because scVI needs raw data)
print("Read anndata")
adata = sc.read_h5ad(data_filepath_full)
#ensure there are no bytestrings 
str_df = adata.obs
str_df = str_df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
str_df = str_df.set_index('Cell_ID',drop=False)
adata.obs = str_df
# for features:
str_df = adata.var
str_df = str_df.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
#str_df = str_df.set_index('features',drop=False)
adata.var = str_df

# read json into dictionary
json_file = open(feature_set_file)
hvg_dict = json.load(json_file)
hvgs = hvg_dict[hvgs_set_name] # add in hvgs variable

adata.obs['Dataset'] = adata.obs['Dataset'].replace('NA', 'FTD_ALS')

# Create query mask
query_mask = np.array([s == "FTD_ALS" for s in adata.obs["Dataset"]])

# Create reference
adata = adata[:, hvgs].copy()
#adata_scvi = adata.copy()
adata_scvi = adata[~query_mask].copy()

# Create query
target_adata = adata[query_mask].copy()

# clean up
gc.collect()

print("=== Data Validation ===")

try:
    # Check for problematic values in raw data
    if sp.issparse(adata_scvi.X):
        data_array = adata_scvi.X.data
    else:
        data_array = adata_scvi.X.flatten()

    print(f"Data contains NaN: {np.isnan(data_array).any()}")
    print(f"Data contains Inf: {np.isinf(data_array).any()}")
    
    if np.isnan(data_array).any() or np.isinf(data_array).any():
        print("ERROR: Data contains NaN or Inf values! Cleaning...")
        # Clean the data
        if sp.issparse(adata_scvi.X):
            adata_scvi.X.data = np.nan_to_num(adata_scvi.X.data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            adata_scvi.X = np.nan_to_num(adata_scvi.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Data min/max: {data_array.min():.3f} / {data_array.max():.3f}")
    print(f"Data mean/std: {data_array.mean():.3f} / {data_array.std():.3f}")

    # Check for extremely large values that could cause overflow
    extreme_threshold = 1e6
    extreme_values = (np.abs(data_array) > extreme_threshold).sum()
    print(f"Values > {extreme_threshold}: {extreme_values}")
    
except Exception as e:
    print(f"Data validation failed: {e}")
    print("Proceeding with caution...")

## Run scVI
# https://www.scvi-tools.org/en/stable/user_guide/notebooks/harmonization.html
print("Preparing scVI")
# if only 1 variable --> make list
if not isinstance(categorical_covariates, list):
    categorical_covariates_save=categorical_covariates
    categorical_covariates = list()
    categorical_covariates.insert(0,categorical_covariates_save)

if not isinstance(continuous_covariates, list):
    continuous_covariates_save=continuous_covariates
    continuous_covariates = list()
    continuous_covariates.insert(0,continuous_covariates_save)

length_cov = len(categorical_covariates)+len(continuous_covariates)

if len(categorical_covariates) == 0:
    categorical_covariates = None

if len(continuous_covariates) == 0:
    continuous_covariates = None

print("categorical_covariate_keys: "+str(categorical_covariates))
print("continuous_covariate_keys: "+str(continuous_covariates))

# setup for scvi

scvi.model.SCVI.setup_anndata(adata_scvi, categorical_covariate_keys=categorical_covariates,continuous_covariate_keys=continuous_covariates)

# Create dictionary from parameter dataframe
keys = param_df.columns.tolist()

# The second row will be your values
values = param_df.iloc[0].tolist()

# Create the dictionary using zip
row = dict(zip(keys, values))

index=0

print("Running "+str(index+1)+" of "+str(len(param_df.index))+" scVI runs")
# set up VAE
vae = scvi.model.SCVI(adata_scvi,
    n_layers=int(row['n_layers']),
    n_latent=int(row['n_latent']),
    n_hidden=int(row['n_hidden']),
    dropout_rate=float(row['dropout_rate']),
    dispersion = str(row['dispersion']),
    gene_likelihood = str(row['gene_likelihood'])
)

# train
vae.train(
    max_epochs=int(row['max_epochs']), 
    early_stopping=bool(row['early_stopping']),
    accelerator="gpu",
    plan_kwargs={
        "lr": 1e-4,
        "reduce_lr_on_plateau": True,
        "lr_patience": 15,
        "lr_factor": 0.6
    }
)

# After scVI training, before scANVI
print("=== Checking scVI model health ===")
vae.save(results_path, overwrite=True)

# Test the model
test_latent = vae.get_latent_representation(adata_scvi[:100])
print(f"scVI latent has NaN: {np.isnan(test_latent).any()}")
print(f"scVI latent std: {test_latent.std():.3f}")

# If std is extremely high (>10) or has NaN, the model is unstable
if np.isnan(test_latent).any() or test_latent.std() > 10:
    print("WARNING: scVI model is numerically unstable!")
    print("Recommendation: Retrain with early stopping and fewer epochs")
else:
    print("scVI model looks stable, proceeding to scANVI")

# get result
adata_scvi.obsm["X_scVI"] = vae.get_latent_representation()
output = pd.DataFrame(adata_scvi.obsm["X_scVI"])
output = output.set_index(adata_scvi.obs_names)
output2 = output.set_axis(["scVI_" + str(s) for s in output.axes[1].to_list()], axis=1)
# save
output2.to_csv(results_path+"scVI-predict_"+str(index)+"_"+str(int(row['max_epochs']))+"_"+
                str(float(row['dropout_rate']))+"_"+str(int(row['n_layers']))+"_"+
                str(int(row['n_hidden']))+"_"+str(row['dispersion'])+"_"+str(row['gene_likelihood'])+"_cov"+str(length_cov)+
                "..scVI.."+str(int(row['n_latent']))+".."+hvgs_set_name+"_"+job_id+".txt", sep='\t',index=True)

# Visualise
sc.pp.neighbors(adata_scvi, use_rep='X_scVI')
sc.tl.leiden(adata_scvi)
sc.tl.umap(adata_scvi)


sc.pl.umap(
adata_scvi,
color=["Dataset", "C1_named"],
frameon=False,
ncols=1,
save="scvi_model.png"
)


# SCANVI model training from SCVI model

#vae = scvi.model.SCVI.load(os.path.join(results_path, "model.pt"))

# Validate the loaded scVI model works correctly
print("=== Validating loaded scVI model ===")
try:
    test_latent = vae.get_latent_representation(adata_scvi[:100])  # Test on small subset
    print(f"scVI latent representation shape: {test_latent.shape}")
    print(f"scVI latent has NaN: {np.isnan(test_latent).any()}")
    print(f"scVI latent range: [{test_latent.min():.3f}, {test_latent.max():.3f}]")
except Exception as e:
    print(f"ERROR: scVI model validation failed: {e}")
    # Model is corrupted, need to retrain

SCANVI_CELLTYPE_KEY = "celltype_scanvi"


adata_scvi.obs[SCANVI_CELLTYPE_KEY] = adata_scvi.obs["C1_named"]
target_adata.obs[SCANVI_CELLTYPE_KEY] = target_adata.obs["C1_named"]

adata_scvi.obs[SCANVI_CELLTYPE_KEY] = adata_scvi.obs[SCANVI_CELLTYPE_KEY].replace("NA", "Unknown")
target_adata.obs[SCANVI_CELLTYPE_KEY] = target_adata.obs[SCANVI_CELLTYPE_KEY].replace("NA", "Unknown")

# Add validation after string replacement
print("=== Post-replacement cell type validation ===")
print("Reference cell types after replacement:")
print(adata_scvi.obs[SCANVI_CELLTYPE_KEY].value_counts())
print("Query cell types after replacement:")  
print(target_adata.obs[SCANVI_CELLTYPE_KEY].value_counts())

# Check for cell types in query that don't exist in reference
ref_celltypes_final = set(adata_scvi.obs[SCANVI_CELLTYPE_KEY].unique())
query_celltypes_final = set(target_adata.obs[SCANVI_CELLTYPE_KEY].unique())
novel_celltypes = query_celltypes_final - ref_celltypes_final

if novel_celltypes:
    print(f"WARNING: Query has cell types not in reference: {novel_celltypes}")
    print("These will be treated as 'Unknown' during prediction")

print("Cell type distribution in reference:")
print(adata_scvi.obs[SCANVI_CELLTYPE_KEY].value_counts())

np.unique(adata_scvi.obs[SCANVI_CELLTYPE_KEY], return_counts=True)

scanvi_model = scvi.model.SCANVI.from_scvi_model(
    vae,
    unlabeled_category="Unknown",
    labels_key=SCANVI_CELLTYPE_KEY,
)

scanvi_model.train(
    max_epochs=50, 
    n_samples_per_label=100, 
    accelerator="gpu",
    check_val_every_n_epoch=10,
    train_size=0.8,
    plan_kwargs={
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "reduce_lr_on_plateau": True,
        "lr_patience": 5,
        "lr_factor": 0.8
    }
)

SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTION_KEY = "C_scANVI"
SCANVI_PROB_KEY = "C_scANVI_proba"

adata_scvi.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata_scvi)

output = pd.DataFrame(adata_scvi.obsm[SCANVI_LATENT_KEY])
output = output.set_index(adata_scvi.obs_names)
output2 = output.set_axis(["scANVI_" + str(s) for s in output.axes[1].to_list()], axis=1)
# save
output2.to_csv(results_path+"scANVI-latent_"+str(index)+"_"+str(int(row['max_epochs']))+"_"+
                str(float(row['dropout_rate']))+"_"+str(int(row['n_layers']))+"_"+
                str(int(row['n_hidden']))+"_"+str(row['dispersion'])+"_"+str(row['gene_likelihood'])+"_cov"+str(length_cov)+
                "..scVI.."+str(int(row['n_latent']))+".."+hvgs_set_name+"_"+job_id+".txt", sep='\t',index=True)

scanvi_model.save(os.path.join(results_path, "scanvi_c1_model"), overwrite=True)

# Visualise

sc.pp.neighbors(adata_scvi, use_rep=SCANVI_LATENT_KEY)
sc.tl.leiden(adata_scvi)
sc.tl.umap(adata_scvi, min_dist=0.3)

sc.pl.umap(
    adata_scvi,
    color=["Dataset", "C1_named"],
    frameon=False,
    ncols=1,
    save="scanvi_model_latent.png"
)


# Update with Query

# again a no-op in this tutorial, but good practice to use
scvi.model.SCANVI.prepare_query_anndata(target_adata, scanvi_model)
scanvi_query = scvi.model.SCANVI.load_query_data(target_adata, scanvi_model)
scanvi_query.train(
    max_epochs=20,
    plan_kwargs={"weight_decay": 0.0},
    check_val_every_n_epoch=10,
    accelerator="gpu"
)


target_adata.obsm[SCANVI_LATENT_KEY] = scanvi_query.get_latent_representation()
target_adata.obs[SCANVI_PREDICTION_KEY] = scanvi_query.predict()
cell_type_probabilities_df = scanvi_query.predict(target_adata, soft=True)
target_adata.obsm[SCANVI_PROB_KEY] = cell_type_probabilities_df

sc.pp.neighbors(target_adata, use_rep=SCANVI_LATENT_KEY)
sc.tl.leiden(target_adata)
sc.tl.umap(target_adata)

sc.pl.umap(
target_adata,
color=[SCANVI_PREDICTION_KEY],
frameon=False,
ncols=1,
save="scanvi-predict.png"
)

output = pd.DataFrame(target_adata.obsm[SCANVI_LATENT_KEY])
output = output.set_index(target_adata.obs_names)
output2 = output.set_axis(["scANVI_" + str(s) for s in output.axes[1].to_list()], axis=1)
# save
output2.to_csv(results_path+"scANVI-predict_"+str(index)+"_"+str(int(row['max_epochs']))+"_"+
                str(float(row['dropout_rate']))+"_"+str(int(row['n_layers']))+"_"+
                str(int(row['n_hidden']))+"_"+str(row['dispersion'])+"_"+str(row['gene_likelihood'])+"_cov"+str(length_cov)+
                "..scVI.."+str(int(row['n_latent']))+".."+hvgs_set_name+"_"+job_id+".txt", sep='\t',index=True)


# Save Hard Predictions
output_predicted_cells = pd.DataFrame(target_adata.obs[SCANVI_PREDICTION_KEY]) # Ensure it's a DataFrame
try:
    output_predicted_cells = output_predicted_cells.set_index(target_adata.obs_names)
except Exception as e:
    print(f"WARNING: output_predicted_cells.set_index didn't work. Error: {e}")
    output_predicted_cells.index = target_adata.obs_names

output_predicted_cells.to_csv(results_path+"scANVI-predict-cells_"+str(index)+"_"+str(int(row['max_epochs']))+"_"+
                               str(float(row['dropout_rate']))+"_"+str(int(row['n_layers']))+"_"+
                               str(int(row['n_hidden']))+"_"+str(row['dispersion'])+"_"+str(row['gene_likelihood'])+"_cov"+str(length_cov)+
                               "..scVI.."+str(int(row['n_latent']))+".."+hvgs_set_name+"_"+job_id+".txt", sep='\t',index=True)

# Save Prediction Probabilities
# cell_type_probabilities_df already has cells as index and categories as columns from predict(soft=True)
cell_type_probabilities_df.to_csv(results_path+"scANVI-predict-probabilities_"+str(index)+"_"+str(int(row['max_epochs']))+"_"+
                                  str(float(row['dropout_rate']))+"_"+str(int(row['n_layers']))+"_"+
                                  str(int(row['n_hidden']))+"_"+str(row['dispersion'])+"_"+str(row['gene_likelihood'])+"_cov"+str(length_cov)+
                                  "..scVI.."+str(int(row['n_latent']))+".."+hvgs_set_name+"_"+job_id+".txt", sep='\t',index=True)
