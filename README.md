# cpiVAE: Cross-Platform Proteomics Data Imputation

## Overview

The cpiVAE learns a shared latent representation across different proteomics platforms. The model consists of platform-specific encoders and decoders connected through a shared latent space, allowing for bidirectional translation between platforms while preserving biological signals.

### Key Features

- **Cross-platform imputation**: Translate measurements between Olink, SomaScan, and other proteomics platforms
- **Multiple model architectures**: Support for VAE, VampPrior, Vector Quantized VAE, and any custom architecture.
- **Comprehensive evaluation**: Built-in KNN and WNN baselines for comparision.
- **Feature importance analysis**: Integration with Captum for interpretability
- **Hyperparameter optimization**: Bayesian optimization with Optuna


## Full Documentation

For detailed documentation of all scripts and functions, please check out [https://joelbaderlab.github.io/cpiVAE_v1/](https://joelbaderlab.github.io/cpiVAE_v1/).



## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Training Workflow

0. **Acquire data**: For a example run, you may download the supplementory table in [Wang, Baihan, et al.](https://www.nature.com/articles/s41467-025-56935-2#data-availability) from [this figshare](https://figshare.com/articles/dataset/CKB_Olink_SomaScan_overlapping_proteins/27931350).

1. **Prepare your data**: CSV files with matching sample IDs between platforms
```bash
python scripts/split_data.py --platform_a data/olink_overlap.csv --platform_b data/somascan_overlap.csv --output_dir data/
```

2. **Hyperparameter tuning** (optional but highly recommended for your own dataset):
```bash
python scripts/tune.py --config configs/default.yaml \
    --platform_a data/olink_tune.csv \
    --platform_b data/somascan_tune.csv
```

3. **Train the model**:
```bash
python scripts/train.py --config configs/default.yaml \
    --platform_a data/olink_overlap_train.csv \
    --platform_b data/somascan_overlap_train.csv \
    --output_dir outputs_vae
```
For imputation only, you may choose to change to use the build-in log preoprocessing in the default config file for better results. Doing so may produce some skewed figures in the visulization script, as it expect manual log-preprocessing.

4. **Perform cross-platform imputation**:
```bash
python scripts/impute.py --input_data data/olink_overlap_test.csv \
    --source_platform a --target_platform b \
    --experiment_dir outputs_vae/joint_vae_experiment/version_X \
    --output data/somascan_overlap_imputed.csv
```

### Quick Imputation with Pre-trained Weights

For immediate testing, you can use our pre-trained model to perform cross-platform imputation:
- Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/1-5rpWkobsJ6LiYyXDjU580DqGFmpLWfe?usp=sharing).
- Determine the direction imputation. In our case, platform a is Olink and platform b is Somascan

```bash
python scripts/impute.py --input_data your_data.csv \
    --source_platform a --target_platform b \
    --experiment_dir public_weight/version_20250715-124150/ \
    --output imputed_output.csv
```

**Important**: Input data must be preprocessed following the same pipelines as that this weight is trained on:
- **SomaScan**: Median normalized format
- **Olink**: Standard QC pipeline
- **Both platforms**: Log2 transformed and z-standardized feature-wise
- **Features**: Must match those in `feature_names.pkl` (no missing features allowed)
- Alternatively, you may subset and retrained the model from the provided example dataset to fit your spesific sets of features.

For optimal results, we recommend training and tuning your own model. See [full imputation documentation](https://joelbaderlab.github.io/cpiVAE_v1/impute) for details.

### Complete Analysis Pipeline

Recreate manuscript figures with the full analysis pipeline including baseline comparisons:

```bash
bash scripts/manuscript_recreate.sh
```

This will:
- Train the jointVAE model
- Run KNN and WNN baselines
- Perform cross-platform imputation
- Generate comparison reports
- Generate latent space and feature importance reports


## Data Format

Input CSV files should have (the following works with --transpose argument. See manuscript_recreate.sh for examples usage):
- **First column**: Sample IDs (must have overlaps between platforms)
- **Remaining columns**: Feature measurements (proteins/analytes)
- **Missing values**: Allowed and handled during preprocessing

Example:
```
SampleID,Protein1,Protein2,Protein3,...
Sample001,1.23,2.45,0.89,...
Sample002,1.67,2.91,1.12,...
```

## Training Configuration

Models are configured via YAML files in `configs/`:
- `default.yaml`: Standard jointVAE configuration

Key parameters:
- `model_type`: Architecture
- `latent_dim`: Latent space dimension
- `learning_rate`, `batch_size`, `max_epochs`: Training parameters
- `loss_weights`: Balance between loss components

They can be set autoamticlly by runing the tuning script. 


## Evaluation Framework

The framework provides evaluation across multiple dimensions:
- **Imputation accuracy**: Feature-wise and sample-wise correlations, R², RMSE, per-protein performance metrics
- **Latent space quality**: Platform alignment, dimensionality reduction (PCA/UMAP/t-SNE)
- **Feature importance**: Network topology analysis, PPI reference comparison, cross-feature dependency patterns
- **Biological validation**: Phenotype association preservation, cis-pQTL discovery concordance, genetic signal retention

## Additional Functions

### Feature Importance Analysis
```bash
python scripts/feature_importance_analysis.py \
    --importance_a_to_b importance_scores.csv \
    --network_type directed \
    --threshold_method absolute_importance
```

### Latent Space Analysis
```bash
python scripts/latent_space_analysis.py \
    --latent_a latent_representations_a.csv \
    --latent_b latent_representations_b.csv
```

### Quality Control
```bash
python scripts/qc.py --data_file data/platform_data.csv
python scripts/qc_feature.py --data_file data/platform_data.csv
```

## Citation

If you use this software in your research, please cite:

```bibtex
[To be added]
```

## Acknowledgement

A portion of the code in this repository was generated by Claude 4, a generative model from Anthropic, and subsequently reviewed and edited by human developers.
