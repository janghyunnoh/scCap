# scCap: Single-cell Knowledge-augmented Clustering for Annotation-free Phenotype Prediction

**scCap** is an annotation-free framework that constructs clusters through knowledge-augmented clustering and leverages them to enable accurate and interpretable phenotype prediction.

The pipeline consists of four main stages:
1. **Preparation** – Set up the environment, download pretrained single-cell foundation models, and prepare public datasets.  
2. **Preprocessing** – Perform standard preprocessing and encode cells using the pretrained single-cell foundation model. 
3. **Clustering** – Construct knowledge-guided clusters through a two-step process of initialization and refinement. 
4. **Prediction** – Train a hierarchical multiple instance learning (hier-mil) framework with dual-level attention to aggregate information across cell and cluster levels for phenotype prediction.

   
## 1. Preparation

### 1.1 Hardware Requirements

The scCap pipeline requires GPU acceleration for scGPT embedding generation and efficient execution on large-scale single-cell datasets.

| Dataset | Cells | Recommended GPU | Recommended RAM |
|:-------:|:-----:|:---------------:|:---------------:|
| COVID | 26,947 | <4 GB | <4 GB |
| Kidney | 225,177 | <16 GB | <32 GB |
| Cardio | 592,689 | <24 GB | <48 GB |

> **Note:** Recommendations are based on the peak resource usage observed during our experiments. Larger datasets may require additional GPU memory and system RAM.


### 1.2 Environment Setup

You can set up **scCap** using either (1) a prebuilt Docker image or (2) a Conda-based environment.  
Both methods provide an identical software environment for running all experiments.  
**We highly recommend using Docker** for easier setup and reproducibility across systems.   

#### Option 1. Docker Setup

For a ready-to-use environment, we provide a prebuilt Docker image in the [scCap Docker Hub repository](https://hub.docker.com/r/mjuailab/sccap).  
This image contains a fully configured environment identical to the Conda setup.  

```bash
# Step 1: Clone the repository
# ----------------------------
git clone https://github.com/janghyunnoh/scCap.git
cd scCap

# Step 2: Pull the prebuilt Docker image from Docker Hub
# ------------------------------------------------------
# You can use other tags (e.g., v1.2), but we recommend :latest as the stable default.

docker pull mjuailab/sccap:latest

# Step 3: Run the container with GPU support and sufficient shared memory
# -----------------------------------------------------------------------
# [shared_memory_size] defines the shared memory allocated to the container (e.g., 32g or 64g).
# Increase this value if you encounter a PyTorch “bus error” during embedding.
# [local_project_directory] should point to your local scCap project path (mounted to /workspace inside the container).

docker run -it --gpus all --shm-size=[shared_memory_size] \
  -v [local_project_directory]:/workspace \
  mjuailab/sccap:latest
```

#### Option 2. Conda Setup

```bash
git clone https://github.com/janghyunnoh/scCap.git  
cd scCap
conda env create -f environment.yml # Full research environment (may be heavy for basic usage)
conda activate scCap
```

> **Note:** For a lightweight setup, we also provide a minimal requirements.txt containing only the core dependencies.


### 1.3 Download Pretrained Models

We use the publicly available pretrained scGPT model, specifically the `whole-human` checkpoint, which was trained on 33 million normal human cells. 
The pretrained weights can be obtained from the official scGPT model zoo provided in the [official scGPT repository](https://github.com/bowang-lab/scGPT).
After downloading, place the required model files (`args.json`, `best_model.pt`, `vocab.json`) in the `./scGPT/model_human` directory.
The scGPT codebase and pretrained models are distributed under the MIT license.


### 1.4 Dataset

The following public single-cell RNA-seq datasets were used in our study. Download each dataset from the provided links and place the raw source files under `./data/raw` directory; the corresponding cleaned `.h5ad` files will be generated automatically during preprocessing stage.

- **COVID dataset**  
  [Impaired local intrinsic immunity to SARS-CoV-2 infection in severe COVID-19](https://singlecell.broadinstitute.org/single_cell/study/SCP1289/impaired-local-intrinsic-immunity-to-sars-cov-2-infection-in-severe-covid-19)

- **Cardiac dataset**  
  [Single-nuclei profiling of human dilated and hypertrophic cardiomyopathy](https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy)

- **Kidney dataset**  
  [Human kidney single-cell atlas (CellxGene)](https://cellxgene.cziscience.com/collections/0f528c8a-a25c-4840-8fa3-d156fa11086f)


### 1.5 Tutorial (Optional)

To help users quickly understand the full **scCap** pipeline, we provide both a lightweight tutorial dataset and ready-to-run execution options, including a script and a Google Colab notebook.
This optional tutorial reproduces the full workflow (**Preprocessing → Clustering → Prediction**) on a small demo dataset, allowing users to easily explore the pipeline without requiring a full environment setup.  
The tutorial dataset is a small **derived subset** of the COVID dataset, provided for demonstration purposes only, and does not correspond to the full dataset used in our experiments.

#### Option 1. Run locally (bash script)

To run the tutorial locally, you must first download the [tutorial dataset](https://drive.google.com/file/d/1R2vJoIDXRGx83yU-LpY4g5Vrmpdg-rJD/view?usp=drive_link) and place it in the `./tutorial/data`.  
Once the dataset is placed, you can execute the entire pipeline:

```bash
bash ./tutorial/run_tutorial.sh
```

> **Note:** You can modify parameters such as GPU_ID, directory paths, or the number of folds in run_tutorial.sh.

#### Option 2. Run on Google Colab

Alternatively, you can run the full pipeline interactively on [Google Colab](https://colab.research.google.com/drive/1Wlfi_z6OP0knLYgEDvhqpiW_zkUV3syA?usp=sharing).  
This option is recommended for quick testing and requires no local installation.

 
## 2. Preprocessing

The **preprocessing** stage prepares raw single-cell datasets for **Clustering** and **Prediction**.  
This stage converts raw count matrices into structured AnnData (.h5ad) format, applies quality control and normalization, 
and generates scGPT embeddings that capture biological knowledge learned from large-scale single-cell data.

### 2.1 Overview
This stage consists of the following steps:  
1. **Data Conversion** – Converts dataset-specific raw files (e.g., .mtx, .tsv, .txt) into a structured AnnData (.h5ad) format.
2. **Quality Control & Normalization** – Filter low-quality cells and normalize gene expression values to ensure consistency across samples.
3. **Knowledge Augmentation** - Encode each cell using the pretrained scGPT model to obtain biologically informed 512-dimensional embeddings that represent knowledge learned from millions of human cells.
4. **(Optional) Annotation Integration** - Integrate SingleR-based computational annotations.

Together, these steps transform raw single-cell data into biologically meaningful embeddings, providing a robust foundation for knowledge-augmented clustering and annotation-free phenotype prediction.

### 2.2 Generic Usage
```bash
# Step 1: Convert raw data to AnnData format (.h5ad)
# --------------------------------------------------
# Each dataset has its own converter script in ./data/raw/[dataset_name]/

python ./data/raw/[dataset_name]/[dataset_name].py


# Step 2: Run preprocessing and generate scGPT embeddings
# -------------------------------------------------------
# Performs filtering, normalization, scGPT embedding, and adds optional SingleR annotations.
# Replace [dataset_name] and file paths accordingly.
# The scGPT model defaults to "./scGPT/model_human"

python preprocess.py \
  --input ./data/[dataset_name]/[dataset_name].h5ad \
  --output ./data/[dataset_name]/[dataset_name]_preprocessed.h5ad \
  --min_cells [int] \
  --target_sum [float] \
  --batch_size [float]
```

### 2.3 Argument Details

| Argument | Type | Default | Required | Description |
| :-------- | :---- | :-------- | :---------- | :----------- |
| `--input` | `str` | – | Yes | Path to the input `.h5ad` file. |
| `--output` | `str` | – | Yes | Output path for the preprocessed `.h5ad` file containing embeddings. |
| `--singler` | `str` | – | No | Optional path to a SingleR annotation CSV file with Cell and SingleR_Label columns. If not provided, the step is skipped. |
| `--model` | `str` | `"./scGPT/model_human"` | No | Path to the pretrained scGPT model directory. Defaults to the official human model provided with scGPT. |
| `--min_cells` | `int` | `5` | No | Minimum number of cells in which a gene must be expressed to be retained. |
| `--target_sum` | `float` | `1e4` | No | Total expression value to which each cell is normalized. |
| `--batch_size` | `int` | `128` | No | Batch size for scGPT embedding inference. |

> **Note:** You can adjust these parameters according to your dataset or experimental goals. For example, modifying --min_cells or --target_sum can tune preprocessing sensitivity, while specifying a different pretrained model with --model allows for domain-specific embeddings (e.g., tissue-specific or disease-focused scGPT models).

### 2.4 Output

After completion, the script will generate:

- `[dataset_name]_preprocessed.h5ad` — AnnData file containing:
  - Filtered and normalized gene expression matrix  
  - 512-dimensional scGPT embeddings stored in `adata.obsm["X_scGPT"]`  
  - (Optional) SingleR annotations added to `adata.obs["singler_annotation"]`

This file serves as the input for the next **Clustering** stage, where biological knowledge are jointly augmented to construct knowledge-guided clusters.


## 3. Clustering

The **Clustering** stage constructs knowledge-guided clusters through **initialization**, **refinement**.
This process integrates local transcriptional variation from raw gene expression with biological knowledge encoded in the pretrained scGPT model, enabling knowledge-augmented clustering.
Users can flexibly specify the representation space (either raw or scgpt) for initialization and refinement with arguments.



### 3.1 Overview

This stage performs:  
1. **Initialization** –  Generates initial clusters using the specified representation space (raw or scgpt).
2. **Refinement** – Applies a split–merge strategy within the selected representation space to balance local compactness and global organization.



### 3.2 Generic Usage

```bash
# Replace [dataset_name] with a target dataset name (e.g., covid, cardio, kidney, or a custom dataset)  
# Hyperparameters can be adjusted by the user to suit the experimental setup and dataset characteristics

python clustering.py \
  --input ./data/[dataset_name]/[dataset_name]_preprocessed.h5ad \
  --output ./data/[dataset_name]/[dataset_name]_constructed.h5ad \
  --init-space [raw|scgpt] \
  --refine-space [raw|scgpt] \
  --n-hvg [int] \
  --ratio [float] \
  --resolution [float] \
  --threshold [float] \
  --constraint [float] \
  --n_neighbors [int] \
  --n_pcs [int] \
```

### 3.3 Argument Details

| Argument | Type | Default | Required | Description |
| :-------- | :---- | :-------- | :---------- | :----------- |
| `--input` | `str` | – | Yes | Path to the input `.h5ad` file generated after preprocessing. |
| `--output` | `str` | – | Yes | Path to save the output `.h5ad` file containing constructed clusters. |
| `--init-space` | `str` | `"raw"` | No | Representation space used for initial clustering. Choose between `raw` (gene expression) and `scgpt` (embedding space). |
| `--refine-space` | `str` | `"scgpt"` | No | Representation space for refinement steps. Choose between `raw` (gene expression) and `scgpt` (embedding space).|
| `--n-hvg` | `int` | `3000` | No | Number of highly variable genes (HVGs) to select. Set to `0` to skip HVG filtering. |
| `--ratio` | `float list` | `2.0` | No | Merge thresholds controlling how easily clusters are merged. Lower values make merging more aggressive (fewer, larger clusters), while higher values make merging more conservative (more, finer clusters). Multiple values allow evaluation across different merging scales. |
| `--resolution` | `float` | `1.0` | No | Leiden clustering resolution controlling the number of initial clusters. |
| `--threshold` | `float` | `0.5` | No | Split threshold determining when clusters should be subdivided, based on the ratio of intra- to inter-group distances. Lower values lead to finer splits, while higher values retain coarser cluster structures. |
| `--constraint` | `float` | `0.5` | No | Maximum allowed size of a merged cluster (as a fraction of total cells). Lower values restrict cluster growth; higher values allow larger merges.|
| `--n_neighbors` | `int` | `15` | No | Number of neighbors for graph construction during clustering. Larger values yield smoother cluster boundaries. |
| `--n_pcs` | `int` | `50` | No | Number of principal components used for dimensionality reduction (PCA). |

> **Note:** Adjust parameters according to dataset size and analysis objectives. For instance, tuning parameters such as --resolution and --threshold allows users to control the clustering granularity.

### 3.4 Output

After completion, the script generates:
- `[dataset_name]_constructed.h5ad` — AnnData file containing the final refined clusters stored in adata.obs["refined_cluster"].  
- Intermediate results (init_cluster, split_*, merged_*) are also retained within the same AnnData object for reference and reproducibility.


## 4. Prediction

The **Prediction** stage trains a hier-mil framework for phenotype prediction using the clusters constructed in the previous stage. This implementation is adapted from the [hier-mil repository](https://github.com/minhchaudo/hier-mil), and we thank **Chau Do** and **Harri Lähdesmäki** for making their code publicly available.


### 4.1 Overview

This stage performs:
1. **Data loading** – Loads the constructed `.h5ad` file generated from the clustering stage.
2. **Model training** – Trains the hier-mil framework with cross-validation and hyperparameter optimization using Optuna.
3. **Phenotype prediction** – Generates patient-level predictions and reports AUROC scores for each cross-validation fold.

### 4.2 Generic Usage

```bash
# Replace [gpu_number] with the GPU ID you want to use (e.g., 0, 1)
# Replace [dataset_name] with one of: covid, cardio, kidney or your data

CUDA_VISIBLE_DEVICES=[gpu_number] python ./hier-mil/run.py \
  --data_path ./data/[dataset_name]/[dataset_name]_constructed.h5ad \
  --task 2 \
  --patient_id_key patient \
  --label_key label \
  --cell_type_annot_key refined_cluster \
  --attn1 1 \
  --device cuda \
  --n_tune_trials 30 \
  --n_folds_hyperparam_tune 5 \
  --n_folds 5 \
  --n_repeats 5 \
  --n_epochs 100 \
  --output ./result/[dataset_name]/[dataset_name]_result.txt
```

> **Note:** Training configurations can be modified depending on dataset size and computational resources. For example, increasing `--n_tune_trials` improves hyperparameter optimization, while adjusting `--n_folds` or `--n_repeats` balances evaluation stability and runtime. For full argument details and model architecture explanations, please refer to the [hier-mil repository](https://github.com/minhchaudo/hier-mil).


### 4.3 Output

After training completes, the script will produce:

- `[dataset_name]_result.txt` — summary file saved in `./result/[dataset_name]/`,  
  containing cross-validation results (e.g., seed and AUROC score for each fold).


## Contact

For any questions or feedback, please contact:  
**Janghyun Noh** – [jacknoh9902@gmail.com](mailto:jacknoh9902@gmail.com)


## Citation

If you find **scCap** useful in your research, please cite the following work:

```bibtex
@article{
}
```
