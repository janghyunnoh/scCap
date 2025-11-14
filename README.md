# scCap: Single-cell Knowledge-augmented Clustering for Annotation-free Phenotype Prediction

**scCap** is an annotation-free framework that constructs biologically meaningful clusters through knowledge-augmented clustering and leverages them to enable accurate and interpretable phenotype prediction.

The pipeline consists of four main stages:
1. **Preparation** – Set up the environment, download pretrained single-cell foundation models, and prepare public datasets.  
2. **Preprocessing** – Perform standard preprocessing and encode cells using the pretrained single-cell foundation model. 
3. **Clustering** – Construct biologically meaningful clusters through a three-step process of initialization, refinement, and selection. 
4. **Prediction** – Train a hierarchical multiple instance learning (hier-mil) framework with dual-level attention to aggregate information across cell and cluster levels for phenotype prediction.

   
## 1. Preparation
You can set up **scCap** using either (1) a Conda-based environment or (2) a prebuilt Docker image.  
Both methods provide an identical software environment for running all experiments.  
**We highly recommend using Docker** for easier setup and reproducibility across systems.  

### 1.1 Conda Setup
```bash
git clone https://github.com/janghyunnoh/scCap.git  
cd scCap
conda env create -f environment.yml
conda activate scCap
```

### 1.2 Docker Setup
For a ready-to-use environment, download the prebuilt [scCap Docker image](https://drive.google.com/drive/folders/10wwlLMg0m2H0RiBAizxwqY2hC47tY5dn?usp=drive_link).  
This image provides a fully configured environment identical to the Conda setup.  

```bash
# Step 1: Download the prebuilt Docker image
# ------------------------------------------
# Download 'sccap_v1.2.tar.gz' from the link above.


# Step 2: Load the image into Docker
# ----------------------------------
# Import the downloaded image so it can be used to create containers.
# Make sure the file path matches your local download directory.

docker load -i [path_to_downloaded_image]/sccap_v1.2.tar.gz


# Step 3: Run the container with GPU support and sufficient shared memory
# -----------------------------------------------------------------------
# [shared_memory_size] defines the shared memory allocated to the container (e.g., 32g or 64g).
# Increase this value if you encounter a PyTorch “bus error” during embedding.
# [local_project_directory] should point to your local scCap project path (mounted to /workspace inside the container).

docker run -it --gpus all --shm-size=[shared_memory_size] \
  -v [local_project_directory]:/workspace \
  sccap:v1.2
```

### 1.3 Download Pretrained Models
Download the [whole-human pretrained model](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y) and place it inside the `./scGPT/model_human` directory.  
For more details about pretrained weights and model usage, refer to the [official scGPT repository](https://github.com/bowang-lab/scGPT).

### 1.4. Dataset

The following public single-cell RNA-seq datasets were used in our study.
Download each dataset from the provided links and place the raw source files under `./data/raw` directory.  
Cleaned .h5ad files will be generated automatically during preprocessing.  

- **COVID dataset**  
  [Impaired local intrinsic immunity to SARS-CoV-2 infection in severe COVID-19](https://singlecell.broadinstitute.org/single_cell/study/SCP1289/impaired-local-intrinsic-immunity-to-sars-cov-2-infection-in-severe-covid-19)

- **Cardiac dataset**  
  [Single-nuclei profiling of human dilated and hypertrophic cardiomyopathy](https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy)

- **Kidney dataset**  
  [Human kidney single-cell atlas (CellxGene)](https://cellxgene.cziscience.com/collections/0f528c8a-a25c-4840-8fa3-d156fa11086f)


### 1.5. Tutorial (Optional)

To help users quickly understand the full **scCap** pipeline, we provide a lightweight tutorial dataset and a ready-to-run bash script.
This optional tutorial reproduces the full workflow — **Preprocessing** → **Clustering** → **Prediction** — on a small demo dataset. 

Download the [tutorial dataset](https://drive.google.com/drive/folders/1iOvqZRoR9JT3GLmGMcsBxiKx313M-BGy?usp=drive_link) and place it inside the `./tutorial/data`.  
> The dataset is a compact version of the COVID dataset, designed to demonstrate the entire scCap pipeline in a simplified setting.

Once the dataset is placed, you can either start from **Preprocessing** and run each stage step by step,  
or execute the entire pipeline in one go using the provided bash script:

```bash
bash ./tutorial/run_tutorial.sh
```
> **Tip:**  
> You can edit parameters such as GPU_ID, directory paths, or number of folds inside the run_tutorial.sh file.  


 
## 2. Preprocessing

The **preprocessing** stage prepares raw single-cell datasets for downstream **Clustering** and **Prediction**.  
To enable biologically meaningful clustering, this stage converts raw count matrices into structured AnnData (.h5ad) format,
applies quality control and normalization, and generates scGPT embeddings that capture biological knowledge learned from large-scale single-cell data.

### 2.1 Overview
This stage consists of the following steps:  
1. **Data Conversion** – Converts dataset-specific raw files (e.g., .mtx, .tsv, .txt) into a structured AnnData (.h5ad) format.
2. **Quality Control & Normalization** – Filter low-quality cells and normalize gene expression values to ensure consistency across samples.
3. **Knowledge Augmentation** - Encode each cell using the pretrained scGPT model to obtain biologically informed 512-dimensional embeddings that represent knowledge learned from millions of human cells.
4. **(Optional) Annotation Integration** - Integrate SingleR-based computational annotations for reference-based comparison.

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
  --output ./data/[dataset_name]/[dataset_name]_preprocessed.h5ad
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

> **Note:**  
> You can adjust these parameters according to your dataset or experimental goals.  
> For example, modifying --min_cells or --target_sum can tune preprocessing sensitivity,  
> while specifying a different pretrained model with --model allows for domain-specific embeddings
> (e.g., tissue-specific or disease-focused scGPT models).

### 2.4 Output

After completion, the script will generate:

- `[dataset_name]_preprocessed.h5ad` — AnnData file containing:
  - Filtered and normalized gene expression matrix  
  - 512-dimensional scGPT embeddings stored in `adata.obsm["X_scGPT"]`  
  - (Optional) SingleR annotations added to `adata.obs["singler_annotation"]`

This file serves as the input for the next **Clustering** stage, where biological knowledge are jointly augmented to construct biologically meaningful clusters.


## 3. Clustering

The **Clustering** stage constructs biologically meaningful clusters through **initialization**, **refinement**, and **selection**.
This process integrates local transcriptional variation from raw gene expression with biological knowledge encoded in the pretrained scGPT model, enabling knowledge-augmented clustering.
Users can flexibly specify the representation space (either raw or scgpt) for initialization and refinement with arguments.



### 3.1 Overview

This stage performs:  
1. **Initialization** –  Generates initial clusters using the specified representation space (raw or scgpt).
2. **Refinement** – Applies a split–merge strategy within the selected representation space to balance local compactness and global biological organization.
3. **Selection** – Evaluates candidate cluster sets using clustering quality metrics to identify the optimal cluster configuration.


### 3.2 Generic Usage

```bash
# Replace [dataset_name] with one of: covid, cardio, kidney or your data

python clustering.py \
  --input ./data/[dataset_name]/[dataset_name]_preprocessed.h5ad \
  --output ./data/[dataset_name]/[dataset_name]_constructed.h5ad \
  --init-space [raw|scgpt] \
  --refine-space [raw|scgpt] \
  --n-hvg [int] \
  --ratios [float list] \
  --threshold [float] \
  --resolution [float] \
  --n_neighbors [int] \
  --n_pcs [int] \
  --ㅡmax-cells [int]
```

### 3.3 Argument Details

| Argument | Type | Default | Required | Description |
| :-------- | :---- | :-------- | :---------- | :----------- |
| `--input` | `str` | – | Yes | Path to the input `.h5ad` file generated after preprocessing. |
| `--output` | `str` | – | Yes | Path to save the output `.h5ad` file containing constructed clusters. |
| `--init-space` | `str` | `"raw"` | No | Representation space used for initial clustering. Choose between `raw` (gene expression) and `scgpt` (embedding space). |
| `--refine-space` | `str` | `"scgpt"` | No | Representation space for refinement steps. Choose between `raw` (gene expression) and `scgpt` (embedding space).|
| `--n-hvg` | `int` | `3000` | No | Number of highly variable genes (HVGs) to select. Set to `0` to skip HVG filtering. |
| `--ratios` | `float list` | `[2.0, 2.1, 2.2, 2.3, 2.4, 2.5]` | No | Merge thresholds controlling how easily clusters are merged. Lower values make merging more aggressive (fewer, larger clusters), while higher values make merging more conservative (more, finer clusters). Multiple values allow evaluation across different merging scales. |
| `--threshold` | `float` | `0.5` | No | Split threshold determining when clusters should be subdivided, based on the ratio of intra- to inter-group distances. Lower values lead to finer splits, while higher values retain coarser cluster structures. |
| `--resolution` | `float` | `1.0` | No | Leiden clustering resolution controlling the number of initial clusters. |
| `--n_neighbors` | `int` | `15` | No | Number of neighbors for graph construction during clustering. Larger values yield smoother cluster boundaries. |
| `--n_pcs` | `int` | `50` | No | Number of principal components used for dimensionality reduction (PCA). |
| `--max-cells` | `int` | `None` | No | Limits the number of cells used in selection for faster evaluation. If not set, all cells are used. |

> **Note:**  
> Adjust parameters according to dataset size and analysis objectives.
> For instance, tuning --n-hvg and --threshold changes clustering granularity,
> while --max-cells offers a trade-off between computational speed and evaluation stability.

### 3.4 Output

After completion, the script generates:
- `[dataset_name]_constructed.h5ad` — AnnData file containing the final optimal clusters stored in adata.obs["optimal_cluster"].  
- Intermediate results (init_cluster, split_*, merged_*) are also retained within the same AnnData object for reference and reproducibility.


## 4. Prediction

The **Prediction** stage trains a hier-mil framework for phenotype prediction using the clusters constructed in the previous stage.  
This implementation is adapted from the [hier-mil repository](https://github.com/minhchaudo/hier-mil), and we thank **Chau Do** and **Harri Lähdesmäki** for making their code publicly available.


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
  --cell_type_annot_key optimal_cluster \
  --attn1 1 \
  --device cuda \
  --n_tune_trials 30 \
  --n_folds_hyperparam_tune 5 \
  --n_folds 5 \
  --n_repeats 5 \
  --n_epochs 100 \
  --output ./result/[dataset_name]/[dataset_name]_result.txt
```

> **Note:**  
> Training configurations can be modified depending on dataset size and computational resources.
> For example, increasing `--n_tune_trials` improves hyperparameter optimization,  
> while adjusting `--n_folds` or `--n_repeats` balances evaluation stability and runtime.  
> For full argument details and model architecture explanations, please refer to the [hier-mil repository](https://github.com/minhchaudo/hier-mil).


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
