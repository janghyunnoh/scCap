#!/bin/bash
# ============================================================
# scCap Tutorial Script
# Runs the full pipeline (Preprocessing → Clustering → Prediction)
# using pre-generated tutorial data.
# ============================================================

set -e  # Exit immediately if a command exits with a non-zero status

# ============================================================
# 1. USER SETTINGS
# ============================================================
### >>> MODIFY THESE IF NEEDED <<<

# GPU ID (change this if multiple GPUs are available)
GPU_ID=0    # Example: 0, 1, 2, 3

# Root tutorial directory (already created)
TUTORIAL_DIR="./tutorial"

### <<< DO NOT MODIFY BELOW UNLESS NECESSARY >>>
# ============================================================

# Define paths
DATA_DIR="${TUTORIAL_DIR}/data"
RESULT_DIR="${TUTORIAL_DIR}/result"

INPUT_TUTORIAL="${DATA_DIR}/tutorial_dataset.h5ad"
PREPROCESSED="${DATA_DIR}/tutorial_preprocessed.h5ad"
CONSTRUCTED="${DATA_DIR}/tutorial_constructed.h5ad"
RESULT_FILE="${RESULT_DIR}/tutorial_result.txt"

# Define hier-mil path (absolute)
HIERMIL_PATH="$(realpath "$(dirname "$0")/../hier-mil/run.py")"

echo "====================================================================="
echo "scCap Tutorial Pipeline"
echo "---------------------------------------------------------------------"
echo "Root tutorial directory:   $TUTORIAL_DIR"
echo "---------------------------------------------------------------------"
echo "Input tutorial data:       $INPUT_TUTORIAL"
echo "Preprocessed output:       $PREPROCESSED"
echo "Constructed clusters:      $CONSTRUCTED"
echo "Result file:               $RESULT_FILE"
echo "Hier-MIL script:           $HIERMIL_PATH"
echo "GPU ID:                    $GPU_ID"
echo "====================================================================="
echo ""

# ============================================================
# 2. PREPROCESSING
# ============================================================
echo "[Step 1] Preprocessing..."
python preprocess.py \
  --input "$INPUT_TUTORIAL" \
  --output "$PREPROCESSED"

echo "Preprocessing completed."
echo ""

# ============================================================
# 3. CLUSTERING
# ============================================================
echo "[Step 2] Clustering..."
python clustering.py \
  --input "$PREPROCESSED" \
  --output "$CONSTRUCTED" \
  --n_pcs 5 \
  --resolution 1.0 \
  --threshold 0.5 \
  --ratios 2.0 2.1 \
  --init-space raw \
  --refine-space scgpt

echo "Clustering completed."
echo ""

# ============================================================
# 4. PREDICTION (Hier-MIL)
# ============================================================
echo "[Step 3] Phenotype prediction with Hier-MIL..."
CUDA_VISIBLE_DEVICES=$GPU_ID python "$HIERMIL_PATH" \
  --data_path "$CONSTRUCTED" \
  --task 2 \
  --patient_id_key patient \
  --label_key label \
  --cell_type_annot_key optimal_cluster \
  --attn1 1 \
  --device cuda \
  --n_tune_trials 2 \
  --n_folds_hyperparam_tune 2 \
  --n_folds 2 \
  --n_repeats 2 \
  --n_epochs 10 \
  --output "$RESULT_FILE"

echo "Prediction completed."
echo ""

# ============================================================
# 5. SUMMARY
# ============================================================
echo "Tutorial pipeline successfully finished."
echo "--------------------------------------------------------------"
echo "Preprocessed file : $PREPROCESSED"
echo "Clustered file     : $CONSTRUCTED"
echo "Result file        : $RESULT_FILE"
echo "=============================================================="
