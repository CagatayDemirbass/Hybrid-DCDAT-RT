# Code & Models — Thesis Reproducibility Package

This package accompanies the thesis and contains all notebooks required to reproduce **pretraining**, **ablations** (including denoising), and the **final/best model evaluation**. All datasets are downloaded at runtime; compiled CUDA ops are built inside Colab.

> **Environment used for results**
> - **Platform:** Google Colab
> - **GPU:** NVIDIA **A100**
> - **Python:** **3.11**
> - **PyTorch:** **2.3.1+cu121**

---

## Folder Layout

```
Code_and_Models/
├─Ablation_FULL_SCRIPT.ipynb
├─Ablation_Gatefuse.ipynb
├─Ablation_Denoising.ipynb
├─Backbone_(optional_neck)_Pretraining_FULL_SCRIPT.ipynb
├─Best_Model.ipynb
├─Gradient_Flow_Diagnostic.ipynb
├─README.md
```

*No raw datasets are bundled in the ZIP.*

*Important note (libraries and versions). All of the required python libraries and versions are in the scripts ready to install and run..*
---

## Quick Start (Colab)

1. **Runtime** → *Change runtime type* → **GPU (A100)**.
2. Open a notebook (e.g., `notebooks/Ablation_FULL_SCRIPT.ipynb`).
3. Run the setup cell(s):

```bash
!python --version

# Core
!pip install torch==2.3.1+cu121 torchvision --extra-index-url https://download.pytorch.org/whl/cu121
!pip install numpy==1.26.4 tqdm pycocotools torchmetrics lxml scipy
!pip install -U cmake ninja wheel

# (Pretraining notebook only)
!pip install -U "datasets>=2.17.0" "pyarrow>=14.0"
!pip install --no-binary=mmcv mmcv==2.2.0
```

4. **Build CUDA ops** (run these in the detection notebooks before training):
```bash
# NATTEN v0.14.6
git clone --depth 1 --branch v0.14.6 https://github.com/SHI-Labs/NATTEN.git
cd NATTEN
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0"   # A100
pip install .
cd ..

# DAT (cloned for completeness)
git clone --depth 1 https://github.com/LeapLabTHU/DAT.git

# DCNv4
git clone --depth 1 https://github.com/OpenGVLab/DCNv4.git
cd DCNv4/DCNv4_op
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0"
python -m pip install . --no-build-isolation -v
cd ../..
```

5. **Download datasets** (see next section), then run training/evaluation cells.

---

## Datasets

### Pascal VOC 2007 & 2012 (for detection)
Downloaded inside detection notebooks via **Kaggle** using `opendatasets`:

```bash
# Put your Kaggle token at ~/.kaggle/kaggle.json (permission 600)
!mkdir -p ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json
!pip -q install opendatasets
```

```python
import opendatasets as od
od.download(
    "https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012",
    data_dir="/content/voc"
)
# Path used in notebooks:
# VOC_ROOT = "/content/voc/pascal-voc-2007-and-2012"
# which contains: VOCdevkit/{VOC2007, VOC2012}
```

### Mini‑ImageNet (100 classes, for pretraining)
Loaded in the pretraining notebook via Hugging Face Datasets:
```python
from datasets import load_dataset
train_ds = load_dataset("timm/mini-imagenet", split="train")       # 50,000
val_ds   = load_dataset("timm/mini-imagenet", split="validation")  # 10,000
```

---

## Notebooks

### `Ablation_FULL_SCRIPT.ipynb` — all ablations **except** denoising
- End‑to‑end detection on **VOC 2007+2012**.
- Contains model, loss, data, training, and evaluation (mAP@0.5:0.95 & mAP@0.5).
- Builds/uses **NATTEN v0.14.6**, **DCNv4**, clones **DAT**.
- **Typical config:** `NUM_CLASSES=20`, `NUM_QUERIES=200`, `IMG_SIZE=480`, `BATCH_SIZE=8`, `EPOCHS=50`, `BASE_LR=1e-4`, `ACCUM_STEPS=3`, `SEED=42`.
- **Decoding (non‑DN):** conf `0.05`, Top‑K `100`, background gate `(1 - P(bg))` on, optional IoU‑aware scoring.
- **Checkpoints:** `/content/drive/MyDrive/voc_hybrid_yeni/{latest.pth,best.pth,history.json}`.
- **Optional pretrained init:** set `PRETRAIN_PATH="/content/pretrained_backbone_ablation.pth"`  
  (or point to `pretrained_backbone_neck.pth` exported by pretraining).

### `Ablation_Gatefuse.ipynb` — GateFuse variant
- Same pipeline as the full script, but with the **GateFuse** model.
- Either run this notebook or swap the model cell in the full script.

### `Ablation_Denoising.ipynb` — denoising ablation
- End‑to‑end detection with **denoising** in model and its own criterion loss.
- DN losses warmed up via `adjust_dn_weights` (from 1/3 to full).
- Same training loop and checkpointing layout as the full script.

### `Backbone_(optional_neck)_Pretraining_FULL_SCRIPT.ipynb` — Mini‑ImageNet pretraining
- Pretrains **backbone** and optionally a **light neck** on Mini‑ImageNet (100 classes).
- **Augmentations:** RandomResizedCrop(224), Flip, RandAugment, ColorJitter, Normalize, RandomErasing; Val: Resize(256)+CenterCrop(224).
- **Training:** `num_epochs=200`, AdamW (parameter‑group multipliers), warm‑up + cosine LR, optional EMA, strict FP32 (TF32 disabled).
- **Outputs to Drive:** `config.json`, `latest.pth`, `best_model.pth`, **`pretrained_backbone_neck.pth`** (for detection notebooks).

### `Best_Model.ipynb` — final/best model evaluation
- Loads the best checkpoint and evaluates on VOC.
- Default weights: `/content/drive/MyDrive/voc_hybrid_yeni/best.pth` (change via `WEIGHTS_PATH`).
- Prints mAP@0.5:0.95 and mAP@0.5; can be used for quick inference.


### `Gradient_Flow_Diagnostic.ipynb` — Advanced Gradient Flow Diagnoser

**What it is.** A lightweight diagnostic notebook to **detect vanishing/low/high gradients** and **non‑updating parameters** during detector training. It tracks per‑parameter gradient stats and parameter changes, then prints a structured report.

**How it works.**
- Monitors each trainable tensor every iteration (norm/mean/std/zero‑ratio).
- Flags: _vanishing_ (‖g‖ < 1e‑7), _low_ (‖g‖ < 1e‑5), _high_ (‖g‖ > 10.0).
- Tracks **never updated**, **no‑update despite gradient**, **unchanged from init**.
- Prints module‑level breakdown and a “healthy parameters” list.

**Defaults** (editable at the top of the notebook):
- `ITERS=2000`, `BATCH_SIZE=8`, `LR=2e-4`, `IMG_SIZE=480`, `NUM_CLASSES=20`
- Thresholds: `VANISHING_GRAD_THRESHOLD=1e-7`, `LOW_GRAD_THRESHOLD=1e-5`, `HIGH_GRAD_THRESHOLD=10.0`,
  `PARAM_CHANGE_THRESHOLD=1e-8`
- Dataset root: `VOC_ROOT=/content/voc/pascal-voc-2007-and-2012`

**Inputs & assumptions.**
- Uses the same `build_model`, `build_criterion`, and `build_optimizer` utilities as the detection notebooks.
- Loads Pascal VOC via the in‑notebook dataset wrapper (consistent with the ablation scripts).

**Outputs.**
- Console summary: _Advanced Gradient Flow Analysis_ (with critical issues & healthy params).
- Text reports saved to the working directory:
  - `gradient_flow_detailed_report.txt` (full per‑param table)
  - `gradient_analysis_report.txt` (category-wise summary)

> Notes: This Gradient_Flow_Diagnostic.ipynb **does not aim to reach high mAP**; it performs a short run to expose gradient/parameter pathologies. It is a **diagnostic aid**, not part of the official ablation results.



**Usage (Colab).**
1. Install the same dependencies used in `Ablation_FULL_SCRIPT.ipynb` (PyTorch 2.3.1+cu121, NATTEN, DCNv4, etc.).
2. Ensure Pascal VOC is downloaded to `/content/voc/pascal-voc-2007-and-2012` (or update `VOC_ROOT`).
3. Run the notebook cells; optionally tune `ITERS` and thresholds.
4. Inspect the printed analysis and the two text reports.
---

## Ablations Supported but Not Separately Packaged (PGI, CSI, IoU)

To keep the repo concise, ablations like **PGI**, **CSI**, and **IoU** variants are not shipped as separate notebooks. They are enabled by small edits in `Ablation_FULL_SCRIPT.ipynb`:

- **Model cell** (search `build_model(`): toggle/swap the submodule or flag to activate PGI/CSI components in backbone/neck/decoder.
- **Criterion cell:** if needed, adjust the loss dictionary (add terms or weights).
- **Decoding/Eval cell (IoU‑aware):** set `use_iou_head=True` in `decode_outputs(...)`; it supports an optional `pred_iou` and a `gamma` exponent for re‑weighting.

> **Denoising** is the only ablation that requires a distinct pipeline, hence it has its own notebook.

---


## Dependencies

- **Core:** Python 3.11, PyTorch **2.3.1+cu121**, `torchvision`, `numpy==1.26.4`, `cmake`, `ninja`, `wheel`
- **CUDA ops:** **NATTEN v0.14.6**, **DCNv4** (built from source; `FORCE_CUDA=1`, `TORCH_CUDA_ARCH_LIST="8.0"`)
- **Training/IO:** `tqdm`, `pycocotools`, `torchmetrics`, `lxml`, `scipy`
- **Data:** `opendatasets` (Kaggle), `datasets>=2.17.0`, `pyarrow>=14.0` (pretraining)
- **Pretraining only:** `mmcv==2.2.0` (install with `--no-binary=mmcv`)

A sample `requirements.txt` (PyTorch via extra index, not plain PyPI):
```txt
numpy==1.26.4
tqdm
pycocotools
torchmetrics
lxml
scipy
opendatasets
datasets>=2.17.0
pyarrow>=14.0
cmake
ninja
wheel
mmcv==2.2.0
```

---

## Troubleshooting

- **Kaggle 403/401:** ensure `~/.kaggle/kaggle.json` exists and permission is `600`.
- **CUDA build errors (NATTEN/DCNv4):**
  ```bash
  export FORCE_CUDA=1
  export TORCH_CUDA_ARCH_LIST="8.0"   # A100
  ```
  Rebuild in a fresh runtime after installing `cmake`, `ninja`, `wheel`.
- **Torch/CUDA mismatch:** use torch **2.3.1+cu121** with the CUDA 12.1 extra index.
- **Different GPUs (e.g., T4):** set `TORCH_CUDA_ARCH_LIST` appropriately (e.g., `"7.5"`) and reinstall compiled extensions.
- **OOM:** reduce `BATCH_SIZE` / `IMG_SIZE`, or increase `ACCUM_STEPS`.

---

- **NATTEN** (Neighborhood Attention) — `SHI-Labs/NATTEN`  
- **DCNv4** — `OpenGVLab/DCNv4`  
- **DAT** (cloned for completeness) — `LeapLabTHU/DAT`  
- **Hugging Face Datasets** — `timm/mini-imagenet`  
- **PyTorch**, **torchvision**, **mmcv`

---

