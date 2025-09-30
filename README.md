# DocMMIR: A Framework for Document Multi-modal Information Retrieval

<div align="center">

![DocMMIR Overview](images.png)

[![arXiv](https://img.shields.io/badge/arXiv-2505.19312-b31b1b.svg)](https://arxiv.org/abs/2505.19312)
[![Dataset](https://img.shields.io/badge/HF%20Dataset-Lord--Jim%2FDocMMIR-yellow)](https://huggingface.co/datasets/Lord-Jim/DocMMIR)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

</div>

## Overview

**DocMMIR** is a comprehensive framework for document-level multimodal information retrieval. This repository contains:
- Training and evaluation code for multimodal retrieval models
- Data processing scripts for ArXiv, Wikipedia, and Slide datasets
- Multiple baseline models (CLIP, BLIP, ColPali, VisualBERT, etc.)
- Evaluation metrics and analysis tools
- Question-answer generation pipeline

## Features

- Support for multiple multimodal encoders (CLIP, BLIP, VisualBERT, ColPali, etc.)
- Flexible fusion strategies (weighted sum, MLP, attention)
- Document-level retrieval with both text and image modalities
- Distributed training with PyTorch Lightning
- Comprehensive evaluation metrics (MRR, Recall@K, NDCG)
- Neptune.ai integration for experiment tracking
- Cumulative learning experiments

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Git LFS (for downloading large model checkpoints)

### Setup

```bash
# Clone the repository
git clone https://github.com/J1mL1/DocMMIR.git
cd DocMMIR

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Repository Structure

```
DocMMIR/
├── src/                          # Source code
│   ├── dataset.py               # Dataset loading and preprocessing
│   ├── retrieval_model.py       # Main retrieval model (PyTorch Lightning)
│   ├── fusion_module.py         # Multimodal fusion strategies
│   ├── metrics.py               # Evaluation metrics
│   ├── utils.py                 # Utility functions
│   ├── encoders/                # Text and image encoders
│   │   ├── text_model.py       # Text encoder wrapper
│   │   └── image_model.py      # Image encoder wrapper
│   ├── models/                  # Baseline model implementations
│   │   ├── clip_model.py       # CLIP-based model
│   │   ├── blip_model.py       # BLIP-based model
│   │   ├── colpali_model.py    # ColPali model
│   │   ├── visualBert_model.py # VisualBERT model
│   │   ├── e5v_model.py        # E5V model
│   │   ├── vlm2vec_model.py    # VLM2Vec model
│   │   └── ...                 # Other baselines
│   ├── train/                   # Training scripts
│   │   ├── train.py            # Standard training
│   │   └── train_cumulative.py # Cumulative learning experiments
│   └── test/                    # Testing scripts
│       ├── test.py             # Model evaluation
│       ├── compute_embeddings.py # Precompute document embeddings
│       └── batch_embedding.py  # Batch embedding computation
├── scripts/                      # Data processing scripts
│   ├── arxiv_scripts/           # ArXiv dataset processing
│   ├── wikipages_scripts/       # Wikipedia dataset processing
│   ├── slides_scripts/          # Slides dataset processing
│   └── ...                      # Utility scripts
├── qa_generation/                # Question-answer generation
│   ├── qwen7b_qa_generate.py   # QA generation with Qwen
│   ├── qwen_perplexity.py      # Quality evaluation
│   └── split_json.py           # Data splitting
├── outputs/                      # Model outputs and checkpoints
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Quick Start

### 1. Download the Dataset

Download the DocMMIR dataset from Hugging Face:

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Download dataset
huggingface-cli download Lord-Jim/DocMMIR --repo-type dataset --local-dir ./data/DocMMIR

# Extract images
cd data/DocMMIR
cat archives/image.tar.gz.part* > image.tar.gz
tar -xzf image.tar.gz
rm image.tar.gz
cd ../..
```

### 2. Train a Model

```bash
# Example: Train a CLIP-based model
python src/train/train.py \
    --text_model bert-base-uncased \
    --image_model openai/clip-vit-large-patch14 \
    --data_path data/DocMMIR/full_dataset.json \
    --image_dir data/DocMMIR/media \
    --fusion_strategy weighted_sum \
    --batch_size 32 \
    --max_epochs 10 \
    --devices 4 \
    --experiment_name clip_vit_l_14_full
```

**Key Training Parameters:**
- `--text_model`: Text encoder (e.g., bert-base-uncased, roberta-base)
- `--image_model`: Image encoder (e.g., openai/clip-vit-large-patch14)
- `--fusion_strategy`: Fusion method (weighted_sum, mlp, attention)
- `--batch_size`: Training batch size
- `--devices`: Number of GPUs
- `--experiment_name`: Experiment identifier

For all training options:
```bash
python src/train/train.py --help
```

### 3. Evaluate a Model

```bash
# Evaluate on test set
python src/test/test.py \
    --text_model bert-base-uncased \
    --image_model openai/clip-vit-large-patch14 \
    --checkpoint_path outputs/checkpoints/your_model/best.ckpt \
    --data_path data/DocMMIR/full_dataset.json \
    --image_dir data/DocMMIR/media \
    --fusion_strategy weighted_sum \
    --batch_size 64

# Zero-shot evaluation (no fine-tuning)
python src/test/test.py \
    --text_model bert-base-uncased \
    --image_model openai/clip-vit-large-patch14 \
    --data_path data/DocMMIR/full_dataset.json \
    --image_dir data/DocMMIR/media \
    --fusion_strategy weighted_sum \
    --zero_shot \
    --batch_size 64
```






### Model Implementations (`src/models/`)

The repository includes various multimodal baseline models:

| Model | File | Description |
|-------|------|-------------|
| **CLIP** | `clip_model.py` | Contrastive Language-Image Pre-training |
| **BLIP** | `blip_model.py` | Bootstrapping Language-Image Pre-training |
| **ColPali** | `colpali_model.py` | Contextualized Late Interaction for Documents |
| **VisualBERT** | `visualBert_model.py` | Vision and Language BERT |
| **E5V** | `e5v_model.py` | E5 with Vision capabilities |
| **VLM2Vec** | `vlm2vec_model.py` | Vision-Language Model to Vector |
| **MARVEL** | `MARVEL.py` | Multimodal Representation Learning |
| **SigLIP** | `siglip2_model.py` | Sigmoid Loss for Language-Image Pre-training |
| **BERT** | `bert_model.py` | Text-only BERT baseline |
| **ALIGN** | `align_model.py` | Large-scale noisy image-text alignment |

Each model can be used as a backbone for the retrieval framework.

## Usage Guide

### Training

#### Basic Training

```bash
python src/train/train.py \
    --text_model bert-base-uncased \
    --image_model openai/clip-vit-large-patch14 \
    --data_path data/DocMMIR/full_dataset.json \
    --image_dir data/DocMMIR/media \
    --fusion_strategy weighted_sum \
    --batch_size 32 \
    --max_epochs 10 \
    --lr_text 2e-5 \
    --lr_image 2e-5 \
    --devices 4 \
    --strategy ddp \
    --experiment_name my_experiment
```

#### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--text_model` | Text encoder model name | Required |
| `--image_model` | Image encoder model name | Required |
| `--data_path` | Path to training JSON file | Required |
| `--image_dir` | Directory containing images | Required |
| `--fusion_strategy` | Fusion method (weighted_sum, mlp, attention) | weighted_sum |
| `--batch_size` | Training batch size | 32 |
| `--max_epochs` | Maximum training epochs | 10 |
| `--lr_text` | Learning rate for text encoder | 2e-5 |
| `--lr_image` | Learning rate for image encoder | 2e-5 |
| `--weight_decay` | Weight decay for optimization | 0.01 |
| `--loss_type` | Loss function (infonce, bce) | infonce |
| `--devices` | Number of GPUs | 1 |
| `--strategy` | Training strategy (ddp, dp, fsdp) | ddp |
| `--precision` | Training precision (32, 16, bf16) | 32 |
| `--experiment_name` | Experiment identifier | Required |
| `--output_dir` | Output directory for checkpoints | outputs/checkpoints |

### Evaluation

#### Standard Evaluation

```bash
python src/test/test.py \
    --text_model bert-base-uncased \
    --image_model openai/clip-vit-large-patch14 \
    --checkpoint_path outputs/checkpoints/my_experiment/best.ckpt \
    --data_path data/DocMMIR/test.json \
    --image_dir data/DocMMIR/media \
    --fusion_strategy weighted_sum \
    --batch_size 64 \
    --devices 1
```

#### Zero-Shot Evaluation

Evaluate pretrained models without fine-tuning:

```bash
python src/test/test.py \
    --text_model bert-base-uncased \
    --image_model openai/clip-vit-large-patch14 \
    --data_path data/DocMMIR/test.json \
    --image_dir data/DocMMIR/media \
    --fusion_strategy weighted_sum \
    --zero_shot \
    --batch_size 64
```

#### Global Retrieval Evaluation

For large-scale retrieval across all documents:

```bash
# Step 1: Precompute document embeddings
python src/test/compute_embeddings.py \
    --checkpoint_path outputs/checkpoints/my_experiment/best.ckpt \
    --data_path data/DocMMIR/full_dataset.json \
    --image_dir data/DocMMIR/media \
    --output_dir embeddings/my_experiment

# Step 2: Run global retrieval test
python src/test/test.py \
    --checkpoint_path outputs/checkpoints/my_experiment/best.ckpt \
    --is_global_test \
    --doc_dir embeddings/my_experiment \
    --batch_size 64
```

#### Domain-Specific Experiments

Train and test on individual domains:

```bash
# ArXiv domain
python src/train/train.py \
    --data_path data/DocMMIR/arxiv_train.json \
    --experiment_name arxiv_only \
    ...

# Wikipedia domain
python src/train/train.py \
    --data_path data/DocMMIR/wiki_train.json \
    --experiment_name wiki_only \
    ...

# Slides domain
python src/train/train.py \
    --data_path data/DocMMIR/slide_train.json \
    --experiment_name slide_only \
    ...
```

#### Modality Ablation Studies

Test with single modality:

```bash
# Text-only retrieval
python src/train/train.py \
    --modality text_only \
    --text_model bert-base-uncased \
    --data_path data/DocMMIR/full_dataset.json \
    --experiment_name text_only_ablation

# Image-only retrieval
python src/train/train.py \
    --modality image_only \
    --image_model openai/clip-vit-large-patch14 \
    --data_path data/DocMMIR/full_dataset.json \
    --image_dir data/DocMMIR/media \
    --experiment_name image_only_ablation
```

## Supported Models

The framework supports various encoder models. Model selection is based on keywords in the model name:

### Multimodal Encoders (Support Both Text and Image)

| Model | Keyword Trigger | Example Model Names | Text Encoder | Image Encoder |
|-------|----------------|---------------------|--------------|---------------|
| **CLIP** | `vit` | `ViT-B-32`, `ViT-L-14`, `openai/clip-vit-base-patch32` | Yes | Yes |
| **SigLIP2** | `siglip2` | `siglip2-base`, `google/siglip-base-patch16-224` | Yes | Yes |
| **BLIP** | `blip` | `Salesforce/blip-image-captioning-base` | Yes | Yes |
| **VLM2Vec** | `vlm2vec` | `vlm2vec-base`, `vlm2vec-large` | Yes | Yes |
| **ALIGN** | `align` | `align-base`, `kakaobrain/align-base` | Yes | Yes |
| **E5V** | `e5v` | `e5v-base`, `e5v-large` | Yes | Yes |
| **ColPali** | `colpali` | `vidore/colpali` | Yes | Yes |

### Text-Only Encoders

| Model | Keyword Trigger | Example Model Names | Notes |
|-------|----------------|---------------------|-------|
| **BERT** | `bert` | `bert-base-uncased`, `bert-large-uncased`, `roberta-base` | Supports BERT and RoBERTa variants |

### Usage Notes

1. **Model Name Matching**: The framework detects which model to use based on keywords in the model name string (case-insensitive)
2. **Priority Order**: Checks are performed in the order listed in the code. If multiple keywords match, the first match is used
3. **Pretrained Weights**: Use `--pretrained_weights` to specify custom checkpoint paths
4. **Custom Models**: To add new models, implement them in `src/models/` and register in the encoder files

### Example Model Specifications

```bash
# CLIP models (triggered by "vit" keyword)
--text_model ViT-B-32 --image_model ViT-B-32
--text_model ViT-L-14 --image_model ViT-L-14

# BERT text encoder with CLIP image encoder
--text_model bert-base-uncased --image_model ViT-L-14

# SigLIP2 multimodal
--text_model siglip2-base --image_model siglip2-base

# BLIP multimodal
--text_model blip-base --image_model blip-base

# ColPali multimodal
--text_model colpali --image_model colpali

# E5V multimodal
--text_model e5v-base --image_model e5v-base

# VLM2Vec multimodal
--text_model vlm2vec-base --image_model vlm2vec-base

# ALIGN multimodal
--text_model align-base --image_model align-base
```

## Data Format

### Input JSON Format

The training/testing data should be in JSON format with the following structure:

```json
[
  {
    "id": "arxiv_12345",
    "title": "Deep Learning for Computer Vision",
    "class": "computer_science",
    "query": "What are the main applications of deep learning in computer vision?",
    "texts": [
      "Deep learning has revolutionized computer vision...",
      "Convolutional neural networks are the backbone...",
      "..."
    ],
    "images": [
      "arxiv/12345/figure1.jpg",
      "arxiv/12345/figure2.jpg"
    ],
    "num_images": 2,
    "domain": "arxiv"
  },
  ...
]
```

### Required Fields
- `id`: Unique identifier for the document
- `query`: Query text for retrieval
- `texts`: List of text chunks from the document
- `images`: List of image file paths (relative to dataset root)
- `num_images`: Number of images in the document
- `domain`: Data source (arxiv, wiki, slide)

### Optional Fields
- `title`: Document title
- `class`: Document category/topic

## Advanced Features

### Multi-GPU Training

```bash
# Data Parallel (DP)
python src/train/train.py --devices 4 --strategy dp ...

# Distributed Data Parallel (DDP) - Recommended
python src/train/train.py --devices 4 --strategy ddp ...

# Fully Sharded Data Parallel (FSDP) - For very large models
python src/train/train.py --devices 8 --strategy fsdp ...
```

### Mixed Precision Training

```bash
# FP16 (faster, less memory)
python src/train/train.py --precision 16 ...

# BF16 (better numerical stability, requires Ampere+ GPUs)
python src/train/train.py --precision bf16 ...
```

## Performance Tips

1. **Data Loading**: Use `--num_workers 8` or higher for faster data loading
2. **Caching**: Precompute embeddings for frequent evaluation
3. **Mixed Precision**: Use `--precision 16` for 2x speedup
4. **Batch Size**: Maximize batch size within GPU memory limits
5. **Distributed Training**: Use multiple GPUs with `--strategy ddp`

## Examples

### Example 1: Train CLIP Model on Full Dataset

```bash
python src/train/train.py \
    --text_model bert-base-uncased \
    --image_model openai/clip-vit-large-patch14 \
    --data_path data/DocMMIR/full_dataset.json \
    --image_dir data/DocMMIR/media \
    --fusion_strategy weighted_sum \
    --loss_type infonce \
    --batch_size 32 \
    --max_epochs 10 \
    --devices 4 \
    --strategy ddp \
    --precision 16 \
    --experiment_name clip_vit_l_14_fullset
```

### Example 2: Train with Different Fusion Strategies

```bash
# Weighted sum fusion
python src/train/train.py --fusion_strategy weighted_sum --experiment_name fusion_ws ...

# MLP fusion
python src/train/train.py --fusion_strategy mlp --experiment_name fusion_mlp ...

# Attention fusion
python src/train/train.py --fusion_strategy attention --experiment_name fusion_attn ...
```

---

## Citation

If you use this code or dataset in your research, please cite our paper:

```bibtex
@article{li2025docmmir,
  title   = {DocMMIR: A Framework for Document Multi-modal Information Retrieval},
  author  = {Li, Zirui and Wu, Siwei and Li, Yizhi and Wang, Xingyu and Zhou, Yi and Lin, Chenghua},
  journal = {arXiv preprint arXiv:2505.19312},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.19312}
}
```

## Resources

- **Paper**: [DocMMIR on arXiv](https://arxiv.org/abs/2505.19312)
- **Dataset**: [DocMMIR on Hugging Face](https://huggingface.co/datasets/Lord-Jim/DocMMIR)
- **Code**: [GitHub Repository](https://github.com/J1mL1/DocMMIR)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project builds upon several excellent open-source projects:
- [PyTorch Lightning](https://lightning.ai/) - Deep learning framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pretrained models
- [OpenCLIP](https://github.com/mlfoundations/open_clip) - CLIP implementations
- [Neptune.ai](https://neptune.ai/) - Experiment tracking

## Contact

For questions or issues, please:
- Open an issue on [GitHub](https://github.com/J1mL1/DocMMIR/issues)
- Contact the authors via email (see paper)


