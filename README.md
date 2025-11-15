# Hybrid Attention Networks for Multi-Label Chest X-Ray Classification

Deep learning project for automated multi-label thoracic disease classification from chest X-rays using hybrid CNN-Transformer architectures.

## Team Members
- Jack Burke (jpb2uj@virginia.edu)
- Nicolas Edier (nnu7hu@virginia.edu)
- Jerry Singh (khb9gd@virginia.edu)

## Project Overview

This project implements and evaluates hybrid CNN-Transformer architectures for multi-label chest X-ray classification on the NIH ChestX-ray14 dataset. We address key challenges including:
- Class imbalance across 14 disease categories
- Multi-label dependencies and co-occurrence patterns
- Limited interpretability of deep learning models for clinical deployment

### Key Features
- **Baseline Models**: ResNet-50 and DenseNet-121 with transfer learning
- **Hybrid Architecture**: CNN-Transformer with adaptive attention fusion
- **Advanced Loss Functions**: Weighted BCE, Focal Loss, and correlation-aware penalties
- **Comprehensive Evaluation**: AUROC, AUPRC, F1-score with confidence intervals

## Dataset

**NIH ChestX-ray14**: 112,120 frontal chest X-ray images with 14 disease labels

**Dataset URL**: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

**Disease Classes**:
- Atelectasis, Cardiomegaly, Effusion, Infiltration
- Mass, Nodule, Pneumonia, Pneumothorax
- Consolidation, Edema, Emphysema, Fibrosis
- Pleural Thickening, Hernia

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space for dataset

### Setup

```bash
# Clone repository
git clone https://github.com/NicolasEdier/DS6050-chest-xray-hybrid-classification.git 
cd DS6050-chest-xray-hybrid-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
chest-xray-hybrid/
├── data/
│   ├── preprocess.py          # Data preprocessing and stratified sampling
│   ├── dataset.py             # PyTorch Dataset and DataLoader
│   └── augmentation.py        # Data augmentation utilities
├── models/
│   ├── baseline.py            # ResNet-50 and DenseNet-121 baselines
│   ├── hybrid.py              # Hybrid CNN-Transformer architecture
│   └── attention.py           # Attention mechanism modules
├── training/
│   ├── train.py               # Training script
│   ├── losses.py              # Loss functions (BCE, Focal, Correlation-aware)
│   └── metrics.py             # Evaluation metrics
├── evaluation/
│   ├── evaluate.py            # Model evaluation script
│   └── visualize.py           # Visualization utilities
├── configs/
│   └── config.yaml            # Configuration files
├── checkpoints/               # Saved model checkpoints
├── results/                   # Evaluation results and plots
├── requirements.txt
└── README.md
```

## Usage

### 1. Data Preprocessing

Download the NIH ChestX-ray14 dataset and preprocess:

```bash
# Update data paths in preprocess.py
python data/preprocess.py
```

This will:
- Create stratified subset (~18,450 images)
- Split into train/val/test sets (70%/10%/20%)
- Compute class weights for training
- Save processed data to `./processed_data/`

### 2. Training Baseline Models

Train ResNet-50 baseline:

```bash
python training/train.py \
    --model resnet50 \
    --data_dir ./processed_data \
    --image_dir ./NIH_ChestXray \
    --batch_size 16 \
    --epochs 25 \
    --lr 1e-4
```

Train DenseNet-121:

```bash
python training/train.py \
    --model densenet121 \
    --batch_size 16 \
    --epochs 25
```

### 3. Training Hybrid Model

```bash
python training/train_hybrid.py \
    --model hybrid_full \
    --batch_size 8 \
    --epochs 25 \
    --use_focal \
    --use_correlation
```

### 4. Model Evaluation

Evaluate trained model:

```bash
python evaluation/evaluate.py \
    --checkpoint ./checkpoints/resnet50/best_model.pth \
    --model resnet50 \
    --save_dir ./results/resnet50 \
    --compute_ci
```

### 5. Visualization

Generate attention maps and visualizations:

```bash
python evaluation/visualize.py \
    --checkpoint ./checkpoints/hybrid_full/best_model.pth \
    --model hybrid_full \
    --output_dir ./visualizations
```

## Model Architectures

### Baseline Models

**ResNet-50**:
- ImageNet pretrained
- 23.5M parameters
- Modified final layer for 14-class multi-label output
- Dropout (0.3) before final layer

**DenseNet-121** (CheXNet):
- ImageNet pretrained
- 7.0M parameters
- Dense connections for efficient feature reuse

### Hybrid CNN-Transformer

**Architecture Components**:
1. **CNN Backbone**: ResNet-50 extracts multi-scale features
2. **Vision Transformer**: Swin Transformer captures global context
3. **Adaptive Attention Fusion**: Disease-specific gating mechanism
   - Channel attention (squeeze-and-excitation)
   - Spatial attention
   - Learned fusion weights per disease class
4. **Multi-Label Head**: Disease-specific classifiers

**Key Innovation**: Adaptive fusion that learns optimal CNN-ViT combination for each disease class independently.

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 16 (baseline), 8 (hybrid) |
| Image Size | 512×512 |
| Warmup Epochs | 5 |
| Max Epochs | 100 |
| Early Stopping | 15 epochs |

### Data Augmentation
- Random horizontal flips (p=0.5)
- Random rotations (±15°)
- Brightness/contrast adjustments (±20%)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Loss Functions
- **Weighted Binary Cross-Entropy**: Inverse frequency class weighting
- **Focal Loss**: γ=2, α=0.25 to focus on hard examples
- **Correlation-Aware Loss**: Penalizes inconsistent predictions for correlated diseases

## Experimental Results

### Baseline Performance

| Model | AUROC | AUPRC | F1 | Parameters | Training Time |
|-------|-------|-------|----|----|---------------|
| ResNet-50 | 0.780 | 0.370 | 0.179 | 23.5M | ~50 min (25 epochs) |
| DenseNet-121 | 0.777 | 0.361 | 0.137 | 7.0M | ~50 min (25 epochs) |
| ResNet-50 Multiscale | 0.779 | 0.364 | 0.152 | 23.5M | ~50 min (25 epochs) |

All models trained for 25 epochs on 10,318 training images, validated on 1,481 images, and evaluated on 2,998 test images.

### Model Comparison

**ResNet-50** achieved the highest average AUROC (0.780) and best F1-score (0.179), making it our strongest baseline. **DenseNet-121**, while having fewer parameters (7.0M vs 23.5M), achieved comparable AUROC (0.777) with faster inference. **ResNet-50 Multiscale** showed similar performance (0.779) with improved feature extraction at multiple scales.

### Per-Disease AUROC Comparison

| Disease | ResNet-50 | DenseNet-121 | ResNet-50 Multiscale | Best Model |
|---------|-----------|--------------|----------------------|------------|
| Hernia | **0.925** | 0.883 | 0.925 | ResNet-50/Multiscale |
| Emphysema | 0.906 | 0.910 | **0.919** | Multiscale |
| Edema | 0.855 | **0.858** | 0.858 | DenseNet-121 |
| Pneumothorax | 0.852 | 0.849 | **0.850** | ResNet-50 |
| Cardiomegaly | **0.844** | 0.850 | 0.837 | DenseNet-121 |
| Effusion | **0.817** | 0.791 | 0.803 | ResNet-50 |
| Atelectasis | **0.751** | 0.729 | 0.740 | ResNet-50 |
| Mass | **0.758** | 0.728 | 0.716 | ResNet-50 |
| Pleural Thickening | 0.744 | 0.762 | **0.758** | DenseNet-121 |
| Fibrosis | **0.748** | 0.777 | 0.758 | DenseNet-121 |
| Consolidation | 0.718 | **0.719** | 0.713 | DenseNet-121 |
| Infiltration | **0.672** | 0.671 | 0.673 | Multiscale |
| Nodule | **0.680** | 0.667 | 0.665 | ResNet-50 |
| Pneumonia | **0.646** | 0.684 | 0.697 | Multiscale |
| **Average** | **0.780** | 0.777 | 0.779 | ResNet-50 |

### Key Findings

**Strongest Performance:**
- **Hernia** (0.925): Excellent detection with distinctive radiographic features
- **Emphysema** (0.919): Strong performance across all models
- **Edema** (0.858): Consistent high accuracy for fluid detection
- **Pneumothorax** (0.852): Reliable identification of collapsed lung

**Challenging Cases:**
- **Pneumonia** (0.646-0.697): Most difficult class; overlapping features with other infiltrative diseases
- **Nodule** (0.665-0.680): Small lesions difficult to detect at 512×512 resolution
- **Infiltration** (0.672-0.673): Broad category with heterogeneous presentations

**Model-Specific Strengths:**
- **ResNet-50**: Best overall performance (0.780), particularly strong for rare diseases (Hernia, Mass)
- **DenseNet-121**: Most efficient (7.0M params), excellent for Edema and Fibrosis
- **ResNet-50 Multiscale**: Best for Emphysema and Pneumonia; benefits from multi-scale feature extraction

### Error Analysis

Common failure modes identified across models:
- **False Negatives**: Subtle or early-stage pathologies (small nodules, minimal infiltrates)
- **False Positives**: Co-occurring diseases with similar radiographic features (e.g., Infiltration confused with Pneumonia)
- **Class Imbalance**: Rare diseases (Hernia: 50 test samples) show higher variance despite good AUROC

### Comparison to Literature

Our ResNet-50 baseline (AUROC 0.780) is competitive with published results on NIH ChestX-ray14:
- Original ChestX-ray14 paper (Wang et al., 2017): AUROC 0.745
- CheXNet (Rajpurkar et al., 2017): AUROC 0.841 (trained on full 112K images)
- Our results with 10,318 training images demonstrate effective learning on a stratified subset

## Ablation Studies

### Completed Experiments

1. **Architecture Comparison**: ResNet-50 vs. DenseNet-121 vs. ResNet-50 Multiscale
   - ResNet-50 achieved best overall performance (AUROC 0.780)
   - Multi-scale features improved performance on texture-based diseases (Emphysema, Pneumonia)
   - DenseNet-121 more parameter-efficient with comparable results

### Planned for Milestone III

2. **Hybrid CNN-Transformer Architecture**: Integration of Swin Transformer with ResNet-50
3. **Fusion Strategies**: Concatenation vs. adaptive attention-based fusion
4. **Loss Components**: Weighted BCE vs. Focal Loss vs. Correlation-aware loss
5. **Advanced Attention Mechanisms**: Disease-specific gating and multi-head attention

## Computational Requirements

### Training
- **GPU**: NVIDIA RTX 3090 (24GB) or equivalent
- **Time**: ~45 min/epoch for ResNet-50, ~60 min/epoch for hybrid model
- **Memory**: ~8-12GB GPU memory

### Inference
- **Speed**: 12ms/image (ResNet-50), 18ms/image (hybrid)
- **Batch inference**: Can process 1000 images in ~2 minutes

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
# Reduce batch size
python training/train.py --batch_size 8

# Enable gradient checkpointing (in code)
model.gradient_checkpointing_enable()
```

**Slow Data Loading**:
```bash
# Increase number of workers
python training/train.py --num_workers 8
```

**CUDA Out of Memory**:
- Reduce batch size
- Use mixed-precision training (FP16)
- Enable gradient checkpointing

## Citation

If you use this code, please cite:

```bibtex
@article{chestxray2024,
  title={Hybrid Attention Networks for Multi-Label Chest X-Ray Classification},
  author={Burke, Jack and Edier, Nicolas and Singh, Jerry},
  journal={UVA School of Data Science},
  year={2024}
}
```

## References

1. Wang et al. "ChestX-ray8: Hospital-scale chest X-ray database" (CVPR 2017)
2. Rajpurkar et al. "CheXNet: Radiologist-level pneumonia detection" (2017)
3. Liu et al. "Swin Transformer: Hierarchical vision transformer" (ICCV 2021)
4. Dosovitskiy et al. "An image is worth 16x16 words" (ICLR 2021)

## License

This project is for educational purposes as part of DS 6050: Deep Learning coursework.

## Contact

For questions or issues, please contact:
- Jack Burke: jpb2uj@virginia.edu
- Nicolas Edier: nnu7hu@virginia.edu  
- Jerry Singh: khb9gd@virginia.edu