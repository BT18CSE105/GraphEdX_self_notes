# GraphEdX - Training & Pretrained Models

Complete information about which datasets the pretrained models are trained on.

---

## Overview:

GraphEdX has **separate pretrained models for EACH dataset**.

The paper trains and evaluates GraphEdX on each of the 7 datasets independently, then provides pretrained checkpoints for each.

**Key Point:** There is NO single universal pretrained model. Instead:
- Model trained on **mutagenicity** → best for mutagenicity
- Model trained on **aids** → best for aids
- Model trained on **linux** → best for linux
- ... and so on

---

## Pretrained Models Available

### Model Checkpoint Structure

```
checkpoints/
├── equal/                          # Cost setting
│   └── GRAPHEDX_xor_on_node/       # Model variant
│       └── weights/
│           ├── mutagenicity*_best.pt
│           ├── aids*_best.pt
│           ├── linux*_best.pt
│           ├── ogbg-code2*_best.pt
│           ├── ogbg-molhiv*_best.pt
│           ├── ogbg-molpcba*_best.pt
│           └── yeast*_best.pt
│
├── unequal/                        # Cost setting
│   └── GRAPHEDX_xor_on_node/
│       └── weights/
│           ├── mutagenicity*_best.pt
│           ├── aids*_best.pt
│           ├── linux*_best.pt
│           ├── ogbg-code2*_best.pt
│           ├── ogbg-molhiv*_best.pt
│           ├── ogbg-molpcba*_best.pt
│           └── yeast*_best.pt
│
└── label/                          # Cost setting
    └── GRAPHEDX_xor_on_node/
        └── weights/
            ├── mutagenicity*_best.pt
            ├── aids*_best.pt
            ├── linux*_best.pt
            ├── ogbg-code2*_best.pt
            ├── ogbg-molhiv*_best.pt
            ├── ogbg-molpcba*_best.pt
            └── yeast*_best.pt
```

**Total pretrained models: 7 datasets × 3 cost settings = 21 checkpoints**

Plus additional checkpoints for model variations (DA, AD, etc.)

---

## Using Pretrained Models

### Example 1: Load Model Trained on Specific Dataset

```python
import graphedx as gedx

# Load model trained on mutagenicity
model = gedx.load_model(
    cost_setting='equal',
    edge_var='XOR',
    node_var='AD',
    checkpoint_path='checkpoints/equal/GRAPHEDX_xor_on_node/weights/mutagenicity_LAM0.5_best.pt',
    device='cuda'
)

# This model was trained on mutagenicity dataset
# Use it to compute GED on mutagenicity test set
test_data = gedx.load_dataset('mutagenicity', split='test', cost_setting='equal')
metrics = gedx.evaluate(model, test_data)
```

### Example 2: Test Cross-Dataset Transfer

```python
import graphedx as gedx

# Load model trained on AIDS
aids_model = gedx.load_model(
    cost_setting='equal',
    checkpoint_path='checkpoints/equal/GRAPHEDX_xor_on_node/weights/aids_LAM0.5_best.pt'
)

# Test on different dataset (transfer learning)
linux_test = gedx.load_dataset('linux', split='test', cost_setting='equal')

# Will this AIDS-trained model work on Linux graphs?
predictions = []
for G1, G2, true_ged in linux_test[:100]:
    pred = gedx.compute_ged(G1, G2, model=aids_model)
    predictions.append(pred)

# Likely poor performance (domain mismatch)
# But shows transfer learning possibility
```

### Example 3: Auto-loading Correct Checkpoint

```python
import graphedx as gedx

# When using library's auto-loading:
ged = gedx.compute_ged(G1, G2, cost_setting='equal')

# Internally, library knows:
# 1. Which dataset you're working with
# 2. Which cost_setting
# 3. Loads appropriate checkpoint automatically
# 4. Or trains new model if checkpoint missing
```

---

## Paper's Training Procedure

### Phase 1: Individual Dataset Training

```
For each dataset in [mutagenicity, aids, linux, ogbg-code2, ogbg-molhiv, ogbg-molpcba, yeast]:
    For each cost_setting in [equal, unequal, label]:
        Split data: 70% train, 15% val, 15% test
        
        Initialize model
        For each epoch:
            Train on 70% data
            Validate on 15% data
            Save best checkpoint (by validation MAE)
        
        Final test on 15% data (report MAE, RMSE)
```

### Phase 2: Cross-Dataset Validation

```
For each model trained on Dataset_A:
    Test on all 7 datasets
    Report generalization performance
```

### Phase 3: Ablation Studies

```
For each dataset:
    For each ablation variant (remove component):
        Train model variant
        Test and compare
        Measure impact
```

---

## What Each Checkpoint Contains

### Checkpoint File Format

```python
checkpoint = {
    'model_state_dict': model.state_dict(),  # All model weights
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch_number,
    'best_val_mae': best_validation_mae,
    'best_val_rmse': best_validation_rmse,
    'config': training_config,
    'dataset_name': 'mutagenicity',  # Which dataset trained on
    'cost_setting': 'equal'
}
```

### What you can infer from filename:

```
mutagenicity_LAM0.5_XOR_AD_best.pt
│             │     │   │    │
│             │     │   │    └─ Best checkpoint (not last)
│             │     │   └───── Node variation (AD = AlignDiff)
│             │     └───────── Edge variation (XOR)
│             └───────────────── Lambda parameter (alignment vs cost weight)
└──────────────────────────── Dataset trained on
```

---

## Training Details by Dataset

### Mutagenicity Model
```
Trained on: mutagenicity training set (~2800 pairs)
Validated on: mutagenicity validation set (~600 pairs)
Tested on: mutagenicity test set (~600 pairs)

Training details:
- Epochs: ~50-100
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Best validation MAE: [see paper]
- Test MAE: [see paper]

Checkpoint location:
checkpoints/equal/GRAPHEDX_xor_on_node/weights/mutagenicity_LAM0.5_best.pt
checkpoints/unequal/GRAPHEDX_xor_on_node/weights/mutagenicity_LAM0.5_best.pt
checkpoints/label/GRAPHEDX_xor_on_node/weights/mutagenicity_LAM0.5_best.pt
```

### AIDS Model
```
Trained on: aids training set (~700 pairs)
Validated on: aids validation set (~150 pairs)
Tested on: aids test set (~150 pairs)

Training details:
- Similar to mutagenicity
- Smaller dataset → may overfit easier

Checkpoint location:
checkpoints/equal/GRAPHEDX_xor_on_node/weights/aids_LAM0.5_best.pt
checkpoints/unequal/GRAPHEDX_xor_on_node/weights/aids_LAM0.5_best.pt
checkpoints/label/GRAPHEDX_xor_on_node/weights/aids_LAM0.5_best.pt
```

### Linux Model
```
Trained on: linux training set (~700 pairs)
Validated on: linux validation set (~150 pairs)
Tested on: linux test set (~150 pairs)

Challenge:
- Much larger graphs (100-300 nodes)
- Sparser structure
- Slower training/inference

Checkpoint location:
checkpoints/equal/GRAPHEDX_xor_on_node/weights/linux_LAM0.5_best.pt
checkpoints/unequal/GRAPHEDX_xor_on_node/weights/linux_LAM0.5_best.pt
checkpoints/label/GRAPHEDX_xor_on_node/weights/linux_LAM0.5_best.pt
```

### OGBG-Code2 Model
```
Trained on: ogbg-code2 training set (~70k pairs)
Validated on: ogbg-code2 validation set (~15k pairs)
Tested on: ogbg-code2 test set (~15k pairs)

Challenge:
- HUGE dataset (100k pairs)
- Requires efficient batching
- Long training time

Checkpoint location:
checkpoints/equal/GRAPHEDX_xor_on_node/weights/ogbg-code2_LAM0.5_best.pt
checkpoints/unequal/GRAPHEDX_xor_on_node/weights/ogbg-code2_LAM0.5_best.pt
checkpoints/label/GRAPHEDX_xor_on_node/weights/ogbg-code2_LAM0.5_best.pt
```

### OGBG-MolHIV Model
```
Trained on: ogbg-molhiv training set (~28k pairs)
Validated on: ogbg-molhiv validation set (~6k pairs)
Tested on: ogbg-molhiv test set (~6k pairs)

Advantage:
- Balanced size (large but manageable)
- Good for benchmarking

Checkpoint location:
checkpoints/equal/GRAPHEDX_xor_on_node/weights/ogbg-molhiv_LAM0.5_best.pt
checkpoints/unequal/GRAPHEDX_xor_on_node/weights/ogbg-molhiv_LAM0.5_best.pt
checkpoints/label/GRAPHEDX_xor_on_node/weights/ogbg-molhiv_LAM0.5_best.pt
```

### OGBG-MolPCBA Model
```
Trained on: ogbg-molpcba training set (~308k pairs)
Validated on: ogbg-molpcba validation set (~66k pairs)
Tested on: ogbg-molpcba test set (~66k pairs)

Challenge:
- MASSIVE dataset (440k pairs)
- Production-scale training
- Requires serious GPU memory

Checkpoint location:
checkpoints/equal/GRAPHEDX_xor_on_node/weights/ogbg-molpcba_LAM0.5_best.pt
checkpoints/unequal/GRAPHEDX_xor_on_node/weights/ogbg-molpcba_LAM0.5_best.pt
checkpoints/label/GRAPHEDX_xor_on_node/weights/ogbg-molpcba_LAM0.5_best.pt
```

### Yeast Model
```
Trained on: yeast training set (~1400 pairs)
Validated on: yeast validation set (~300 pairs)
Tested on: yeast test set (~300 pairs)

Domain:
- Biological networks (proteins)
- Different from chemistry & software

Checkpoint location:
checkpoints/equal/GRAPHEDX_xor_on_node/weights/yeast_LAM0.5_best.pt
checkpoints/unequal/GRAPHEDX_xor_on_node/weights/yeast_LAM0.5_best.pt
checkpoints/label/GRAPHEDX_xor_on_node/weights/yeast_LAM0.5_best.pt
```

---

## Transfer Learning Possibilities

### Within-Domain Transfer
```
Best performance:
  - Train on AIDS, test on AIDS ✓ Best
  - Train on mutagenicity, test on mutagenicity ✓ Best

Cross-domain but similar:
  - Train on AIDS (chemicals), test on mutagenicity (chemicals) ✓ Good
  - Train on ogbg-molhiv (molecules), test on ogbg-molpcba (molecules) ✓ Good

Expected: Reasonable transfer within chemistry domain
```

### Cross-Domain Transfer
```
May work but with degradation:
  - Train on AIDS (chemistry), test on Linux (software) ✗ Poor
  - Train on yeast (biology), test on linux (software) ✗ Poor

Expected: Significant performance drop
Reason: Different graph distributions, structure patterns
```

### Transfer Learning Strategy
```python
# For new domain:

# Option 1: Use existing pretrained + fine-tune
model = gedx.load_model(cost_setting='equal', checkpoint_path='...')
model, history = gedx.finetune(model, new_domain_data, epochs=10)

# Option 2: Train from scratch (recommended)
model, history = gedx.train(new_domain_data, epochs=100)
```

---

## Paper's Reported Results

### Mutagenicity (Equal Cost)
```
Model trained on: mutagenicity
Performance on mutagenicity test set:
  MAE: [see paper]
  RMSE: [see paper]

Compared to baselines:
  GraphSim: [see paper]
  SimGNN: [see paper]
  Result: GraphEdX is better ✓
```

(Similar tables for other datasets in paper)

---

## Checkpoint Download Instructions

### From Paper Link

```bash
# Download all checkpoints
wget https://rebrand.ly/graph-edit-distance

# Extract to correct location
unzip checkpoints.zip
# Creates:
#   checkpoints/
#   ├── equal/
#   ├── unequal/
#   └── label/
```

### Automatic in Library

```python
import graphedx as gedx

# First time:
model = gedx.load_model(cost_setting='equal')
# If checkpoint not found locally, auto-downloads

# Subsequent times:
model = gedx.load_model(cost_setting='equal')
# Loads from cache (no download)
```

---

## Training Your Own Models

### If You Want to Train from Scratch

```python
import graphedx as gedx

# Load data
train_data = gedx.load_dataset('mutagenicity', split='train', cost_setting='equal')
val_data = gedx.load_dataset('mutagenicity', split='val', cost_setting='equal')

# Train model
model, history = gedx.train(
    train_graphs=train_data,
    val_graphs=val_data,
    cost_setting='equal',
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device='cuda'
)

# Your model will be similar to paper's checkpoint
# but maybe slightly different (random initialization, etc.)

# Save it
gedx.save_model(model, 'my_mutagenicity_model.pt')
```

### Reproducing Paper Results

```bash
# Exactly reproduce paper's checkpoints:

# For mutagenicity, equal cost, XOR-AD variant:
./scripts/GraphEdX.sh train 0 mutagenicity equal XOR AD

# After training, checkpoint saved to:
# checkpoints/equal/GRAPHEDX_xor_on_node/weights/mutagenicity_LAM0.5_best.pt

# Then test:
./scripts/GraphEdX.sh test 0 mutagenicity equal XOR AD
```

---

## Summary

### Key Points

| Aspect | Details |
|--------|---------|
| **Pretrained Models** | 7 datasets × 3 cost settings = 21 checkpoints |
| **Training Strategy** | Each model trained independently on its dataset |
| **Best Use** | Use model trained on your target dataset |
| **Transfer** | Works within domain (chemistry→chemistry), poor across domains |
| **Download** | From paper's link or auto-download in library |
| **Training Your Own** | Supported via `gedx.train()` function |

### Recommendation

```
For best results:
1. Use pretrained checkpoint trained on YOUR dataset
2. If new dataset: fine-tune existing checkpoint
3. If very different domain: train from scratch

Example workflow:
- New chemical compound? → Use AIDS or mutagenicity checkpoint + finetune
- New software dependency graph? → Use Linux checkpoint + finetune
- Completely new domain? → Train from scratch
```

### Paper Citation

All pretrained checkpoints and training scripts available in:
- **Paper**: https://arxiv.org/abs/2409.17687
- **Github**: [Paper repository link]
- **Download**: https://rebrand.ly/graph-edit-distance

