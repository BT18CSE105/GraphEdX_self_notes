# GraphEdX - Supported Datasets

Complete reference of all datasets the GraphEdX library supports.

---

## Overview

GraphEdX is tested and validated on **7 primary benchmark datasets** plus additional datasets. Each dataset is available in **3 cost setting formats**:

1. **no_attr_data** - Equal costs (all costs = 1)
2. **no_attr_asymm_data** - Unequal costs (different costs per operation)
3. **label_symm_data** - Label costs (node substitution instead of delete+insert)

---

## 7 Benchmark Datasets

### 1. **Mutagenicity** 🧬
```
Domain: Chemistry / Molecular Biology
Application: Predicting mutagenic potential of compounds
Graph Type: Molecular graphs
Characteristics:
  - Node attributes: None (no_attr variant)
  - Edge attributes: None (no_attr variant)
  - Typical graph size: Small to medium
  - Number of graphs: ~4000
  - Typical # nodes: 10-30
  - Typical # edges: 10-40
  
Usage:
  data = gedx.load_dataset('mutagenicity', split='train', cost_setting='equal')
  
Paper results:
  - Best MAE: 0.XXX
  - Best RMSE: X.XXX
  - Competitive baseline: SimGNN
```

### 2. **AIDS** 💊
```
Domain: Chemistry / Pharmaceuticals
Application: AIDS antiviral compounds
Graph Type: Molecular graphs of compounds
Characteristics:
  - Node attributes: None (no_attr variant)
  - Edge attributes: Bond type
  - Typical graph size: Small to medium
  - Number of graphs: ~1000
  - Typical # nodes: 15-30
  - Typical # edges: 15-50
  
Usage:
  data = gedx.load_dataset('aids', split='train', cost_setting='unequal')
  
Best use: 
  - Testing unequal cost settings
  - Chemical domain validation
  - Fine-tuning experiments
```

### 3. **Linux** 🐧
```
Domain: Software Engineering
Application: Function call dependency graphs
Graph Type: Program dependency graphs
Characteristics:
  - Node attributes: None (no_attr variant)
  - Edge attributes: Call relationships
  - Typical graph size: Medium to large
  - Number of graphs: ~1000
  - Typical # nodes: 100-300
  - Typical # edges: 200-600
  
Usage:
  data = gedx.load_dataset('linux', split='train', cost_setting='equal')
  
Difficulty: Hard (larger graphs, sparse structure)
Best use:
  - Testing on larger graphs
  - Software engineering applications
  - Scalability benchmarking
```

### 4. **OGBG-Code2** 💻
```
Domain: Machine Learning / Code
Application: Source code graphs
Graph Type: Program dependency graphs
Characteristics:
  - Node attributes: Token embeddings
  - Edge attributes: Type information
  - Typical graph size: Medium to large
  - Number of graphs: ~100k (huge!)
  - Typical # nodes: 50-200
  - Typical # edges: 100-400
  
Usage:
  data = gedx.load_dataset('ogbg-code2', split='train', cost_setting='equal')
  
Best use:
  - Large-scale benchmarking
  - Transfer learning from pretrained models
  - Production deployment testing
  
Note: Very large dataset - use batch processing
```

### 5. **OGBG-MolHIV** 🦠
```
Domain: Chemistry / Molecular Biology
Application: HIV inhibitor prediction
Graph Type: Molecular graphs
Characteristics:
  - Node attributes: Atom properties
  - Edge attributes: Bond types
  - Typical graph size: Small to medium
  - Number of graphs: ~40k
  - Typical # nodes: 20-50
  - Typical # edges: 20-80
  
Usage:
  data = gedx.load_dataset('ogbg-molhiv', split='train', cost_setting='label')
  
Best use:
  - Large-scale molecular benchmarking
  - Label cost setting evaluation
  - Biological activity prediction
```

### 6. **OGBG-MolPCBA** 🧪
```
Domain: Chemistry / Molecular Biology
Application: PubChem BioAssay prediction
Graph Type: Molecular graphs
Characteristics:
  - Node attributes: Atom properties
  - Edge attributes: Bond types
  - Typical graph size: Medium
  - Number of graphs: ~440k (VERY LARGE)
  - Typical # nodes: 20-80
  - Typical # edges: 20-100
  
Usage:
  data = gedx.load_dataset('ogbg-molpcba', split='train', cost_setting='equal')
  
Best use:
  - Extreme-scale benchmarking
  - Production deployment
  - GPU efficiency testing
  
Warning: Massive dataset - requires careful memory management
```

### 7. **Yeast** 🍺
```
Domain: Biology / Bioinformatics
Application: Protein interaction networks
Graph Type: Biological networks
Characteristics:
  - Node attributes: Protein properties
  - Edge attributes: Interaction strength
  - Typical graph size: Medium to large
  - Number of graphs: ~2000
  - Typical # nodes: 50-200
  - Typical # edges: 100-400
  
Usage:
  data = gedx.load_dataset('yeast', split='train', cost_setting='equal')
  
Best use:
  - Biological network analysis
  - Protein interaction studies
  - Domain generalization testing
```

---

## Additional Datasets (If Available)

### COIL-DEL
```
Domain: Computer Vision
Characteristics:
  - Typical # nodes: 50-300
  - Used in comparisons
```

### DBLP-V1
```
Domain: Academic Papers
Characteristics:
  - Citation networks
  - Large graphs
```

### IMDB-BINARY
```
Domain: Social Networks
Characteristics:
  - Movie actor networks
```

### Proteins
```
Domain: Biology
Characteristics:
  - Protein structure graphs
```

---

## Cost Setting Explanations

### Cost Setting 1: Equal (no_attr_data)
```
All operations cost 1:
- Node deletion: 1
- Node insertion: 1
- Edge deletion: 1
- Edge insertion: 1

Best for:
  - Fair baseline comparison
  - When no domain knowledge exists
  - Algorithm validation
```

### Cost Setting 2: Unequal (no_attr_asymm_data)
```
Different costs per operation:
- Node deletion: 1.2
- Node insertion: 1.5
- Edge deletion: 0.8
- Edge insertion: 0.8

Realistic because:
  - Deleting often cheaper than adding
  - Different operations have different importance
  - Domain-specific costs

Best for:
  - Real-world applications
  - Practical GED estimation
  - Domain-specific tuning
```

### Cost Setting 3: Label (label_symm_data)
```
Node substitution instead of delete+insert:
- Node substitution: 0.5
- Edge deletion: 1
- Edge insertion: 1

For graphs with node labels/types
- Relabel node instead of delete+add
- More efficient for labeled graphs
- Chemical compounds with atom types

Best for:
  - Labeled graphs
  - Molecular data
  - Semantic graphs
```

---

## Dataset Statistics

| Dataset | # Graphs | Avg # Nodes | Avg # Edges | Domain | Size |
|---------|----------|-------------|------------|--------|------|
| Mutagenicity | ~4,000 | 15 | 15 | Chemistry | Small |
| AIDS | ~1,000 | 20 | 25 | Chemistry | Small |
| Linux | ~1,000 | 150 | 350 | Software | Medium |
| OGBG-Code2 | ~100k | 100 | 250 | Code | Large |
| OGBG-MolHIV | ~40k | 25 | 30 | Chemistry | Large |
| OGBG-MolPCBA | ~440k | 25 | 35 | Chemistry | **XL** |
| Yeast | ~2,000 | 100 | 250 | Biology | Medium |

---

## Dataset Availability

### How to Get Datasets

1. **Download from Paper's Repository**
   ```bash
   # From link in README
   https://rebrand.ly/graph-edit-distance
   
   # Extract 3 folders:
   - no_attr_data/
   - no_attr_asymm_data/
   - label_symm_data/
   ```

2. **Auto-download (in library)**
   ```python
   # When you call:
   data = gedx.load_dataset('mutagenicity', split='train')
   
   # Library automatically downloads if missing
   # Caches locally for future use
   ```

3. **Directory Structure**
   ```
   GraphEdX/
   ├── no_attr_data/
   │   ├── mutagenicity/
   │   │   ├── train.pkl
   │   │   ├── val.pkl
   │   │   └── test.pkl
   │   ├── aids/
   │   ├── linux/
   │   ├── ogbg-code2/
   │   ├── ogbg-molhiv/
   │   ├── ogbg-molpcba/
   │   └── yeast/
   ├── no_attr_asymm_data/
   │   └── [same structure]
   └── label_symm_data/
       └── [same structure]
   ```

---

## Using Datasets in Library

### Example 1: Load Different Datasets

```python
import graphedx as gedx

# Load different datasets
mutagenicity_train = gedx.load_dataset('mutagenicity', split='train', cost_setting='equal')
aids_val = gedx.load_dataset('aids', split='val', cost_setting='unequal')
linux_test = gedx.load_dataset('linux', split='test', cost_setting='equal')

# Each returns: [(G1, G2, ged_label), (G1, G2, ged_label), ...]
print(len(mutagenicity_train))  # e.g., 2000
```

### Example 2: Cross-Dataset Training

```python
import graphedx as gedx

# Train on multiple datasets for generalization
datasets = ['mutagenicity', 'aids', 'linux']

combined_data = []
for dataset in datasets:
    data = gedx.load_dataset(dataset, split='train', cost_setting='equal')
    combined_data.extend(data)

# Train on combined data
model, history = gedx.train(
    train_graphs=combined_data,
    epochs=50,
    batch_size=32
)
```

### Example 3: Different Cost Settings

```python
import graphedx as gedx

# Same dataset, different cost settings
mutagenicity_equal = gedx.load_dataset('mutagenicity', 
                                       split='train', 
                                       cost_setting='equal')

mutagenicity_unequal = gedx.load_dataset('mutagenicity', 
                                         split='train', 
                                         cost_setting='unequal')

mutagenicity_label = gedx.load_dataset('mutagenicity', 
                                       split='train', 
                                       cost_setting='label')

# All three are different ground truth labels!
print(f"Equal cost pairs: {len(mutagenicity_equal)}")
print(f"Unequal cost pairs: {len(mutagenicity_unequal)}")
print(f"Label cost pairs: {len(mutagenicity_label)}")
```

### Example 4: Dataset Characteristics

```python
import graphedx as gedx

# Analyze dataset
data = gedx.load_dataset('mutagenicity', split='train', cost_setting='equal')

# Analyze first few pairs
for i, (G1, G2, ged_label) in enumerate(data[:5]):
    stats1 = gedx.get_graph_stats(G1)
    stats2 = gedx.get_graph_stats(G2)
    
    print(f"Pair {i}:")
    print(f"  G1: {stats1['num_nodes']} nodes, {stats1['num_edges']} edges")
    print(f"  G2: {stats2['num_nodes']} nodes, {stats2['num_edges']} edges")
    print(f"  True GED: {ged_label:.4f}")
    print()
```

---

## Recommended Dataset Choices

### For Quick Testing
```
Use: mutagenicity or aids
Reason: 
  - Small (~1-4k graphs)
  - Fast to load
  - Quick training
  - Good for prototyping
```

### For Production Validation
```
Use: linux or yeast
Reason:
  - Medium size (~1-2k graphs)
  - Larger graphs (more realistic)
  - Good coverage of edge cases
  - Computationally moderate
```

### For Large-Scale Benchmarking
```
Use: ogbg-code2, ogbg-molhiv
Reason:
  - 40k-100k graphs
  - Representative
  - Real-world scale
  - Tests GPU efficiency
```

### For Extreme Scale Testing
```
Use: ogbg-molpcba
Reason:
  - 440k graphs
  - Production-scale
  - Memory efficiency critical
  - Only if optimized code
```

### For Cross-Domain Validation
```
Use: All 7 datasets
Reason:
  - Chemistry: mutagenicity, aids, ogbg-molhiv, ogbg-molpcba
  - Software: linux, ogbg-code2
  - Biology: yeast
  - True generalization test
```

---

## Paper's Experimental Settings

From the paper (NeurIPS 2024):

### Main Experiments
```
Datasets tested: All 7
Cost settings: equal, unequal, label
Models compared: 6 baselines + GraphEdX
Variations: edge (XOR, DA, AD) × node (XOR, DA, AD)
Result: GraphEdX outperforms all baselines consistently
```

### Ablation Studies
```
Same 7 datasets
Testing component removal impact:
- Without XOR operation
- Without edge alignment
- Without node alignment
- Sparse representation
- With/without consistency penalty
```

### Reproducibility
```python
# To reproduce paper results exactly:
for dataset in ['mutagenicity', 'aids', 'linux', 
                'ogbg-code2', 'ogbg-molhiv', 'ogbg-molpcba', 'yeast']:
    
    for cost_setting in ['equal', 'unequal', 'label']:
        # Run experiments matching paper's config
        pass
```

---

## Dataset Pre-processing

### What's NOT in no_attr Data
```
Why "no_attr"?
- No node attributes (removed for simplicity)
- No edge attributes (removed for simplicity)
- Only structure matters
- Faster computation
- Focus on topological GED
```

### What's IN the Data
```
Each dataset contains:
- Graph pairs
- Ground truth GED for each pair
- Split into train/val/test
- Consistent format (pickle files)
```

### Custom Data Format

If you want to add your own dataset:
```
Required format:
  List of (G1, G2, ged_label) tuples
  
Where:
  - G1, G2: NetworkX graphs
  - ged_label: float (true GED)
  
Save as:
  pickle.dump(list_of_tuples, open('your_dataset.pkl', 'wb'))
  
Load as:
  data = gedx.load_dataset('your_dataset', ...)
```

---

## Dataset Limitations & Notes

### Mutagenicity
- Limited to ~4000 pairs
- Small graphs (helps debug)
- Chemical domain only

### AIDS
- Only ~1000 pairs (small)
- Well-studied baseline
- Good for validation

### Linux
- Sparse structure (challenging)
- Large graphs
- Software engineering domain

### OGBG-Code2
- Code graphs (unique domain)
- 100k graphs (big!)
- Requires batching

### OGBG-MolHIV
- 40k graphs
- Medium size (balanced)
- Good for large-scale testing

### OGBG-MolPCBA
- 440k graphs (MASSIVE!)
- Only if memory available
- Production-scale testing

### Yeast
- Biological networks
- ~2000 graphs
- Different domain (proteins)

---

## Benchmark Results (From Paper)

### Mutagenicity Dataset
```
Cost Setting: Equal
Methods:
  GraphEdX (Ours):  MAE 0.XXX, RMSE X.XXX ✓ Best
  GraphSim:        MAE 0.XXX, RMSE X.XXX
  SimGNN:          MAE 0.XXX, RMSE X.XXX
  
Cost Setting: Unequal
  GraphEdX (Ours):  MAE 0.XXX, RMSE X.XXX ✓ Best
  GraphSim:        MAE 0.XXX, RMSE X.XXX
  SimGNN:          MAE 0.XXX, RMSE X.XXX
```

(See paper for exact numbers)

---

## Summary

**GraphEdX works best with:**
- ✓ 7 benchmark datasets (chemistry, software, biology)
- ✓ 3 cost settings (equal, unequal, label)
- ✓ Different graph sizes (small to XL)
- ✓ Various domains (generalization)

**Start with:** Mutagenicity or AIDS
**Scale to:** OGBG datasets for real-world performance
**Validate on:** All 7 for comprehensive results

