# GraphEdX Library Functions - Feature Tiers

Categorization of library functions by implementation priority and sophistication level.

---

## MUST HAVE (MVP - Minimum Viable Product)

These are **essential functions** for a basic working library. Without these, the library doesn't fulfill its core purpose.

### Core Inference Functions

#### 1. **compute_ged(graph1, graph2, cost_setting='equal', model=None)** ⭐⭐⭐ CRITICAL
```python
Why: This is THE function - everything else exists to support this
Use: Single graph pair GED computation
Complexity: Medium
Effort: 1-2 days
LOC: 50-100

Key responsibilities:
- Accept two graphs (NetworkX or internal format)
- Load default model if not provided
- Handle cost settings
- Return single float value
- Error handling for invalid inputs
```

#### 2. **load_model(cost_setting='equal', edge_var='XOR', node_var='AD', checkpoint_path=None, device='cuda')** ⭐⭐⭐ CRITICAL
```python
Why: Without this, users can't use the model
Use: Initialize or load pretrained models
Complexity: Medium
Effort: 1-2 days
LOC: 80-120

Key responsibilities:
- Map cost_setting to correct config file
- Initialize GMN encoder and model architecture
- Load checkpoint (auto-download if needed)
- Handle device placement (GPU/CPU)
- Return ready-to-use model in eval mode
```

#### 3. **load_graph(file_path, format='networkx')** ⭐⭐⭐ CRITICAL
```python
Why: Users need way to get graphs into library
Use: Load graphs from files (.gml, .pickle, .json)
Complexity: Low-Medium
Effort: 1 day
LOC: 60-100

Key responsibilities:
- Detect file format
- Parse accordingly
- Handle missing files
- Preserve node/edge attributes
- Return consistent graph object
```

#### 4. **create_graph(nodes, edges, node_features=None, edge_features=None, directed=False)** ⭐⭐⭐ CRITICAL
```python
Why: Users need way to create graphs programmatically
Use: Build graphs from scratch
Complexity: Low
Effort: 0.5 days
LOC: 40-60

Key responsibilities:
- Validate inputs
- Create NetworkX graph
- Add features if provided
- Return graph object
```

### Data Loading

#### 5. **load_dataset(dataset_name, split='train', cost_setting='equal')** ⭐⭐⭐ CRITICAL
```python
Why: Need benchmark datasets for testing/training
Use: Load mutagenicity, aids, linux, etc.
Complexity: Medium
Effort: 1 day
LOC: 80-120

Key responsibilities:
- Map dataset_name to path
- Auto-download if missing
- Load pickle files
- Return (G1, G2, ged_label) tuples
- Handle different cost settings
```

#### 6. **compute_ged_batch(graph_pairs, cost_setting='equal', model=None)** ⭐⭐⭐ CRITICAL
```python
Why: Users need efficient batch processing (GPU acceleration)
Use: Compute GED for 100s or 1000s of pairs
Complexity: Medium
Effort: 1 day
LOC: 60-100

Key responsibilities:
- Create proper batches
- Handle variable graph sizes
- Forward pass on all batches
- Return list of scores
- Significant speedup vs sequential
```

### Utility Functions

#### 7. **get_graph_stats(graph)** ⭐⭐ IMPORTANT
```python
Why: Basic graph analysis users expect
Use: Get num_nodes, num_edges, density, avg_degree, diameter
Complexity: Low
Effort: 0.5 days
LOC: 40-60

Key responsibilities:
- Compute standard graph statistics
- Return dict format
- Handle edge cases (empty graphs, etc.)
```

#### 8. **evaluate(model, test_graphs)** ⭐⭐ IMPORTANT
```python
Why: Users need standard metrics (MSE, RMSE, MAE, R²)
Use: Evaluate model performance
Complexity: Low
Effort: 1 day
LOC: 50-80

Key responsibilities:
- Compute predictions
- Calculate MSE, RMSE, MAE
- Compute R² score
- Return metrics dict
```

---

## GOOD TO HAVE (Enhanced Features)

These functions **significantly improve usability** but aren't strictly necessary for core functionality.

### Advanced Inference

#### 9. **compute_ged_with_alignment(graph1, graph2, cost_setting='equal', model=None)** ⭐⭐
```python
Why: Users want to understand WHY GED is what it is
Use: Get alignment info, cost breakdown, debugging
Complexity: High
Effort: 2-3 days
LOC: 150-250

Key responsibilities:
- Extract alignment matrices (node, edge)
- Compute individual operation costs
- Provide confidence scores
- Return comprehensive dict with all details

Adds value through:
- Interpretability
- Debugging
- Understanding model decisions
```

#### 10. **compute_ged_matrix(graphs, cost_setting='equal', model=None)** ⭐⭐
```python
Why: All-pairs distance matrix useful for clustering, similarity search
Use: Analyze graph collections
Complexity: Medium
Effort: 1 day
LOC: 60-100

Key responsibilities:
- Create all pairs
- Batch compute efficiently
- Build symmetric NxN matrix
- Return numpy array
```

### Training & Fine-tuning

#### 11. **train(train_graphs, val_graphs=None, cost_setting='equal', epochs=100, ...)** ⭐⭐
```python
Why: Power users need to train on custom data
Use: Train from scratch or on domain-specific graphs
Complexity: High
Effort: 2-3 days
LOC: 200-300

Key responsibilities:
- Full training loop
- Loss computation (MSE + consistency penalty)
- Learning rate scheduling
- Checkpoint saving (best model)
- Validation tracking

Adds value through:
- Customization
- Domain adaptation
- Research capabilities
```

#### 12. **finetune(model, train_graphs, val_graphs=None, epochs=10, learning_rate=0.0001)** ⭐⭐
```python
Why: Transfer learning is powerful for specialized domains
Use: Adapt pretrained model to new data
Complexity: Medium
Effort: 1-2 days
LOC: 100-150

Key responsibilities:
- Load pretrained weights
- Train with lower learning rate
- Quick adaptation to new domain
- Return fine-tuned model
```

#### 13. **save_model(model, checkpoint_path)** ⭐⭐
```python
Why: Users need to persist trained models
Use: Save trained/fine-tuned models
Complexity: Low
Effort: 0.5 days
LOC: 30-50

Key responsibilities:
- Save model state dict
- Include configuration
- Create parent directories
- .pt format
```

### Configuration & Setup

#### 14. **set_cost_setting(cost_type='equal', costs=None)** ⭐⭐
```python
Why: Users need to customize costs for their domain
Use: Set custom node/edge operation costs
Complexity: Low
Effort: 0.5 days
LOC: 40-60

Key responsibilities:
- Validate cost_type
- Store in config
- Handle custom costs dict
- Return config object
```

#### 15. **get_default_config(cost_setting='equal', dataset='mutagenicity')** ⭐⭐
```python
Why: Reduces setup complexity with sane defaults
Use: Get pre-configured settings
Complexity: Low
Effort: 0.5 days
LOC: 40-60

Key responsibilities:
- Load YAML configs
- Merge dataset + model configs
- Return OmegaConf object
```

#### 16. **normalize_graph(graph)** ⭐⭐
```python
Why: Ensure consistent graph preprocessing
Use: Normalize node/edge features
Complexity: Low
Effort: 0.5 days
LOC: 40-80

Key responsibilities:
- Normalize node features (zero-mean, unit-var)
- Normalize edge features
- Handle missing attributes
```

### Data Utilities

#### 17. **batch_graphs(graphs, batch_size=32)** ⭐⭐
```python
Why: Memory efficient processing
Use: Create mini-batches for processing
Complexity: Low
Effort: 0.5 days
LOC: 30-50

Key responsibilities:
- Split into chunks
- Return list of batches
```

#### 18. **split_dataset(graphs, train_ratio=0.7, val_ratio=0.15)** ⭐⭐
```python
Why: Standard ML workflow
Use: Create train/val/test splits
Complexity: Low
Effort: 0.5 days
LOC: 40-60

Key responsibilities:
- Random shuffle
- Split by ratios
- Return (train, val, test)
```

### Visualization

#### 19. **visualize_alignment(graph1, graph2, alignment_result, output_path=None)** ⭐⭐
```python
Why: Pictures worth 1000 words for understanding alignments
Use: Visual analysis of node/edge matches
Complexity: Medium
Effort: 1-2 days
LOC: 100-150

Key responsibilities:
- Draw two graphs side-by-side
- Show alignment edges with confidence colors
- Save to PNG or display
- Handle large graphs gracefully
```

---

## PRO LEVEL (Advanced Features)

These functions are **sophisticated additions** for power users, researchers, and production systems. Optional but impressive.

### Advanced Analysis

#### 20. **validate_alignment_consistency(node_alignment, edge_alignment)** ⭐
```python
Why: Understand model quality and correctness
Use: Verify alignment consistency constraint from paper
Complexity: High (Math)
Effort: 1-2 days
LOC: 80-120

Key responsibilities:
- Check if node-edge alignments are consistent
- Compute consistency score
- Return boolean + score
- Implement paper's consistency formula

Why it's "pro":
- Requires understanding of the paper's math
- Enables quality assessment
- Useful for research/debugging
```

#### 21. **get_alignment_confidence(alignment_matrix)** ⭐
```python
Why: Uncertainty quantification in predictions
Use: Understand confidence of each alignment
Complexity: Medium
Effort: 0.5 days
LOC: 40-60

Key responsibilities:
- Extract max value per row
- Return confidence array
- Identify uncertain alignments
- Enable post-hoc filtering
```

#### 22. **compute_ged_matrix(graphs, cost_setting='equal', model=None)** ⭐ (moved here - complex optimization needed)
```python
Why: Power user feature for large-scale analysis
Use: All-pairs similarity search, clustering
Complexity: High (Memory optimization)
Effort: 1-2 days
LOC: 120-200

Key responsibilities:
- Optimize memory for large graphs
- Streaming computation possible
- GPU acceleration
- Efficient distance matrix computation

Why it's "pro":
- Requires optimization thinking
- Only needed for research/analysis at scale
- Complex memory management
```

### Baseline Comparisons

#### 23. **compare_methods(graph_pairs, methods=['graphedx', 'graphsim', 'simgnn'])** ⭐
```python
Why: Benchmark against competing methods
Use: Compare GraphEdX vs baselines
Complexity: Medium
Effort: 1 day
LOC: 80-120

Key responsibilities:
- Compute GED with multiple methods
- Collect results
- Format comparison output
- Return standardized results

Why it's "pro":
- Requires integration with multiple code bases
- Useful for research papers
- Shows relative performance
```

#### 24. **compute_graphsim_ged(graph1, graph2)** ⭐
#### 25. **compute_simgnn_ged(graph1, graph2)** ⭐
#### 26. **compute_exact_ged_heuristic(graph1, graph2)** ⭐
```python
Why: Enable fair method comparison
Use: Baseline implementations
Complexity: Medium
Effort: 0.5-1 day each
LOC: 50-100 each

Key responsibilities:
- Wrap baseline models
- Provide consistent interface
- Handle graph conversion
- Return GED scores

Why it's "pro":
- Requires understanding other algorithms
- Integration complexity
- Only useful for comparative studies
- Not needed for basic library use
```

### Configuration Advanced

#### 27. **load_config_from_file(yaml_path)** ⭐
```python
Why: Advanced users want full control
Use: Load custom YAML configs
Complexity: Low
Effort: 0.5 days
LOC: 30-50

Key responsibilities:
- Parse YAML
- Return OmegaConf object
- Handle missing files
- Validate config structure
```

### Specialized Utilities

#### 28. **compute_similarity_score(ged, max_ged=None)** ⭐
```python
Why: Convert GED to similarity for ML pipelines
Use: Normalize GED to [0, 1] scale
Complexity: Low
Effort: 0.25 days
LOC: 20-30

Key responsibilities:
- Invert GED to similarity
- Handle normalization
- Return score in [0, 1]
```

#### 29. **from_networkx(nx_graph)** & **to_networkx(graph)** ⭐
```python
Why: Interoperability with ecosystem
Use: Convert between formats
Complexity: Low
Effort: 0.5 days each
LOC: 30-50 each

Key responsibilities:
- Bidirectional format conversion
- Preserve all attributes
- Enable ecosystem integration
```

#### 30. **set_verbosity(level='info')** ⭐
```python
Why: Production-quality logging
Use: Control logging output
Complexity: Low
Effort: 0.25 days
LOC: 20-30

Key responsibilities:
- Set logger level
- Update loguru configuration
- Enable/disable verbose output
```

---

## Summary Table

| Function | Category | Priority | Effort | Impact |
|----------|----------|----------|--------|--------|
| `compute_ged()` | Core | MUST | 1-2d | ⭐⭐⭐ |
| `load_model()` | Core | MUST | 1-2d | ⭐⭐⭐ |
| `load_graph()` | I/O | MUST | 1d | ⭐⭐⭐ |
| `create_graph()` | I/O | MUST | 0.5d | ⭐⭐⭐ |
| `load_dataset()` | Data | MUST | 1d | ⭐⭐⭐ |
| `compute_ged_batch()` | Core | MUST | 1d | ⭐⭐⭐ |
| `get_graph_stats()` | Utils | MUST | 0.5d | ⭐⭐ |
| `evaluate()` | Training | MUST | 1d | ⭐⭐ |
| `compute_ged_with_alignment()` | Analysis | GOOD | 2-3d | ⭐⭐ |
| `compute_ged_matrix()` | Core | GOOD | 1d | ⭐⭐ |
| `train()` | Training | GOOD | 2-3d | ⭐⭐ |
| `finetune()` | Training | GOOD | 1-2d | ⭐⭐ |
| `save_model()` | I/O | GOOD | 0.5d | ⭐⭐ |
| `set_cost_setting()` | Config | GOOD | 0.5d | ⭐⭐ |
| `get_default_config()` | Config | GOOD | 0.5d | ⭐⭐ |
| `normalize_graph()` | Data | GOOD | 0.5d | ⭐⭐ |
| `batch_graphs()` | Utils | GOOD | 0.5d | ⭐⭐ |
| `split_dataset()` | Data | GOOD | 0.5d | ⭐⭐ |
| `visualize_alignment()` | Viz | GOOD | 1-2d | ⭐⭐ |
| `validate_alignment_consistency()` | Analysis | PRO | 1-2d | ⭐ |
| `get_alignment_confidence()` | Analysis | PRO | 0.5d | ⭐ |
| `compare_methods()` | Benchmark | PRO | 1d | ⭐ |
| `compute_graphsim_ged()` | Baseline | PRO | 0.5-1d | ⭐ |
| `compute_simgnn_ged()` | Baseline | PRO | 0.5-1d | ⭐ |
| `compute_exact_ged_heuristic()` | Baseline | PRO | 0.5d | ⭐ |
| `load_config_from_file()` | Config | PRO | 0.5d | ⭐ |
| `compute_similarity_score()` | Utils | PRO | 0.25d | ⭐ |
| `from_networkx()` | Convert | PRO | 0.5d | ⭐ |
| `to_networkx()` | Convert | PRO | 0.5d | ⭐ |
| `set_verbosity()` | Utils | PRO | 0.25d | ⭐ |

---

## Implementation Timeline

### Phase 1: MUST HAVE (2-3 weeks)
```
Week 1:
  compute_ged()                     (2 days)
  load_model()                      (2 days)
  load_graph()                      (1 day)
  create_graph()                    (0.5 days)

Week 2:
  load_dataset()                    (1 day)
  compute_ged_batch()               (1 day)
  get_graph_stats()                 (0.5 days)
  evaluate()                        (1 day)

Week 2-3:
  Testing & Integration             (2-3 days)

Total: ~12-14 days (2.5 weeks)
```

### Phase 2: GOOD TO HAVE (3-4 weeks)
```
Week 1:
  train()                           (2-3 days)
  finetune()                        (1-2 days)
  save_model()                      (0.5 days)

Week 2:
  compute_ged_with_alignment()      (2-3 days)
  visualize_alignment()             (1-2 days)
  compute_ged_matrix()              (1 day)

Week 2-3:
  set_cost_setting()                (0.5 days)
  get_default_config()              (0.5 days)
  normalize_graph()                 (0.5 days)
  batch_graphs()                    (0.5 days)
  split_dataset()                   (0.5 days)

Week 3-4:
  Testing & Integration             (3-4 days)

Total: ~18-22 days (3.5-4 weeks)
```

### Phase 3: PRO LEVEL (2-3 weeks)
```
Week 1:
  compare_methods()                 (1 day)
  compute_graphsim_ged()            (0.5-1 days)
  compute_simgnn_ged()              (0.5-1 days)
  compute_exact_ged_heuristic()     (0.5 days)

Week 2:
  validate_alignment_consistency()  (1-2 days)
  get_alignment_confidence()        (0.5 days)
  load_config_from_file()           (0.5 days)

Week 2-3:
  Utility functions                 (1 day)
  Testing & Integration             (2-3 days)

Total: ~12-16 days (2-3 weeks)
```

---

## Recommended MVP Release

**Minimum features for v0.1 release:**
- `compute_ged()` ✓
- `load_model()` ✓
- `load_graph()` ✓
- `create_graph()` ✓
- `load_dataset()` ✓
- `get_graph_stats()` ✓
- `evaluate()` ✓

**Total LOC: ~400-500**
**Development time: ~2 weeks**
**Users can:** Load graphs, compute GED, evaluate on benchmarks

---

## Feature Expansion Path

```
v0.1 (MVP)              → v0.2 (Production) → v1.0 (Research)
├─ MUST HAVE            ├─ MUST HAVE        ├─ Everything
├─ Basic usage          ├─ GOOD HAVE        ├─ Baselines
└─ Works!               ├─ Training         ├─ Advanced analysis
                        ├─ Fine-tuning      ├─ Publication-ready
                        └─ Better docs      └─ Enterprise features
```

---

## Success Metrics by Tier

### MUST HAVE
- ✓ Code runs without errors
- ✓ Can compute GED between any two graphs
- ✓ Results reasonable vs paper numbers
- ✓ Clear API, minimal friction

### GOOD TO HAVE
- ✓ Users can train on custom data
- ✓ Can interpret model decisions
- ✓ Visualization working
- ✓ Setup complexity reduced

### PRO LEVEL
- ✓ Production deployment ready
- ✓ Research reproducibility guaranteed
- ✓ Comparison with baselines published
- ✓ Full paper algorithms implemented

---

This tiered approach lets you ship value quickly (v0.1 in 2 weeks), then progressively enhance.
