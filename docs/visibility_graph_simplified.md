# Grasynda Visibility Graph - Simplified Implementation

## Overview
The Grasynda Visibility Graph has been simplified to focus on **degree matching only**, removing unnecessary complexity from motif sampling and hybrid methods.

## Simplifications Made

### 1. Removed Components
- ❌ **Motif extraction** (`extract_motifs` method)
- ❌ **Motif sampling** generation (`_generate_by_motif_sampling`)
- ❌ **Hybrid generation** (`_generate_hybrid`)
- ❌ **Motif storage** (`self.motif_patterns`)
- ❌ **Motif parameters** (`motif_window`, `generation_method`)

### 2. Current Implementation
The simplified implementation uses **only degree matching** with two visibility graph types:

#### Visibility Graph Types
1. **Natural Visibility Graph**
   - Points connect if the line between them is not blocked
   - Formula: `y_k < y_i + (y_j - y_i) * (t_k - t_i) / (t_j - t_i)`

2. **Horizontal Visibility Graph**  
   - Points connect if all intermediate points are below both
   - Formula: `y_k < min(y_i, y_j)`

#### Generation Method: Degree Matching

**Algorithm:**
1. Build visibility graph from time series
2. Extract degree sequence (number of connections per node)
3. Create value-degree mapping (which values had which degrees)
4. Generate new series:
   - Sample degrees from original distribution
   - For each degree, pick a value that originally had that degree
5. Preserve graph topology while creating new data

## Usage

```python
from src.grasynda_visibility import GrasyndaVisibilityGraph

# With STL Decomposition (Recommended)
gen = GrasyndaVisibilityGraph(
    period=12,
    visibility_type='horizontal',  # or 'natural'
    use_decomposition=True
)

# Without STL Decomposition (Raw data)
gen = GrasyndaVisibilityGraph(
    period=12,
    visibility_type='horizontal',
    use_decomposition=False  # Works on raw y values
)

# Generate synthetic data
synthetic_df = gen.transform(train_df)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | int | None | Seasonal period for STL (required if `use_decomposition=True`) |
| `visibility_type` | str | 'horizontal' | Type of visibility graph: 'natural' or 'horizontal' |
| `use_decomposition` | bool | True | Use STL decomposition or work on raw data |
| `robust` | bool | False | Use robust STL decomposition |

## Key Findings from TSTR Experiments

| Method | MASE | Notes |
|--------|------|-------|
| Vis + Decomposition | 1.137 | ✅ Works well |
| Vis WITHOUT Decomposition | 2.064 | ❌ Pure noise |
| Grasynda Uniform | 1.138 | Comparable to visibility |

**Conclusion:** STL decomposition is **essential** for visibility graphs to produce useful synthetic data.

## File Structure

```
src/grasynda_visibility.py
├── VisibilityGraph (Helper class)
│   ├── natural_visibility()
│   ├── horizontal_visibility()
│   └── extract_degree_sequence()
└── GrasyndaVisibilityGraph (Main class)
    ├── __init__()
    ├── transform()
    ├── _decompose()
    ├── _learn_visibility_patterns()
    ├── _generate_synthetic_series()
    ├── _generate_by_degree_matching()
    ├── _reconstruct_decomposed()
    └──_reconstruct_raw()
```

## Clean, Simple, and Effective! 🎯
