# Pure Visibility Graph Approach - Quick Guide

## 🎯 What Changed

**OLD APPROACH** (You didn't want this):
- Still used quantiles
- Just quantized graph features (degree) instead of values
- Was essentially "quantile-graph with extra steps"

**NEW APPROACH** (Pure visibility graph):
- **NO QUANTILES AT ALL** ✅
- Directly uses visibility graph properties for generation
- Three generation methods available

---

## 🚀 How It Works Now

### **Step 1: Decompose**
STL decomposition: `y(t) = Trend(t) + Seasonal(t) + Remainder(t)`

### **Step 2: Build Visibility Graph**
From remainder, construct adjacency matrix based on visibility rules

### **Step 3: Extract Graph Properties**
- **Degree sequence**: how connected each time point is
- **Degree-to-value map**: which values have which degrees
- **Motifs**: local graph patterns (optional)

### **Step 4: Generate New Remainder** (3 Methods)

#### **Method 1: Degree Matching** (Recommended)
```python
1. Sample degrees from original degree distribution  
2. For each degree, sample a value that had that degree originally
3. This preserves graph-theoretic properties
```

#### **Method 2: Motif Sampling**
```python
1. Extract local graph motifs (3-node subgraphs)
2. Chain together motifs to build new series
3. Preserves local structure
```

#### **Method 3: Hybrid**
```python
1. Start with degree matching
2. Apply light smoothing to maintain temporal continuity
3. Balance between structure and smoothness
```

### **Step 5: Reconstruct**
`Synthetic y(t) = Trend(t) + Seasonal(t) + Generated Remainder(t)`

---

## 📝 Usage

```python
from src.grasynda_visibility import GrasyndaVisibilityGraph

# Create with degree matching (best for preserving graph properties)
vg = GrasyndaVisibilityGraph(
    period=12,  # seasonal period
    visibility_type='horizontal',  # or 'natural'
    generation_method='degree_matching'  # or 'motif_sampling' or 'hybrid'
)

# Generate synthetic data
synthetic = vg.transform(train_data)

# Augment training set
train_augmented = pd.concat([train_data, synthetic])
```

---

## 🔬 Three Generation Methods Explained

### **Degree Matching** 
- **Idea**: Time points with similar connectivity should have similar values
- **Process**: Sample degree → find values with that degree → pick one randomly
- **Pros**: Preserves degree distribution exactly
- **Cons**: May lose some temporal smoothness  
- **Best for**: When graph structure is most important

### **Motif Sampling**
- **Idea**: Preserve local graph patterns
- **Process**: Extract 3-node motifs → chain them together
- **Pros**: Maintains local structure
- **Cons**: Slower, can be repetitive
- **Best for**: When local patterns matter

### **Hybrid**
- **Idea**: Balance structure and smoothness
- **Process**: Degree matching + smoothing filter
- **Pros**: Good compromise
- **Cons**: Slightly modifies degree distribution
- **Best for**: General use when you want robustness

---

## 🧪 Test Script

Run this to compare methods:
```bash
python test_visibility_graph.py
```

This compares:
1. **Original** (no augmentation)
2. **Grasynda Uniform** (quantile-based, baseline)
3. **Grasynda Vis Degree** (pure visibility, degree matching)
4. **Grasynda Vis Hybrid** (visibility + smoothing)
5. **SeasonalNaive** (benchmark)

---

## 🎓 Why This Is Novel Research

1. **No quantization** - First Grasynda variant without discrete states
2. **Graph-theoretic properties** - Preserves degree distribution, not value distribution
3. **Structural generation** - Values chosen based on connectivity, not position
4. **Multiple strategies** - Degree, motif, and hybrid approaches

This is fundamentally different from traditional augmentation!

---

## 📊 What to Expect

**If visibility graphs work well:**
- Should excel on data with structural patterns
- May outperform on periodic/seasonal data
- Better preservation of autocorrelation structure

**Metrics to watch:**
- MASE (forecasting accuracy)
- Degree distribution similarity
- Autocorrelation preservation
- Graph topology metrics

Good luck! 🚀
