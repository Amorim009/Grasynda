# Grasynda Visibility Graph - Research Documentation

## 🎯 Concept Overview

**GrasyndaVisibilityGraph** is a novel variant of Grasynda that uses **visibility graphs** instead of quantile-based graphs to capture temporal structure and generate synthetic time series.

---

## 📊 Comparison: Quantile Graph vs Visibility Graph

### **Original Grasynda (Quantile Graph)**
1. Decompose → **Quantize values** into bins
2. Each time point assigned to a quantile based on **remainder value**
3. Learn transitions between **value-based quantiles**
4. Generate by walking quantile states
5. Sample from **values in that quantile**

### **New Grasynda (Visibility Graph)**
1. Decompose → **Build visibility graph** from remainder
2. Each time point assigned to a state based on **graph features** (degree, clustering)
3. Learn transitions between **graph-feature states**
4. Generate by walking graph-feature states
5. Sample from **values with similar graph properties**

---

## 🔍 What is a Visibility Graph?

A visibility graph converts a time series into a graph where:
- **Nodes** = time points
- **Edges** = "visibility" between points

### **Two Types Implemented:**

#### **1. Natural Visibility**
Points `(t_a, y_a)` and `(t_b, y_b)` are connected if no intermediate point blocks the line of sight.

**Mathematical condition:** All intermediate points `(t_c, y_c)` must satisfy:
```
y_c < y_a + (y_b - y_a) × (t_c - t_a) / (t_b - t_a)
```

**Properties:**
- Captures peaks, valleys, and structure
- More edges (denser graph)
- More computationally expensive
- Better for complex patterns

#### **2. Horizontal Visibility**
Points are connected if all intermediate values are **below** the minimum of the two points.

**Mathematical condition:**
```
y_c < min(y_a, y_b) for all c between a and b
```

**Properties:**
- Simpler and faster
- Fewer edges (sparser graph)
- Highlights local extrema
- Good for periodicity detection

---

## 🛠️ How GrasyndaVisibilityGraph Works

### **Step-by-Step Process:**

#### **Step 1: Decomposition** (Same as original)
```python
STL decomposition: y(t) = Trend(t) + Seasonal(t) + Remainder(t)
```

#### **Step 2: Build Visibility Graph**
```python
# For each time series remainder:
adj_matrix = VisibilityGraph.horizontal_visibility(remainder)
# or
adj_matrix = VisibilityGraph.natural_visibility(remainder)
```

#### **Step 3: Extract Node Features**
```python
features = {
    'degree': [number of connections for each node],
    'clustering': [local clustering coefficient],
    'betweenness': [centrality measure]
}
```

#### **Step 4: Quantize Based on Features**
Instead of quantizing remainder values, quantize **node degree** (or other feature):
```python
quantiles = pd.qcut(node_degrees, n_quantiles=25, labels=False)
```

#### **Step 5: Build Transition Matrix**
Same as original - learn transitions between degree-based states.

#### **Step 6: Generate Synthetic Series**
1. Walk through degree states using transition probabilities
2. For each state, sample from **original remainder values** that had similar degree
3. Reconstruct: Trend + Seasonal + Sampled Remainder

---

## 💡 Why This Might Work Better

### **Advantages of Visibility Graphs:**

1. **Structural Pattern Capture**
   - Quantiles only care about value ranges
   - Visibility graphs capture **shape and structure**

2. **Multi-Scale Information**
   - Node degree reflects local connectivity
   - Graph properties capture global patterns
   - Preserves both local and global dynamics

3. **Robustness to Outliers**
   - Graph topology less sensitive to extreme values
   - Degree-based states more stable than value-based quantiles

4. **Periodicity Detection**
   - Visibility graphs naturally identify periodic patterns
   - Horizontal visibility excels at finding cycles

5. **Temporal Dependencies**
   - Graph structure encodes temporal relationships
   - Better preservation of autocorrelation structure

---

## 🧪 Usage Examples

### **Basic Usage:**

```python
from src.grasynda_visibility import GrasyndaVisibilityGraph

# Horizontal visibility (faster, good for periodicity)
grasynda_vis = GrasyndaVisibilityGraph(
    n_quantiles=25,
    quantile_on='remainder',
    period=12,  # monthly data
    ensemble_transitions=False,
    visibility_type='horizontal',
    feature_type='degree'
)

# Generate synthetic data
synthetic_data = grasynda_vis.transform(train_df)

# Check graph statistics
stats = grasynda_vis.get_graph_statistics('series_id_1')
print(stats)
```

### **Natural Visibility (More Complex):**

```python
# Natural visibility (more edges, captures complex patterns)
grasynda_nat = GrasyndaVisibilityGraph(
    n_quantiles=25,
    quantile_on='remainder',
    period=4,  # quarterly data
    ensemble_transitions=False,
    visibility_type='natural',
    feature_type='degree'
)

synthetic_data = grasynda_nat.transform(train_df)
```

### **Using Different Features:**

```python
# Use clustering coefficient instead of degree
grasynda_cluster = GrasyndaVisibilityGraph(
    n_quantiles=25,
    quantile_on='remainder',
    period=12,
    ensemble_transitions=False,
    visibility_type='horizontal',
    feature_type='clustering'  # Different feature!
)
```

---

## 🧮 Parameter Guide

| Parameter | Options | Recommendation |
|-----------|---------|----------------|
| `visibility_type` | `'horizontal'`, `'natural'` | Start with `'horizontal'` (faster) |
| `feature_type` | `'degree'`, `'clustering'` | `'degree'` is most robust |
| `n_quantiles` | 10-50 | 25 is a good default |
| `quantile_on` | `'remainder'`, `'trend'` | `'remainder'` (standard) |

---

## 📈 Expected Performance

### **When Visibility Graph May Outperform:**
- ✅ Data with strong **structural patterns**
- ✅ **Periodic** time series (especially with horizontal visibility)
- ✅ Series with **local extrema** (peaks and valleys)
- ✅ Data where **shape matters** more than exact values
- ✅ **Chaotic** or complex dynamics

### **When Quantile Graph May Be Better:**
- ⚠️ Very **smooth** time series
- ⚠️ Data where **value distribution** is most important
- ⚠️ **Very short** time series (< 50 points)
- ⚠️ Extremely **noisy** data

---

## 🚀 Next Steps for Research

### **1. Run Initial Experiment**
```bash
python test_visibility_graph.py
```

This will compare:
- Grasynda Uniform (quantile-based)
- Grasynda Visibility Natural
- Grasynda Visibility Horizontal

### **2. Full Dataset Comparison**
Modify the test script to run on all 6 datasets and compare MASE scores.

### **3. Feature Engineering Extensions**
Try other graph features:
- **Betweenness centrality** (for longer series)
- **PageRank**
- **Community detection** labels
- **Graph motifs**

### **4. Hybrid Approaches**
Combine quantile and visibility:
```python
# State = (value_quantile, degree_quantile)
# 2D transition matrix
```

### **5. Publications**
If results are promising, this is **novel research** worthy of publication!

---

## 📚 References

**Visibility Graphs in Time Series:**
- Lacasa et al. (2008): "From time series to complex networks: The visibility graph"
- Luque et al. (2009): "Horizontal visibility graphs: Exact results for random time series"

**Applications:**
- Financial time series analysis
- Climate data pattern recognition
- Biomedical signal processing

---

## ⚡ Quick Start

1. **Install dependencies** (already in your environment)
2. **Run test:** `python test_visibility_graph.py`
3. **Compare results** with quantile-based Grasynda
4. **Iterate** on parameters and features

Good luck with your research! 🎓
