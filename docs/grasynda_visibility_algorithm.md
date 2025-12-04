# Grasynda Visibility Graph: The Algorithm

This document details the **Degree Matching** variant of Grasynda Visibility, which was identified as the most effective graph-based augmentation method.

## 🎯 Core Concept
Instead of learning the distribution of *values* (like standard Grasynda), this method learns the distribution of **structural connectivity**. It assumes that if we preserve the "visibility structure" of a time series, we preserve its essential temporal dynamics.

---

## ⚙️ Step-by-Step Process

### **Step 1: Decomposition (Optional but Recommended)**
We first isolate the stochastic component of the time series to build a cleaner graph.
- **Input:** Raw Time Series $y_t$
- **Action:** Apply STL Decomposition
- **Output:** $y_t = \text{Trend}_t + \text{Seasonal}_t + \text{Remainder}_t$
- **Focus:** We only manipulate the **Remainder**. Trend and Seasonality are kept aside and added back later.

### **Step 2: Visibility Graph Construction**
We convert the time series (Remainder) into a graph where each time point is a node.
- **Method:** Horizontal Visibility Graph (HVG)
- **Rule:** Two nodes $i$ and $j$ are connected if all intermediate points $k$ are smaller than both $y_i$ and $y_j$.
  $$y_k < \min(y_i, y_j) \quad \forall k \in (i, j)$$
- **Result:** An Adjacency Matrix $A$ representing the connectivity structure.

### **Step 3: Pattern Extraction (Learning)**
We extract two key pieces of information from the graph:

1.  **Degree Sequence ($D$):**
    - We calculate the degree $k_i$ (number of connections) for every node $i$.
    - This forms a distribution $P(k)$ representing the structural complexity of the series.
    
2.  **Value-Degree Map ($M$):**
    - We map every degree $k$ to the list of actual values that have that degree.
    - $M(k) = \{ y_i \mid \text{degree}(i) = k \}$
    - *Example:* "Nodes with 5 connections usually have values around 0.8 to 1.2".

### **Step 4: Generation (The "Degree Matching" Process)**
To generate a new synthetic time series:

1.  **Sample Structure:**
    - We randomly sample a new sequence of degrees $D_{new}$ from the original degree distribution $P(k)$.
    - *Note:* We sample with replacement to create variations, but the overall statistical distribution of connectivity remains identical to the original.

2.  **Sample Values:**
    - For each target degree $k_{target}$ in our new sequence:
        - We look up the possible values in our map $M(k_{target})$.
        - We randomly pick one value from this set.
    - *Logic:* "I need a point with connectivity 5, so I'll pick a value that historically produced connectivity 5."

### **Step 5: Reconstruction**
- **Input:** Generated Remainder $R_{new}$
- **Action:** Add back the original Trend and Seasonality.
- **Output:** $y_{new} = \text{Trend}_t + \text{Seasonal}_t + R_{new}$

---

## 💡 Why It Works
- **Preserves Topology:** By sampling from the degree distribution, we ensure the new series has the same "roughness" and "inter-dependency" structure as the original.
- **Non-Linear:** Unlike simple statistical sampling, this captures non-linear relationships encoded in the visibility structure.
- **Data-Driven:** No assumptions about the underlying distribution (Gaussian, etc.) are made.

## 🚀 Summary
1. **Time Series** $\rightarrow$ **Graph**
2. **Graph** $\rightarrow$ **Degree Distribution**
3. **Degree Distribution** $\rightarrow$ **New Degrees**
4. **New Degrees** $\rightarrow$ **New Values**
