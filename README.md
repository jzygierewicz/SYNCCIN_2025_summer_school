# SYNCCIN 2025 Summer School - EEG Connectivity Analysis

## 3-Channel Correlation Analysis Notebook

This repository contains an interactive Jupyter notebook for analyzing EEG connectivity using cross-correlation and delay estimation between multiple channels.

### 🚀 **Open in Google Colab**

Click the button below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/3_channels_correlations.ipynb)

### 📋 **What's Included**

- **Interactive Notebook**: `3_channels_correlations.ipynb` with zoomable Plotly visualizations
- **Data Files**: 
  - `simulated_3_channels.joblib` - Simulated 3-channel EEG data
  - `EEG_alpha.joblib` - Real EEG data with alpha rhythm
- **Python Modules**:
  - `mtmvar.py` - MVAR analysis functions
  - `utils.py` - Utility functions
- **Visualizations**: `3chan_sim.png` - Example output

### 🎯 **Features**

- ✅ **Automatic Setup**: Downloads all required files from GitHub
- ✅ **Interactive Plots**: Zoomable and pannable visualizations optimized for Colab
- ✅ **Cross-Correlation Analysis**: Estimate coupling strength between channels
- ✅ **Delay Estimation**: Find optimal time lags for maximum correlation
- ✅ **Network Visualization**: Interactive graph representation of connectivity
- ✅ **Self-Contained**: No manual file uploads needed

### 📚 **Analysis Workflow**

1. **Signal Visualization**: Plot multi-channel EEG signals
2. **Cross-Correlation**: Calculate correlation functions between channel pairs
3. **Delay Analysis**: Determine time delays for maximum correlation
4. **Connectivity Matrix**: Visualize correlation strengths and delays
5. **Network Graph**: Show connectivity as an interactive directed graph

### 🛠 **Usage Instructions**

1. Click the "Open in Colab" button above
2. Run the installation cell to install required packages
3. Execute the download cell to fetch all data and code files
4. Follow the notebook cells sequentially for the complete analysis

### 📖 **Educational Content**

This notebook is designed for the SYNCCIN 2025 Summer School and includes:
- Comprehensive documentation of each analysis step
- Interactive visualizations for better understanding
- Detailed explanations of connectivity methods
- Discussion of method limitations and alternatives

### 🔬 **Analysis Methods**

- **Cross-Correlation**: Bivariate approach to connectivity estimation
- **Time Delay Analysis**: Lag estimation for maximum coupling
- **Network Analysis**: Graph-based representation of connectivity patterns

### ⚠️ **Method Limitations**

The cross-correlation approach demonstrated here:
- Cannot distinguish between direct and indirect connections
- Doesn't account for multivariate effects from other channels
- Primarily captures linear relationships

For more advanced analysis, consider:
- Partial correlation analysis
- Multivariate autoregressive (MVAR) modeling
- Granger causality analysis
- Directed transfer function (DTF) or partial directed coherence (PDC)

---

**Repository**: [SYNCCIN_2025_summer_school](https://github.com/jzygierewicz/SYNCCIN_2025_summer_school)  
**Author**: Jarek Zygierewicz  
**Institution**: University of Warsaw  
**Event**: SYNCCIN 2025 Summer School
