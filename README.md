# SYNCCIN 2025 Summer School - EEG Connectivity Analysis

This repository contains interactive Jupyter notebooks for analyzing EEG connectivity using different methodological approaches. The materials are designed for educational purposes and hands-on learning of connectivity analysis techniques.

## 📓 **Available Notebooks**

### 1. Cross-Correlation Analysis
**File**: `3_channels_correlations.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/3_channels_correlations.ipynb)

**Focus**: Basic connectivity analysis using cross-correlation and delay estimation

### 2. MVAR and DTF Analysis  
**File**: `3_channels_MVAR.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/3_channels_MVAR.ipynb)

**Focus**: Advanced connectivity analysis using Multivariate Autoregressive modeling and Directed Transfer Function

### 3. Bivariate vs. Multivariate DTF Comparison
**File**: `5_channels_bivariate_multivariate_MVAR.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/5_channels_bivariate_multivariate_MVAR.ipynb)

**Focus**: Demonstrates differences between bivariate and multivariate connectivity approaches using 5-channel simulation

### 4. Frequency-Selective DTF Analysis
**File**: `7_channels_bivariate_multivariate_MVAR.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/7_channels_bivariate_multivariate_MVAR.ipynb)

**Focus**: Frequency selectivity of DTF using 7-channel simulation with alpha rhythm transmission

### 5. EEG Reference Effects on DTF
**File**: `DTF_EEG_reference.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/DTF_EEG_reference.ipynb)

**Focus**: Real EEG data analysis comparing linked ears vs. common average reference effects on DTF

### 📋 **Repository Contents**

#### **Interactive Notebooks**
- **`3_channels_correlations.ipynb`** - Cross-correlation and delay analysis with interactive visualizations
- **`3_channels_MVAR.ipynb`** - MVAR modeling, spectral analysis, and DTF computation
- **`5_channels_bivariate_multivariate_MVAR.ipynb`** - Comparison of bivariate vs. multivariate connectivity approaches
- **`7_channels_bivariate_multivariate_MVAR.ipynb`** - Frequency-selective DTF analysis with alpha rhythm simulation
- **`DTF_EEG_reference.ipynb`** - Real EEG data analysis with reference electrode comparison

#### **Data Files**
- **`simulated_3_channels.joblib`** - Simulated 3-channel EEG data with known connectivity structure
- **`simulated_7_channels.joblib`** - Simulated 7-channel EEG data with alpha rhythm transmission
- **`EEG_alpha.joblib`** - Real EEG data with alpha rhythm for validation

#### **Python Modules**
- **`mtmvar.py`** - MVAR analysis functions (model estimation, transfer functions, visualization)
- **`utils.py`** - Utility functions for data handling and exploration
- **`prepare_3channels_simulation.py`** - Script to generate 3-channel simulation data
- **`prepare_7channels_simulation.py`** - Script to generate 7-channel simulation data

#### **Documentation & Visualizations**
- **`README.md`** - This comprehensive guide
- **`3chan_sim.png`** - 3-channel connectivity scheme visualization
- **`5chan_sim.png`** - 5-channel connectivity scheme visualization  
- **`7chan_sim.png`** - 7-channel connectivity scheme visualization


## 📚 **Learning Objectives & Analysis Workflows**

### Notebook 1: Cross-Correlation Analysis
**Learning Goals:**
- Understand basic connectivity concepts
- Learn cross-correlation computation and interpretation
- Explore time delay estimation methods
- Visualize connectivity patterns using network graphs

**Analysis Steps:**
1. **Signal Visualization** - Plot and explore multi-channel EEG signals
2. **Cross-Correlation Computation** - Calculate correlation functions between channel pairs
3. **Delay Analysis** - Determine optimal time lags for maximum correlation
4. **Connectivity Matrices** - Visualize correlation strengths and delays
5. **Network Graphs** - Interactive directed graph representation of connectivity

### Notebook 2: MVAR and DTF Analysis
**Learning Goals:**
- Learn MVAR modeling techniques
- Understand bivariate vs. multivariate approaches
- Learn Directed Transfer Function computation and interpretation
- Compare spectral and connectivity estimates between methods

**Analysis Steps:**
1. **MVAR Model Estimation** - Optimal model order selection using information criteria
2. **Spectral Analysis** - Compute auto-spectra and cross-spectra (bivariate vs. multivariate)
3. **DTF Computation** - Calculate Directed Transfer Function for connectivity analysis
4. **Comparative Visualization** - Interactive plots comparing bivariate and multivariate results
5. **Network Analysis** - Graph-based representation of directional connectivity patterns
6. **Statistical Interpretation** - Summary statistics and connectivity strength analysis

### 🛠 **Usage Instructions**

#### **Getting Started:**
1. **Work through both notebooks sequentially**

2. **Launch in Google Colab**: Click the respective "Open in Colab" button above

3. **Run the setup**: Execute the installation and download cells to automatically fetch all required files

4. **Follow the workflow**: Execute cells sequentially and explore the interactive visualizations

#### **Recommended Learning Path:**
1. **Start with correlations notebook** to understand basic connectivity concepts
2. **Explore the interactive visualizations** to gain intuition about connectivity patterns  
3. **Proceed to MVAR notebook** for advanced techniques and methodological comparisons
4. **Compare results** between different approaches to understand their strengths and limitations

## 🔬 **Methodological Coverage**

### **Basic Methods (Correlations Notebook)**
- **Cross-Correlation Analysis**: Bivariate approach to connectivity estimation
- **Time Delay Estimation**: Lag-based coupling analysis
- **Network Visualization**: Graph-based representation of connectivity patterns

### **Advanced Methods (MVAR Notebook)**  
- **Multivariate Autoregressive (MVAR) Modeling**: System-level connectivity analysis
- **Model Order Selection**: Information criteria for optimal model complexity
- **Spectral Analysis**: Auto-spectra and cross-spectra estimation
- **Directed Transfer Function (DTF)**: Frequency-domain directional connectivity
- **Bivariate vs. Multivariate Comparison**: Methodological strengths and limitations

## 📖 **Educational Content & Theory**

Both notebooks include comprehensive educational materials:
- **Theoretical Background**: Mathematical foundations and conceptual explanations
- **Method Comparisons**: Detailed analysis of different approaches
- **Interpretation Guidelines**: How to read and understand results
- **Best Practices**: Recommendations for real-world applications
- **Limitations Discussion**: When and why methods may fail
- **Further Reading**: References for advanced topics

## ⚠️ **Methodological Considerations**

### **Cross-Correlation Approach Limitations:**
- Cannot distinguish between direct and indirect connections
- Doesn't account for multivariate effects from other channels  
- Primarily captures linear relationships
- May show spurious connections due to common sources

### **MVAR/DTF Approach Advantages:**
- **Multivariate perspective**: Accounts for all channels simultaneously
- **Direct connectivity**: Better at revealing true direct connections
- **Frequency-specific analysis**: Reveals connectivity patterns across frequency bands
- **Directional information**: Quantifies the direction of information flow

### **When to Use Each Method:**
- **Cross-correlation**: Good for initial exploration and understanding basic coupling patterns
- **MVAR/DTF**: Preferred for rigorous connectivity analysis and publication-quality results
- **Combined approach**: Use both methods for comprehensive analysis and validation

## 🎓 **For Educators & Students**

### **Course Integration:**
- **Standalone workshops**: Each notebook can be used independently
- **Progressive curriculum**: Start with correlations, advance to MVAR
- **Hands-on learning**: Interactive exercises with real data
- **Assessment ready**: Clear learning objectives and outcomes

### **Technical Requirements:**
- **No local installation needed**: Runs entirely in Google Colab
- **Automatic dependency management**: All packages installed automatically
- **Cross-platform compatibility**: Works on any device with internet access
- **Cloud-based data**: All files downloaded automatically from GitHub

## 🚀 **Advanced Topics & Extensions**

The notebooks provide a foundation for exploring more advanced topics:

### **Next Steps in Connectivity Analysis:**
- **Partial Directed Coherence (PDC)**: Alternative to DTF with different normalization
- **Granger Causality**: Statistical approach to directional connectivity  
- **Time-varying connectivity**: Dynamic connectivity analysis
- **Nonlinear methods**: Mutual information, transfer entropy
- **Statistical significance**: Surrogate data methods and permutation testing
- **High-dimensional analysis**: Connectivity analysis for large channel arrays

### **Real-World Applications:**
- **Clinical EEG analysis**: Epilepsy, sleep studies, cognitive neuroscience
- **MEG connectivity**: Source-level connectivity analysis
- **fMRI effective connectivity**: BOLD signal connectivity patterns
- **Economic time series**: Financial market interdependencies  
- **Climate data analysis**: Atmospheric and oceanic connectivity patterns

---

## **Contact & Support**

**Repository**: [SYNCCIN_2025_summer_school](https://github.com/jzygierewicz/SYNCCIN_2025_summer_school)  
**Author**: Jarosław Żygierewicz  
**Institution**: University of Warsaw  
**Event**: SYNCCIN 2025 Summer School

### **Quick Access Links:**
- 🔗 **Correlations Notebook**: [Direct Colab Link](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/3_channels_correlations.ipynb)
- 🔗 **MVAR Notebook**: [Direct Colab Link](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/3_channels_MVAR.ipynb)
- � **5-Channel Comparison**: [Direct Colab Link](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/5_channels_bivariate_multivariate_MVAR.ipynb)
- 🔗 **7-Channel Frequency Analysis**: [Direct Colab Link](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/7_channels_bivariate_multivariate_MVAR.ipynb)
- 🔗 **EEG Reference Analysis**: [Direct Colab Link](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/DTF_EEG_reference.ipynb)
- �📁 **Repository**: [GitHub Repository](https://github.com/jzygierewicz/SYNCCIN_2025_summer_school)

### **Citation:**
If you use these materials in your research or teaching, please cite:
```
Zygierewicz, J. (2025). EEG Connectivity Analysis Notebooks. 
SYNCCIN 2025 Summer School. University of Warsaw.
https://github.com/jzygierewicz/SYNCCIN_2025_summer_school
```

**Last Updated**: August 2025  
**License**: Open Educational Resources
