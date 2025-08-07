# SYNCCIN 2025 Summer School - EEG Connectivity Analysis

Interactive Jupyter notebooks for analyzing EEG connectivity using different methodological approaches. Materials designed for educational purposes and hands-on learning of connectivity analysis techniques.

## Available Notebooks

### 1. Cross-Correlation Analysis
**File**: `3_channels_correlations.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/3_channels_correlations.ipynb)

Basic connectivity analysis using cross-correlation and delay estimation.

### 2. MVAR and DTF Analysis  
**File**: `3_channels_MVAR.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/3_channels_MVAR.ipynb)

Advanced connectivity analysis using Multivariate Autoregressive modeling and Directed Transfer Function.

### 3. Bivariate vs. Multivariate DTF Comparison
**File**: `5_channels_bivariate_multivariate_MVAR.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/5_channels_bivariate_multivariate_MVAR.ipynb)

Demonstrates differences between bivariate and multivariate connectivity approaches using 5-channel simulation.

### 4. Frequency-Selective DTF Analysis
**File**: `7_channels_bivariate_multivariate_MVAR.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/7_channels_bivariate_multivariate_MVAR.ipynb)

Frequency selectivity of DTF using 7-channel simulation with alpha rhythm transmission.

### 5. EEG Reference Effects on DTF
**File**: `DTF_EEG_reference.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/DTF_EEG_reference.ipynb)

Real EEG data analysis comparing linked ears vs. common average reference effects on DTF.

### 6. Hyperscanning EEG-HRV Connectivity Analysis
**File**: `Warsaw_pilot_EEG_hyperscanning_analysis.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jzygierewicz/SYNCCIN_2025_summer_school/blob/main/Warsaw_pilot_EEG_hyperscanning_analysis.ipynb)

Real hyperscanning data from child-caregiver dyads with DTF analysis of HRV coupling, EEG connectivity, and multimodal EEG-HRV interactions.

## Repository Contents

**Interactive Notebooks**
- `3_channels_correlations.ipynb` - Cross-correlation and delay analysis
- `3_channels_MVAR.ipynb` - MVAR modeling, spectral analysis, and DTF computation
- `5_channels_bivariate_multivariate_MVAR.ipynb` - Bivariate vs. multivariate connectivity comparison
- `7_channels_bivariate_multivariate_MVAR.ipynb` - Frequency-selective DTF analysis
- `DTF_EEG_reference.ipynb` - Real EEG data analysis with reference electrode comparison
- `Warsaw_pilot_EEG_hyperscanning_analysis.ipynb` - Hyperscanning EEG-HRV connectivity analysis

**Data Files**
- `simulated_3_channels.joblib` - 3-channel EEG simulation with known connectivity
- `simulated_7_channels.joblib` - 7-channel EEG simulation with alpha rhythm transmission
- `EEG_alpha.joblib` - Real EEG data with alpha rhythm

**Python Modules**
- `mtmvar.py` - MVAR analysis functions
- `utils.py` - Utility functions for data handling and visualization
- `warsaw_pilot_data.py` - Hyperscanning analysis pipeline
- `prepare_3channels_simulation.py` - Generate 3-channel simulation data
- `prepare_7channels_simulation.py` - Generate 7-channel simulation data

**Documentation & Visualizations**
- `README.md` - This guide
- `3chan_sim.png`, `5chan_sim.png`, `7chan_sim.png` - Connectivity scheme visualizations

## Usage Instructions

1. Click the "Open in Colab" button for any notebook
2. Run the setup cells to install packages and download data
3. Execute cells sequentially and explore the interactive visualizations
4. Work through notebooks in order: correlations → MVAR → comparisons → real data

## Technical Requirements

- No local installation needed (runs entirely in Google Colab)
- Cloud-based data access

---

## Contact & Support

**Repository**: [SYNCCIN_2025_summer_school](https://github.com/jzygierewicz/SYNCCIN_2025_summer_school)  
**Author**: Jarosław Żygierewicz  
**Institution**: University of Warsaw  
**Event**: SYNCCIN 2025 Summer School

**Citation:**
```
Zygierewicz, J. (2025). EEG Connectivity Analysis Notebooks. 
SYNCCIN 2025 Summer School. University of Warsaw.
https://github.com/jzygierewicz/SYNCCIN_2025_summer_school
```
