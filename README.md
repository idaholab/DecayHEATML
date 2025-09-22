# DecayHEATML
This code uses machine learning to predict decay heat from spent nuclear fuel in molten salt reactors after shutdown. By accounting for operational uncertainties, it supports safer fuel removal, storage, and reactor design, improving nuclear energy safety and efficiency.

## Introduction

This repository contains the supporting materials for the paper titled "A Methodology for Decay Heat Characterization in Molten Salt Reactors." The paper presents a hybrid machine learning and segmented polynomial methodology for predicting decay heat in molten salt reactors (MSRs), addressing the dual challenges of complex operational uncertainties and the need for interpretable models compatible with engineering workflows.

## Abstract

Accurate decay heat prediction in MSRs faces dual challenges: complex operational uncertainties and the need for interpretable models compatible with engineering workflows. This work presents a hybrid machine learning and segmented polynomial methodology that addresses both requirements through three key innovations. First, a modular data architecture encodes MSR-specific operational parameters (power density: 1-100 W/cm³, humidity: 0-0.1 wt%, air ingress: 0-0.1 mol%) with uncertainty-aware temporal discretization spanning 15 orders of magnitude. Second, region-optimized machine learning models achieve 92.3% root mean square error (RMSE) reduction over conventional polynomials while maintaining physical interpretability through automated piecewise equation generation. Third, dual front-end interfaces accelerate safety analyses -- a Jupyter environment enables researchers to explore 10,000+ parameter combinations via interactive widgets, while a Streamlit web application reduces design iteration cycles through production-grade visualization tools. Operational deployment demonstrates prediction times of only a couple hundred milliseconds for 10,000-year decay profiles, enabling real-time optimization of spent fuel container designs.


## Installation

To run the code in this repository, you need to have Python 3.8 or higher installed. We suggest creating a virtual environment. You can install the necessary dependencies using:

```bash
pip install '.'
```
In addition, for the Jupyter Notebook or Webapp, install the dependencies using:
```bash
pip install '.[Interactive]'
```
or
```bash
pip install '.[webapp]'
```
respectively.

## Usage

### Jupyter Notebook Interface

For computational researchers, the Jupyter notebook interface (`notebooks/decay_heat_analysis.ipynb`) provides interactive controls to explore decay heat predictions.

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook examples/interactive_plot.ipynb
   ```
2. Adjust the parameters using the interactive widgets and visualize the decay heat predictions.

### Streamlit Web Application

For engineering teams and regulatory reviews, the Streamlit-based web application (`web_app/app.py`) offers a production-grade interface.

1. Run the Streamlit application:
   ```bash
   streamlit run webapp/DecayHeatML.py
   ```
2. Access the web interface through your browser and interact with the decay heat prediction tools.

## Methodology

### Data Architecture
The data architecture encodes MSR-specific operational parameters and utilizes uncertainty-aware temporal discretization. The dataset schema includes features such as power density, humidity content, air ingress, and log-transformed decay heat.

### Machine Learning Framework
The machine learning framework employs a heterogeneous ensemble approach, combining linear models, tree-based methods, multi-layer perceptrons, and symbolic regression. The framework includes region-optimized training to capture distinct decay heat characteristics across short-term and long-term decay phases.

### Segmented Polynomial Interpretation (SPI)
The SPI method decomposes the temporal domain into regions with distinct polynomial representations, providing interpretable equations for safety analyses and maintaining accuracy across multiple decay heat regimes.

### Applications
The methodology supports real-time evaluation of decay heat scenarios, enabling parametric studies for safety system validation and spent fuel canister optimization. The framework integrates with thermal-hydraulic codes and facilitates collaborative design reviews through its dual-interface architecture.

### Acknowledgements
This work was supported by the Idaho National Laboratory's Laboratory Directed Research & Development (LDRD) Program under DOE Idaho Operations Office Contract DE-AC07-05ID14517.

The manuscript was authored at Idaho National Laboratory by Battelle Energy Alliance LLC, Operator of Idaho National Laboratory under Contract No. DE-AC07-05ID14517 with the U.S. Department of Energy (DOE).

This research made use of Idaho National Laboratory's High Performance Computing systems located at the Collaborative Computing Center and supported by the Office of Nuclear Energy of the U.S. Department of Energy and the Nuclear Science User Facilities under Contract No. DE-AC07-05ID14517.

### References
- ANS 2019. "Decay Heat Power in Light Water Reactors." ANSI/ANS-5.1-2019.
- Bajpai, P., et al. 2020. "Thermochemical Interactions in MSRs." Nuclear Engineering and Design.
- Li, Y., et al. 2023. "Reduced Decay Heat Model for MSRs." RELAP5-3D Documentation.
- Pathirana, A., et al. 2021. "Simplified Decay Heat Models." Journal of Nuclear Materials.
- Massone, L., et al. 2020. "Decay Heat Modeling in MSFR." SAMOFAR Project Report.
- Ebiwonjumi, B., et al. 2021. "Machine Learning for Decay Heat Prediction." Annals of Nuclear Energy.
- Solans, A., et al. 2025. "Isotope-Specific Activity Prediction Using ML." Journal of Nuclear Science.
- Niu, Z., et al. 2019. "ML for Nuclear Half-Life Predictions." Physical Review C.
- Li, W., et al. 2022. "Machine Learning in Nuclear Decay." Nuclear Data Sheets.
- Tano, M.E., et al. 2025. "Decay Heat Characterization in MSRs." Nuclear Engineering and Design.
- For more details, please refer to the full paper and the documentation within the repository.

## Contact
For any questions or further information, please contact the authors:

- Mauricio E. Tano (mauricio.tanoretamales@inl.gov)
- Parikshit Bajpai (parikshit.bajpai@inl.gov)
- Rodrigo de Oliveira (rodrigo.oliveira@inl.gov)
- Elizabeth Parker-Quaife (elizabeth.parker-quaife@inl.gov)
- Krystiane Otis (krystiane.otis@inl.gov)
- Xingyue Yang (xingyue.yang@inl.gov)
