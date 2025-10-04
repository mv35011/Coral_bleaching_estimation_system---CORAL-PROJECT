# 🪸 Project CORAL: The Coral Oracle

> An AI-Powered Early Warning System for Coral Bleaching Events

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Vision](#-project-vision)
- [The Problem: A Silent Crisis](#-the-problem-a-silent-crisis)
- [Our Solution: A Predictive Framework](#-our-solution-a-predictive-framework)
- [System Architecture & Data Flow](#-system-architecture--data-flow)
- [The Data Plan: A Two-Phase Strategy](#-the-data-plan-a-two-phase-strategy)
  - [Phase 1: The Heuristic-Based Prototype](#phase-1-the-heuristic-based-prototype)
  - [Phase 2: The Ground-Truth Model](#phase-2-the-ground-truth-model)
- [Technology Stack](#-technology-stack)
- [Project Roadmap & Milestones](#-project-roadmap--milestones)
- [How to Run This Project](#-how-to-run-this-project)
- [Contributors](#-contributors)

---

## 🌊 Project Vision

**Project CORAL** aims to develop an intelligent, data-driven tool to forecast coral bleaching events for critical reef ecosystems around India. By synthesizing satellite data with machine learning, this project will provide a functional early warning system, demonstrating a powerful application of Green Tech to a pressing environmental challenge.

---

## 🚨 The Problem: A Silent Crisis

Coral reefs are vital marine ecosystems, but they are under severe threat from climate change. Rising sea temperatures cause **coral bleaching**, a stress response that can lead to mass coral mortality. Predicting these events is crucial for conservation efforts, but requires the analysis of complex, multi-variate environmental data.

---

## 💡 Our Solution: A Predictive Framework

We are building an **interactive web application** that allows a user to select a coral reef location and receive an AI-generated risk assessment for bleaching. The system will process a suite of oceanographic variables to generate a clear, actionable prediction.

**The final output will be a predicted coral bleaching percentage, presented with a color-coded risk level (Low, Medium, High).**

---

## 🏗️ System Architecture & Data Flow

Our architecture is designed for modularity and rapid development, ensuring a functional end-to-end pipeline.

### Data Flow Steps:

1. **Frontend Interaction**: A user interacts with a Folium map embedded in a Streamlit web application to select a reef location (e.g., Andaman Islands).

2. **Backend Data Acquisition**: The system queries public APIs from NOAA and Copernicus Marine Service for the latest environmental data corresponding to the selected coordinates (SST, DHW, pH, Salinity).

3. **Data Preprocessing**: The fetched data is cleaned, normalized using a pre-fitted StandardScaler, and structured into a feature vector.

4. **AI Model Inference**: This feature vector is fed into our trained Scikit-learn `GradientBoostingRegressor` model.

5. **Frontend Visualization**: The model's prediction (e.g., "45% Bleaching Risk") is sent back to the Streamlit UI and displayed to the user with supporting charts and risk indicators.

---

## 📊 The Data Plan: A Two-Phase Strategy

Acquiring high-quality "ground-truth" data is the primary challenge in environmental AI. To mitigate this risk and ensure project success, we are adopting a **two-phase approach**.

### Phase 1: The Heuristic-Based Prototype

**Proof of Concept**

This phase focuses on building a fully functional prototype to validate our architecture and modeling approach, even without real-world bleaching observations.

**Problem**: Machine learning models cannot be trained without a target variable (y). The principle of "Garbage In, Garbage Out" means using purely random dummy data would yield a meaningless model.

**Our Solution**: We will generate a scientifically-defensible proxy target variable using a heuristic based on NOAA's official Degree Heating Week (DHW) alert levels.

| DHW Value (Heat Stress) | NOAA Alert Level | Heuristic Observed_Bleaching_% (Our Proxy Label) |
|-------------------------|------------------|--------------------------------------------------|
| DHW = 0 | No Stress | Random value between 1-5% |
| 0 < DHW < 4 | Bleaching Watch/Warning | Random value between 10-30% |
| 4 ≤ DHW < 8 | Alert Level 1 (Bleaching Likely) | Random value between 30-60% |
| DHW ≥ 8 | Alert Level 2 (Mortality Likely) | Random value between 60-90% |

This approach allows our model to learn the complex, non-linear relationships between various environmental inputs and a logically sound, risk-stratified output. This prototype will be fully functional and serve as the basis for our engagement with academic experts.

### Phase 2: The Ground-Truth Model

**The Ultimate Goal**

With the functional prototype in hand, our primary goal is to replace the heuristic labels with real-world field survey data.

**Action Plan**:

- Conduct a time-boxed, intensive literature review of academic papers on Indian coral reefs.
- Identify and contact marine biologists and oceanographic institutions (NIO, ZSI) to request historical bleaching survey data for academic purposes.
- If successful, retrain our `GradientBoostingRegressor` model on this ground-truth data to achieve a higher level of predictive accuracy.

---

## 🛠️ Technology Stack

| Category | Tool / Library | Purpose |
|----------|---------------|---------|
| **Backend / AI** | Python 3.9+ | Core Language |
| | Scikit-learn | Model Training (GradientBoostingRegressor) |
| | Pandas / NumPy | Data Manipulation & Analysis |
| | Xarray / NetCDF4 | Parsing Scientific Data Formats |
| **Frontend / UI** | Streamlit | Interactive Web Application Framework |
| | Folium | Interactive Mapping |
| | Plotly | Data Visualization & Charting |
| **Data Sources** | NOAA / Copernicus | APIs for Environmental Data |
| **Deployment** | GitHub / Streamlit Community Cloud | Version Control & Public Hosting |

---

## 🗓️ Project Roadmap & Milestones

### Sprint 1: Foundation & Data Pipeline (Weeks 1-2)

- [ ] Initialize GitHub repository with this README.
- [ ] Set up the Python environment (`requirements.txt`).
- [ ] Develop scripts to fetch and process data from NOAA & Copernicus.
- [ ] Implement the heuristic labeling system (Phase 1).
- [ ] **Deliverable**: A complete, ML-ready dataset with heuristic labels.

### Sprint 2: Model Development & Baseline UI (Weeks 3-4)

- [ ] Train the first version of the `GradientBoostingRegressor` model.
- [ ] Save the trained model (`.pkl`) and the `StandardScaler`.
- [ ] Build a basic Streamlit app that takes numerical inputs and shows a prediction.
- [ ] **Deliverable**: A baseline predictive model and a simple UI.

### Sprint 3: Interactive UI & Integration (Weeks 5-6)

- [ ] Integrate the Folium map into the Streamlit app.
- [ ] Enable user interaction (clicking map to trigger prediction).
- [ ] Design and implement clear data visualizations for the results.
- [ ] **Deliverable**: A fully integrated, interactive prototype.

### Sprint 4: Refinement, Deployment & Documentation (Weeks 7-8)

- [ ] Refine the model (hyperparameter tuning).
- [ ] Improve UI/UX and add explanatory text.
- [ ] Deploy the application to Streamlit Community Cloud.
- [ ] Prepare the final project report and presentation.
- [ ] **Deliverable**: A publicly deployed application and final documentation.

---

## 🚀 How to Run This Project

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/[your-username]/Project-CORAL.git
   cd Project-CORAL
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   - Open your browser and navigate to `http://localhost:8501`

---

## 👥 Contributors

- **[Your Name]** - [GitHub Profile](https://github.com/your-username)
- **[Teammate's Name]** - [GitHub Profile](https://github.com/teammate-username)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NOAA Coral Reef Watch for providing critical environmental data
- Copernicus Marine Service for oceanographic datasets
- The marine biology community for their invaluable research on coral reef ecosystems

---

<div align="center">

**Built with 💙 for the oceans**

⭐ Star this repo if you believe in protecting our coral reefs!

</div>