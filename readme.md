# 🐠 Project CORAL: The Coral Oracle

> **An AI-powered dashboard and chatbot that predicts and explains coral bleaching risk in Indian reef ecosystems.**  
> Built for environmental data intelligence, marine conservation, and public awareness.

---

## 🌊 1. Overview

**Project CORAL** is an end-to-end data science application designed as an **early warning system for coral bleaching**.  
It combines **machine learning** and **generative AI** to create a powerful, interactive dashboard where users can:

- 🌡️ **Get live predictions** for coral bleaching risk.  
- 🧪 **Simulate climate scenarios** (e.g., increased sea temperature).  
- 📈 **Explore historical data** and visualize past bleaching events.  
- 🤖 **Chat with "Finley"**, an AI-powered fish, to understand coral data naturally.

At its core, Project CORAL merges **data science**, **climate analytics**, and **AI explainability** — making complex environmental insights accessible to everyone.

---

## ⚙️ 2. Key Features

### 🧠 AI-Powered Risk Prediction
A **GradientBoosting Regressor** trained on **30+ years of NOAA satellite data** predicts bleaching risk with **>90% R² accuracy** (on heuristic-based validation).

### 🎛️ "What-If" Scenario Simulator
Interactive sliders let users **modify Sea Surface Temperature** and **Degree Heating Weeks (DHW)** to see real-time changes in predicted risk.

### 📊 Historical Data Explorer
An interactive **Plotly dashboard** visualizes temperature trends, DHW, and bleaching events across **three decades**, helping detect patterns and anomalies.

### 🐟 "Ask Finley" AI Chatbot
A conversational assistant built using the **Groq API (Llama 3)**.  
Finley uses **Retrieval-Augmented Generation (RAG)** to answer context-aware questions about both **live** and **historical** data — making climate science conversational.

---

## 🧰 3. Tech Stack

| Component | Technology | Purpose |
|------------|-------------|----------|
| **Data Science & ML** | `Pandas`, `Scikit-learn`, `Joblib` | Data preprocessing, training, and saving the GradientBoosting model. |
| **App & UI** | `Streamlit`, `Streamlit-Folium`, `Plotly` | For building an elegant, responsive, and interactive dashboard using pure Python. |
| **AI Chatbot** | `Groq API (Llama 3)`, `requests` | Lightweight, fast, and dependency-free conversational AI (no LangChain). |
| **Environment & Config** | `python-dotenv`, `.gitignore` | Secure management of API keys and local environment setup. |

---

## 🧩 4. How to Run Locally

Follow these steps exactly to get **Project CORAL** running on your local system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/mv35011/Coral_bleaching_estimation_system---CORAL-PROJECT.git
cd Coral_bleaching_estimation_system---CORAL-PROJECT
