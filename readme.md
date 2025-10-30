🐠 Project CORAL: The Coral Oracle

A live, AI-powered dashboard and chatbot designed to predict and explain coral bleaching risk in key Indian reef locations.


1. Overview

Project CORAL is an end-to-end data science application that serves as an early warning system for coral bleaching. It combines a classical machine learning model (trained on 30+ years of satellite data) with a generative AI chatbot (powered by Groq) to provide a comprehensive, interactive, and user-friendly dashboard.

The application allows users to:

Get live (or near-live) predictions for coral bleaching risk.

Simulate how changes in ocean temperature would affect this risk.

Explore historical data to identify past bleaching events.

Have a natural language conversation with "Finley," an AI-powered fish, to understand the data.

2. Key Features

AI-Powered Risk Prediction: A GradientBoosting model trained on NOAA satellite data predicts the bleaching risk percentage with over 90% R² accuracy (on its heuristic-based training data).

"What-If" Scenario Simulator: Interactive sliders allow users to change the Sea Surface Temperature and Degree Heating Weeks to see how these factors would alter the model's risk prediction in real-time.

Historical Data Explorer: An interactive Plotly chart displays over 30 years of historical data, including temperature, heat stress, and the model's predicted risk over time, visually highlighting past bleaching events.

"Ask Finley" AI Chatbot: A conversational AI assistant powered by Groq (using Llama 3). Finley has a distinct persona and uses Retrieval-Augmented Generation (RAG) to answer questions about the live and historical data for the selected reef.

3. Tech Stack

This project was intentionally built to be robust and lightweight, avoiding unnecessary, complex dependencies.

Component

Technology

Why?

Data Science & ML

Pandas, Scikit-learn, Joblib

For robust data processing and training a reliable GradientBoosting model.

Application & UI

Streamlit, Streamlit-Folium, Plotly

To build a beautiful, interactive web app using only Python.

AI Chatbot

Groq API (Llama 3), requests

(No LangChain). We call the Groq API directly for maximum speed and to avoid complex dependency issues.

Environment

python-dotenv, .gitignore

For securely managing API keys locally.

4. How to Run Locally (Step-by-Step)

Follow these steps exactly to get the application running on your local machine.

Step 1: Clone the Repository

git clone https://github.com/mv35011/Coral_bleaching_estimation_system---CORAL-PROJECT.git
cd Coral_bleaching_estimation_system---CORAL-PROJECT


Step 2: Create a Python Virtual Environment

It's critical to use a virtual environment to avoid package conflicts.

# Create the virtual environment
python -m venv .venv

# Activate it:
# On Windows (Powershell)
.\.venv\Scripts\Activate
# On Mac/Linux
source .venv/bin/activate


Step 3: Create Your .env File

This file stores your secret API key. It is hidden from GitHub by the .gitignore file.

In the root of the project, create a file named .env

Open it and add your Groq API key:

GROQ_API_KEY="your-actual-api-key-from-groq-goes-here"


Step 4: Create Finley's Context File

The app needs this file to load Finley's persona.

Go into the app/ folder.

Create a file named coral_context.txt

Paste the persona text from the file I provided in the previous step (or ask me for it again).

Step 5: Install All Requirements

Use the requirements.txt file to install the exact, compatible versions of all libraries.

# Make sure your .venv is active!
pip install -r requirements.txt


Step 6: Re-Train The Model (CRITICAL)

This is the most important step. The included .pkl file was trained in a different environment and will fail due to a numpy version mismatch. You must re-train it in your new environment.

Make sure you have retrain_model.py in your project root.

Run the script:

# This will load data from app/ and save the new, compatible model to app/
python retrain_model.py


Step 7: Run the Streamlit App!

You are now ready to launch.

# Run this from the ROOT of your project
streamlit run app/app.py


Your browser should open automatically to http://localhost:8501, and the complete application will be running.

5. Deployment

This app is deployed on Streamlit Community Cloud.

API Key: The GROQ_API_KEY is not in the .env file. It is stored securely in Streamlit's Secrets management.

File Paths: The app is configured to run from the root of the repository, with the "Main file path" in Streamlit's settings pointed to app/app.py.