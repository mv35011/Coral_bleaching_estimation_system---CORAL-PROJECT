import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
import requests
from io import StringIO
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv


from streamlit.runtime.caching import cache_resource

st.set_page_config(
    page_title="Project CORAL",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(APP_DIR, 'coral_bleaching_model.pkl')
HISTORICAL_DATA_FILE = os.path.join(APP_DIR, 'coral_data_PROCESSED.csv')
FINLEY_CONTEXT_FILE = os.path.join(APP_DIR, 'coral_context.txt')


REEF_LOCATIONS = {
    "Andaman_Islands": {"lat": 11.25, "lon": 92.77},
    "Lakshadweep_Islands": {"lat": 10.56, "lon": 72.64},
    "Gulf_of_Mannar": {"lat": 8.80, "lon": 78.25},
    "Gulf_of_Kutch": {"lat": 22.47, "lon": 69.07},
}


@st.cache_resource
def load_model():
    """Load the trained machine learning model from file."""
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{MODEL_FILE}'.")
        return None

@st.cache_data
def load_historical_data():
    """Load the processed historical data from CSV."""
    try:
        df = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=['time'])
        return df
    except FileNotFoundError:
        st.error(f"Error: Historical data file not found at '{HISTORICAL_DATA_FILE}'.")
        return None

@st.cache_data
def load_finley_context():
    """Load the static context (persona) for the chatbot."""
    try:
        with open(FINLEY_CONTEXT_FILE, 'r') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: Finley's context file not found at '{FINLEY_CONTEXT_FILE}'.")
        return None

def get_finley_response_groq(system_prompt, chat_history, rag_context):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set.")
        return "Oops! My brain’s unplugged (no API key)."

    api_url = "https://api.groq.com/openai/v1/chat/completions"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg and msg["content"].strip():
            messages.append(msg)
    messages.append({"role": "user", "content": rag_context.strip()})

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            st.error(f"Groq API returned {response.status_code}: {response.text}")
            return "Hmm… Finley’s brain shorted (API error)."
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "I’m struggling to connect to Groq right now."


@st.cache_data(ttl=3600)
def get_live_data(lat, lon):
    """Fetches the last 30 days of data."""
    end_date = datetime.utcnow() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)
    server_urls = [
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv",
        "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/NOAA_DHW.csv"
    ]
    variables = ["CRW_SST", "CRW_HOTSPOT", "CRW_DHW", "CRW_SSTANOMALY", "CRW_BAA", "CRW_BAA_7D_MAX"]
    query_parts = []
    for var in variables:
        query_parts.append(
            f"{var}[({start_date.strftime('%Y-%m-%d')}T12:00:00Z):1:({end_date.strftime('%Y-%m-%d')}T12:00:00Z)][({lat}):1:({lat})][({lon}):1:({lon})]"
        )
    query = ",".join(query_parts)
    for base_url in server_urls:
        request_url = f"{base_url}?{query}"
        try:
            response = requests.get(request_url, timeout=30)
            response.raise_for_status()
            csv_data = response.text
            if "ERROR" in csv_data or len(csv_data) < 100:
                continue
            df = pd.read_csv(StringIO(csv_data), skiprows=[1])
            df.columns = [
                'time', 'latitude', 'longitude', 'sea_surface_temp_c', 'hotspot_c',
                'degree_heating_week_c_weeks', 'sst_anomaly_c', 'bleaching_alert_area',
                'bleaching_alert_area_7d_max'
            ]
            df['time'] = pd.to_datetime(df['time'])
            return df.sort_values('time').iloc[-1:]
        except requests.exceptions.RequestException:
            continue
    return None


def preprocess_live_data(df):
    """Preprocesses live data to match the model's training format."""
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    return df


def create_risk_gauge(risk_value):
    """Creates a Plotly gauge chart for the risk score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Risk Status"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "rgba(0,0,0,0)"}, 
            'steps': [
                {'range': [0, 30], 'color': '#28a745'}, # Green
                {'range': [30, 60], 'color': '#ffc107'}, # Yellow
                {'range': [60, 100], 'color': '#dc3545'} # Red
            ],
        }))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def get_risk_status(prediction):
    """Returns a status string and color based on the risk percentage."""
    if prediction < 30:
        return "LOW RISK", "#28a745" # Green
    elif prediction < 60:
        return "WARNING", "#ffc107" # Yellow
    else:
        return "CRITICAL", "#dc3545" # Red

def main():
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Please add it to .env or Streamlit Secrets.")
        st.stop()

    model = load_model()
    historical_df = load_historical_data()
    finley_context = load_finley_context()

    if model is None or historical_df is None or finley_context is None:
        st.stop()

    st.title("🐠 Project CORAL: The Coral Oracle")
    st.markdown("""
        Welcome to Project CORAL, an AI-powered early warning system designed to protect India's precious marine ecosystems. 
        This tool provides real-time predictions of coral bleaching risk for key reef locations. 
    """)
    st.divider()

    with st.sidebar:
        st.header("Select a Reef Location")
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        for name, coords in REEF_LOCATIONS.items():
            folium.Marker(
                location=[coords['lat'], coords['lon']],
                popup=name.replace("_", " "),
                tooltip=name.replace("_", " "),
                icon=folium.Icon(color='blue', icon='water')
            ).add_to(m)
        map_data = st_folium(m, width=380, height=380)

        if map_data and map_data.get("last_object_clicked_popup"):
            clicked_name = map_data["last_object_clicked_popup"].replace(" ", "_")
            if clicked_name in REEF_LOCATIONS:
                
                if st.session_state.get("selected_location") != clicked_name:
                    
                    if f"chat_history_{clicked_name}" in st.session_state:
                        st.session_state[f"chat_history_{clicked_name}"] = []
                st.session_state.selected_location = clicked_name

        st.info("Click a marker to load the reef dashboard.")
        st.markdown(f"**Data Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")

        
        with st.expander("About this Project"):
            st.markdown("""
                This app uses a `scikit-learn` model trained on NOAA satellite data to predict coral bleaching risk. 
                
                The conversational chat bot is powered by **Groq** using the Llama 3 model.
                
                Data Sources:
                - NOAA Coral Reef Watch
                - Copernicus Marine Service (for future pH data)
            """)


    if 'selected_location' not in st.session_state:
        st.info("Please select a reef from the map.")
    else:
        location_name = st.session_state.selected_location
        coords = REEF_LOCATIONS[location_name]

        st.header(f"Dashboard for: {location_name.replace('_', ' ')}")

        with st.spinner("Fetching and analyzing data..."):
            live_df_raw = get_live_data(coords['lat'], coords['lon'])
            is_live_data = True
            fallback_date = ""
            if live_df_raw is None or live_df_raw.empty:
                is_live_data = False
                location_historical_df_all = historical_df[historical_df['location_name'] == location_name]
                live_df_raw = location_historical_df_all.sort_values('time').iloc[-1:].copy()
                fallback_date = pd.to_datetime(live_df_raw['time'].iloc[0]).strftime('%B %d, %Y')
                st.warning(
                    f"⚠️ Live data unavailable. Showing last recorded data from {fallback_date}.",
                    icon="🛰️"
                )

        live_df_processed = preprocess_live_data(live_df_raw.copy())
        features_for_model = [
            'sea_surface_temp_c', 'hotspot_c', 'degree_heating_week_c_weeks',
            'sst_anomaly_c', 'bleaching_alert_area', 'bleaching_alert_area_7d_max',
            'year', 'month', 'day_of_year', 'week_of_year'
        ]
        prediction = model.predict(live_df_processed[features_for_model])[0]

        location_historical_df = historical_df[historical_df['location_name'] == location_name].copy()
        max_risk_row = location_historical_df.loc[location_historical_df['bleaching_risk_percent'].idxmax()]

        tab1, tab2, tab3, tab4 = st.tabs([
            "🌊 Live Risk Assessment",
            "🔬 What-If Simulator",
            "📈 Historical Explorer",
            "🐠 Ask Finley"
        ])

        with tab1:
            col1, col2 = st.columns([1, 2])
            with col1:
                
                status_text, status_color = get_risk_status(prediction)
                st.markdown(f"### Status: <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
                st.metric("Predicted Bleaching Risk", f"{prediction:.2f}%")
                st.metric("Current SST", f"{live_df_raw['sea_surface_temp_c'].iloc[0]:.2f} °C")
                st.metric("Current DHW", f"{live_df_raw['degree_heating_week_c_weeks'].iloc[0]:.2f} °C-weeks")
            with col2:
                st.plotly_chart(create_risk_gauge(prediction), use_container_width=True)

        with tab2:
            st.subheader("Simulate Environmental Changes")
            st.info(
                """
                **Why only two sliders?** Sea Surface Temperature (SST) and Degree Heating Weeks (DHW) are the two primary, independent drivers of coral bleaching. 
                Other model features (like HotSpots and SST Anomaly) are derived directly from the temperature.
                """, 
                icon="💡"
            )
            base_sst = live_df_raw['sea_surface_temp_c'].iloc[0]
            base_dhw = live_df_raw['degree_heating_week_c_weeks'].iloc[0]
            sim_sst = st.slider("Sea Surface Temperature (°C)", base_sst - 2, base_sst + 4, base_sst, 0.1)
            sim_dhw = st.slider("Degree Heating Weeks (°C-weeks)", 0.0, base_dhw + 8, base_dhw, 0.1)
            sim_df = live_df_raw.copy()
            sim_df['sea_surface_temp_c'] = sim_sst
            sim_df['degree_heating_week_c_weeks'] = sim_dhw
            sim_df_processed = preprocess_live_data(sim_df)
            sim_prediction = model.predict(sim_df_processed[features_for_model])[0]
            st.metric("Simulated Bleaching Risk", f"{sim_prediction:.2f}%")

        with tab3:
            st.subheader("Explore Historical Trends")
            columns_to_plot = st.multiselect(
                "Select data to plot:",
                options=['sea_surface_temp_c', 'degree_heating_week_c_weeks', 'bleaching_risk_percent'],
                default=['sea_surface_temp_c', 'degree_heating_week_c_weeks', 'bleaching_risk_percent']
            )
            if columns_to_plot:
                fig = px.line(location_historical_df, x='time', y=columns_to_plot,
                              title=f"Historical Data for {location_name.replace('_', ' ')}")
                st.plotly_chart(fig, use_container_width=True)
                st.success(
                    f"**Historical Insight:** Max bleaching risk of {max_risk_row['bleaching_risk_percent']:.2f}% "
                    f"on {max_risk_row['time'].strftime('%B %d, %Y')}."
                )

        with tab4:
            st.subheader(f"Ask Finley about {location_name.replace('_', ' ')}")
            
            st.markdown("""
            Hi! I'm **Finley** 🐠, the local parrotfish for this reef. I know a lot about what's happening here. 
            Ask me a question, or try one of these:
            """)
            st.caption("""
            * "How are you feeling about the water temperature right now?"
            * "What was the worst event that ever happened here?"
            * "What does 'Degree Heating Week' mean?"
            """)
            st.divider()
            
            
            history_key = f"chat_history_{location_name}"
            if history_key not in st.session_state:
                st.session_state[history_key] = []

           
            for message in st.session_state[history_key]:
                avatar = "🐠" if message["role"] == "assistant" else "user"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask Finley about this reef..."):
                
                with st.chat_message("user"):
                    st.markdown(prompt)

                
                rag_context = f"""
Here is the current data for {location_name}:
- Data Status: {"Live" if is_live_data else f"Historical ({fallback_date})"}
- SST: {live_df_raw['sea_surface_temp_c'].iloc[0]:.2f}°C
- DHW: {live_df_raw['degree_heating_week_c_weeks'].iloc[0]:.2f}°C-weeks
- Risk: {prediction:.2f}%
- Historical Max Risk: {max_risk_row['bleaching_risk_percent']:.2f}% on {max_risk_row['time'].strftime('%B %d, %Y')}

My question is: {prompt}
"""
                with st.spinner("Finley is thinking..."):
                    
                    response_text = get_finley_response_groq(
                        finley_context,
                        st.session_state[history_key], 
                        rag_context 
                    )
                    
                    
                    st.session_state[history_key].append({"role": "user", "content": prompt})

                    
                    with st.chat_message("assistant", avatar="🐠"):
                        st.markdown(response_text)
                        
                    
                    st.session_state[history_key].append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()

