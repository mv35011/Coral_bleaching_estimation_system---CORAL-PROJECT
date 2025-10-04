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
import time
st.set_page_config(
    page_title="Project CORAL",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded",
)
REEF_LOCATIONS = {
    "Andaman_Islands": {"lat": 11.25, "lon": 92.77},
    "Lakshadweep_Islands": {"lat": 10.56, "lon": 72.64},
    "Gulf_of_Mannar": {"lat": 8.80, "lon": 78.25},
    "Gulf_of_Kutch": {"lat": 22.47, "lon": 69.07},
}
MODEL_FILE = 'coral_bleaching_model.pkl'
HISTORICAL_DATA_FILE = 'coral_data_PROCESSED.csv'
@st.cache_resource
def load_model():
    """Load the trained machine learning model from file."""
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{MODEL_FILE}'. Please ensure the model is in the same directory.")
        return None


@st.cache_data
def load_historical_data():
    """Load the processed historical data from CSV."""
    try:
        df = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=['time'])
        return df
    except FileNotFoundError:
        st.error(
            f"Error: Historical data file not found at '{HISTORICAL_DATA_FILE}'. Please run the preprocessor script.")
        return None
@st.cache_data(ttl=3600)
def get_live_data(lat, lon):
    """
    Fetches the last 30 days of data, trying multiple servers for reliability.
    """
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

    for i, base_url in enumerate(server_urls):
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
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 60], 'color': 'yellow'},
                {'range': [60, 100], 'color': 'red'}],
        }))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig
def main():
    model = load_model()
    historical_df = load_historical_data()

    if model is None or historical_df is None:
        st.stop()
    st.title("🐠 Project CORAL: The Coral Oracle")
    st.markdown("""
        Welcome to Project CORAL, an AI-powered early warning system designed to protect India's precious marine ecosystems. 
        This tool provides real-time predictions of coral bleaching risk for key reef locations. By leveraging satellite data and machine learning, 
        we aim to provide valuable insights to researchers, conservationists, and the community to aid in the monitoring and preservation of our vital coral reefs.
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
                st.session_state.selected_location = clicked_name

        st.info("Click a marker on the map to load the risk dashboard for that location.")
        st.markdown(f"**Data Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    if 'selected_location' not in st.session_state:
        st.info("Please select a reef from the map in the sidebar to begin.")
    else:
        location_name = st.session_state.selected_location
        coords = REEF_LOCATIONS[location_name]

        st.header(f"Dashboard for: {location_name.replace('_', ' ')}")
        with st.spinner("Fetching and analyzing data..."):
            live_df_raw = get_live_data(coords['lat'], coords['lon'])

            if live_df_raw is None or live_df_raw.empty:
                location_historical_df = historical_df[historical_df['location_name'] == location_name]
                live_df_raw = location_historical_df.sort_values('time').iloc[-1:].copy()
                fallback_date = pd.to_datetime(live_df_raw['time'].iloc[0]).strftime('%B %d, %Y')
                st.warning(
                    f"⚠️ Could not connect to live data servers. Displaying the most recent historical data from **{fallback_date}**.",
                    icon="🛰️")
        tab1, tab2, tab3 = st.tabs(
            ["🌊 Live Risk Assessment", "🔬 'What-If' Scenario Simulator", "📈 Historical Data Explorer"])
        with tab1:
            if live_df_raw is not None and not live_df_raw.empty:
                live_df_processed = preprocess_live_data(live_df_raw.copy())
                features_for_model = [
                    'sea_surface_temp_c', 'hotspot_c', 'degree_heating_week_c_weeks',
                    'sst_anomaly_c', 'bleaching_alert_area', 'bleaching_alert_area_7d_max',
                    'year', 'month', 'day_of_year', 'week_of_year'
                ]
                prediction = model.predict(live_df_processed[features_for_model])[0]
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Predicted Bleaching Risk", value=f"{prediction:.2f}%")
                    st.metric(label="Current Sea Surface Temp",
                              value=f"{live_df_raw['sea_surface_temp_c'].iloc[0]:.2f} °C")
                    st.metric(label="Current Degree Heating Weeks",
                              value=f"{live_df_raw['degree_heating_week_c_weeks'].iloc[0]:.2f} °C-weeks")
                with col2:
                    st.plotly_chart(create_risk_gauge(prediction), use_container_width=True)
            else:
                st.error("Could not retrieve any data to perform risk assessment.")
        with tab2:
            st.subheader("Simulate Environmental Changes")
            st.markdown(
                "Use the sliders below to see how changes in key environmental factors could affect the bleaching risk. The simulation uses the latest available data as a baseline.")

            st.info(
                """
                **Why only two sliders?** Sea Surface Temperature (SST) and Degree Heating Weeks (DHW) are the two primary, independent drivers of coral bleaching. 
                Other model features (like HotSpots and SST Anomaly) are derived directly from the temperature, so we only need to adjust the root cause to see the effect.
                """,
                icon="💡"
            )

            if live_df_raw is not None and not live_df_raw.empty:
                base_sst = live_df_raw['sea_surface_temp_c'].iloc[0]
                base_dhw = live_df_raw['degree_heating_week_c_weeks'].iloc[0]
                sim_sst = st.slider("Sea Surface Temperature (°C)", min_value=base_sst - 2, max_value=base_sst + 4,
                                    value=base_sst, step=0.1)
                sim_dhw = st.slider("Degree Heating Weeks (°C-weeks)", min_value=0.0, max_value=base_dhw + 8,
                                    value=base_dhw, step=0.1)
                sim_df = live_df_raw.copy()
                sim_df['sea_surface_temp_c'] = sim_sst
                sim_df['degree_heating_week_c_weeks'] = sim_dhw
                sim_df_processed = preprocess_live_data(sim_df)

                features_for_model = [
                    'sea_surface_temp_c', 'hotspot_c', 'degree_heating_week_c_weeks',
                    'sst_anomaly_c', 'bleaching_alert_area', 'bleaching_alert_area_7d_max',
                    'year', 'month', 'day_of_year', 'week_of_year'
                ]

                sim_prediction = model.predict(sim_df_processed[features_for_model])[0]
                st.divider()
                st.metric(label="Simulated Bleaching Risk", value=f"{sim_prediction:.2f}%")
                st.info(
                    "This simulation provides an estimate based on the model's learned patterns. Real-world outcomes can be influenced by other complex factors.")
            else:
                st.error("Data is required for the simulator. Please try again later.")
        with tab3:
            st.subheader("Explore Historical Trends")
            location_historical_df = historical_df[historical_df['location_name'] == location_name].copy()
            default_selection = ['sea_surface_temp_c', 'degree_heating_week_c_weeks', 'bleaching_risk_percent']
            columns_to_plot = st.multiselect(
                "Select data to plot:",
                options=default_selection + [col for col in location_historical_df.columns if
                                             col not in default_selection and location_historical_df[col].dtype in [
                                                 'float64', 'int64']],
                default=default_selection
            )
            if columns_to_plot:
                fig = px.line(location_historical_df, x='time', y=columns_to_plot,
                              title=f'Historical Data for {location_name.replace("_", " ")}',
                              labels={'value': 'Value', 'time': 'Date', 'variable': 'Metric'})
                st.plotly_chart(fig, use_container_width=True)
                max_risk_row = location_historical_df.loc[location_historical_df['bleaching_risk_percent'].idxmax()]
                st.success(
                    f"**Historical Insight:** The highest predicted bleaching risk of **{max_risk_row['bleaching_risk_percent']:.2f}%** "
                    f"occurred on **{max_risk_row['time'].strftime('%B %d, %Y')}**, "
                    f"when the Degree Heating Weeks reached {max_risk_row['degree_heating_week_c_weeks']:.2f}."
                )
            else:
                st.warning("Please select at least one metric to plot.")


if __name__ == "__main__":
    main()

