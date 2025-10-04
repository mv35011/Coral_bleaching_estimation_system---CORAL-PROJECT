import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import argparse
import time
import calendar

def get_coral_reef_watch_data(lat, lon, start_date, end_date, max_retries=3):
    """
    Fetches all 6 core NOAA Coral Reef Watch variables for a specific location and time range.
    (This is the same robust function from the main script)
    """
    base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv"
    variables = [
        "CRW_SST", "CRW_HOTSPOT", "CRW_DHW", "CRW_SSTANOMALY",
        "CRW_BAA", "CRW_BAA_7D_MAX"
    ]
    query_parts = []
    for var in variables:
        query_parts.append(
            f"{var}[({start_date}T12:00:00Z):1:({end_date}T12:00:00Z)][({lat}):1:({lat})][({lon}):1:({lon})]")
    query = ",".join(query_parts)
    request_url = f"{base_url}?{query}"

    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...", end=" ")
            response = requests.get(request_url, timeout=120)
            response.raise_for_status()
            csv_data = response.text
            if "ERROR" in csv_data or len(csv_data) < 100:
                print(f"Server error or no data")
                if attempt < max_retries - 1: time.sleep(5)
                continue
            df = pd.read_csv(StringIO(csv_data), skiprows=[1])
            df.columns = [
                'time', 'latitude', 'longitude', 'sea_surface_temp_c', 'hotspot_c',
                'degree_heating_week_c_weeks', 'sst_anomaly_c', 'bleaching_alert_area',
                'bleaching_alert_area_7d_max'
            ]
            df['time'] = pd.to_datetime(df['time'])
            print(f"Success! ({len(df)} rows)")
            return df
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1: time.sleep(10)
    return None

def fetch_data_in_chunks(lat, lon, start_year, end_year, chunk_months=3):
    """
    Fetches data in smaller, reliable monthly chunks.
    (This is the same robust function from the main script)
    """
    all_dfs = []
    for year in range(start_year, end_year + 1):
        for month_start in range(1, 13, chunk_months):
            month_end = min(month_start + chunk_months - 1, 12)
            start_date = f"{year}-{month_start:02d}-01"
            end_day = calendar.monthrange(year, month_end)[1]
            end_date = f"{year}-{month_end:02d}-{end_day}"
            print(f"Fetching {start_date} to {end_date}...")
            df = get_coral_reef_watch_data(lat, lon, start_date, end_date)
            if df is not None and not df.empty:
                all_dfs.append(df)
            else:
                print(f"  No data returned for this chunk.")
            time.sleep(2)
    return all_dfs

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch missing NOAA data for Gulf of Mannar."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='gulf_of_mannar_data.csv',
        help='Path for the output CSV file.'
    )
    parser.add_argument('-s', '--start-year', type=int, default=1985)
    parser.add_argument('-e', '--end-year', type=int, default=datetime.now().year - 1)
    args = parser.parse_args()

    # Corrected, offshore coordinates for Gulf of Mannar
    location_to_fetch = {
        "name": "Gulf_of_Mannar",
        "lat": 8.80,
        "lon": 78.25
    }

    print("=" * 50)
    print("--- Fetching Missing Data for Gulf of Mannar ---")
    print("=" * 50)

    location_dfs = fetch_data_in_chunks(
        location_to_fetch['lat'],
        location_to_fetch['lon'],
        args.start_year,
        args.end_year
    )

    if location_dfs:
        location_df = pd.concat(location_dfs, ignore_index=True)
        location_df['location_name'] = location_to_fetch['name']
        location_df = location_df.sort_values('time').reset_index(drop=True)

        output_filename = args.output
        location_df.to_csv(output_filename, index=False)
        print(f"\n✅ Success! Missing data saved to '{output_filename}'")
    else:
        print("\n❌ FAILED: Could not download data for Gulf of Mannar.")
