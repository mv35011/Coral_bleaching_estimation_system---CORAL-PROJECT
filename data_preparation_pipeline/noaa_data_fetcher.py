import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import argparse
import time
import calendar  # Added for robust date calculations


def get_coral_reef_watch_data(lat, lon, start_date, end_date, max_retries=3):
    """
    Fetches all 6 core NOAA Coral Reef Watch variables for a specific location and time range.

    Args:
        lat (float): Latitude of the target location.
        lon (float): Longitude of the target location.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        max_retries (int): Maximum number of retry attempts.

    Returns:
        pandas.DataFrame: A DataFrame containing the time-series data, or None if the request fails.
    """

    # Correct NOAA Coral Reef Watch ERDDAP server URL
    base_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv"

    # Constructing the query part of the URL for all 6 core variables
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

            # Check for server errors in response
            if "ERROR" in csv_data or len(csv_data) < 100:
                print(f"Server error or no data available")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

            df = pd.read_csv(StringIO(csv_data), skiprows=[1])

            # Rename columns for clarity
            df.columns = [
                'time', 'latitude', 'longitude', 'sea_surface_temp_c', 'hotspot_c',
                'degree_heating_week_c_weeks', 'sst_anomaly_c', 'bleaching_alert_area',
                'bleaching_alert_area_7d_max'
            ]

            df['time'] = pd.to_datetime(df['time'])
            print(f"Success! ({len(df)} rows)")
            return df

        except requests.exceptions.Timeout:
            print(f"Timeout")
            if attempt < max_retries - 1:
                print(f"  Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"  Max retries reached")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"  Max retries reached")
                return None

    return None


def fetch_data_in_chunks(lat, lon, start_year, end_year, chunk_months=3):
    """
    Fetches data in smaller, more reliable monthly chunks.

    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_year (int): Starting year
        end_year (int): Ending year
        chunk_months (int): Number of months per request (3 is a safe default)

    Returns:
        list: List of DataFrames
    """
    all_dfs = []

    for year in range(start_year, end_year + 1):
        for month_start in range(1, 13, chunk_months):
            month_end = min(month_start + chunk_months - 1, 12)

            start_date = f"{year}-{month_start:02d}-01"

            # Get the last day of the end month for that year to handle leap years
            end_day = calendar.monthrange(year, month_end)[1]
            end_date = f"{year}-{month_end:02d}-{end_day}"

            print(f"Fetching {start_date} to {end_date}...")
            df = get_coral_reef_watch_data(lat, lon, start_date, end_date)

            if df is not None and not df.empty:
                all_dfs.append(df)
            else:
                print(f"  No data returned for this chunk.")

            # Be polite to the server - wait between requests
            time.sleep(2)

    return all_dfs


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch NOAA Coral Reef Watch data for key Indian reef locations."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='coral_environmental_data_master.csv',
        help='Path for the output CSV file. (default: coral_environmental_data_master.csv)'
    )
    parser.add_argument(
        '-s', '--start-year',
        type=int,
        default=1985,
        help='Start year for data collection (default: 1985)'
    )
    parser.add_argument(
        '-e', '--end-year',
        type=int,
        default=None,
        help='End year for data collection (default: last complete year)'
    )
    parser.add_argument(
        '-c', '--chunk-months',
        type=int,
        default=3,
        help='Chunk size in months per request (e.g., 3=quarterly, 6=half-year). Default: 3 (safest)'
    )
    args = parser.parse_args()

    reef_locations = {
        "Andaman_Islands": {"lat": 11.25, "lon": 92.77},
        "Lakshadweep_Islands": {"lat": 10.56, "lon": 72.64},
        "Gulf_of_Mannar": {"lat": 8.78, "lon": 78.14},
        "Gulf_of_Kutch": {"lat": 22.47, "lon": 69.07}
    }

    start_year = args.start_year
    end_year = args.end_year if args.end_year else datetime.now().year - 1

    print("=" * 70)
    print("--- Starting Data Acquisition for Project CORAL ---")
    print("=" * 70)
    print(f"Fetching data for the period: {start_year}-01-01 to {end_year}-12-31")
    print(f"Chunk size: {args.chunk_months} months per request")
    print(f"Locations: {len(reef_locations)}")
    total_requests = len(reef_locations) * (end_year - start_year + 1) * (12 // args.chunk_months)
    print(f"Estimated total requests: {total_requests}")
    print(f"Estimated time: ~{int((total_requests * 5) / 60)} minutes")  # Estimated time calc
    print("=" * 70)
    print()

    all_location_dfs = []

    for idx, (name, coords) in enumerate(reef_locations.items(), 1):
        print(f"[{idx}/{len(reef_locations)}] Processing location: {name}")
        print(f"  Coordinates: ({coords['lat']}, {coords['lon']})")
        print("-" * 70)

        location_dfs = fetch_data_in_chunks(
            coords['lat'],
            coords['lon'],
            start_year,
            end_year,
            args.chunk_months
        )

        if location_dfs:
            location_df = pd.concat(location_dfs, ignore_index=True)
            location_df['location_name'] = name

            # Sort by time
            location_df = location_df.sort_values('time').reset_index(drop=True)

            all_location_dfs.append(location_df)
            print(f"✅ Successfully collected {len(location_df)} records for {name}")
        else:
            print(f"❌ Failed to download any data for {name}")

        print()

    if all_location_dfs:
        print("=" * 70)
        print("--- Combining all location data into master file ---")
        print("=" * 70)

        master_df = pd.concat(all_location_dfs, ignore_index=True)

        # Reorder columns to put location_name after coordinates
        cols = ['time', 'latitude', 'longitude', 'location_name',
                'sea_surface_temp_c', 'hotspot_c', 'degree_heating_week_c_weeks',
                'sst_anomaly_c', 'bleaching_alert_area', 'bleaching_alert_area_7d_max']
        master_df = master_df[cols]

        # Sort by location and time
        master_df = master_df.sort_values(['location_name', 'time']).reset_index(drop=True)

        output_filename = args.output
        master_df.to_csv(output_filename, index=False)

        print(f"\n✅ SUCCESS! All data combined and saved to '{output_filename}'")
        print(f"Total rows collected: {len(master_df):,}")
        print(f"Date range: {master_df['time'].min().date()} to {master_df['time'].max().date()}")
        print(f"\nData summary by location:")
        print(master_df.groupby('location_name').size())
        print("\nFirst 5 rows of the master dataset:")
        print(master_df.head())
        print("\nLast 5 rows of the master dataset:")
        print(master_df.tail())
    else:
        print("=" * 70)
        print("❌ FAILED: No data was downloaded for any location")
        print("=" * 70)
        print("\nPossible issues:")
        print("1. Check your internet connection")
        print("2. Verify the NOAA ERDDAP server is accessible")
        print("3. Confirm the coordinates are within the data coverage area")
        print("4. Try a smaller date range or fewer locations")

