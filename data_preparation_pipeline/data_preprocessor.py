import pandas as pd
import numpy as np
import argparse


def apply_heuristic_labels(df):
    """
    Applies bleaching percentage labels based on Degree Heating Weeks (DHW).
    This function creates our target variable for the model.
    """
    print("\nApplying heuristic labels based on DHW...")

    conditions = [
        (df['degree_heating_week_c_weeks'] == 0),
        (df['degree_heating_week_c_weeks'] > 0) & (df['degree_heating_week_c_weeks'] < 4),
        (df['degree_heating_week_c_weeks'] >= 4) & (df['degree_heating_week_c_weeks'] < 8),
        (df['degree_heating_week_c_weeks'] >= 8)
    ]
    choices = [
        np.random.uniform(1, 5, size=len(df)),
        np.random.uniform(10, 30, size=len(df)),
        np.random.uniform(30, 60, size=len(df)),
        np.random.uniform(60, 90, size=len(df))
    ]

    df['bleaching_risk_percent'] = np.select(conditions, choices, default=0)
    df['bleaching_risk_percent'] = df['bleaching_risk_percent'].round(2)

    print("✅ Heuristic labels applied.")
    return df


def feature_engineer(df):
    """
    Creates new time-based features from the 'time' column.
    """
    print("\nStarting feature engineering...")
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)

    print("✅ Time-based features (year, month, day_of_year, week_of_year) created.")
    return df


def clean_data(df):
    """
    Performs initial cleaning and inspection of the dataset.
    """
    print("\n--- Starting Data Cleaning and Inspection ---")
    print("\n1. Initial Data Info:")
    df.info()

    print("\n2. Checking for Missing Values (NaN):")
    missing_values = df.isnull().sum()
    print(missing_values)

    if missing_values.sum() > 0:
        print("\nHandling missing values...")
        df.fillna(method='ffill', inplace=True)
        print("✅ Missing values handled using forward fill.")
    else:
        print("✅ No missing values found.")

    print("\n3. Checking for Duplicates:")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"✅ Removed {duplicates} duplicate rows.")
    else:
        print("✅ No duplicate rows found.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean, feature engineer, and apply heuristic labels to the coral dataset."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input CSV file (e.g., coral_data_COMPLETE.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='coral_data_PROCESSED.csv',
        help='Path for the output processed CSV file.'
    )
    args = parser.parse_args()

    print(f"--- Loading data from '{args.input_file}' ---")
    try:
        raw_df = pd.read_csv(args.input_file)
        print(f"✅ Successfully loaded {len(raw_df)} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at '{args.input_file}'")
        exit()
    cleaned_df = clean_data(raw_df)
    engineered_df = feature_engineer(cleaned_df)
    final_df = apply_heuristic_labels(engineered_df)
    try:
        final_df.to_csv(args.output, index=False)
        print(f"\n--- Processing Complete! ---")
        print(f"✅ Final processed data saved to '{args.output}'")
        print("\nFinal dataset preview:")
        print(final_df.head())
    except Exception as e:
        print(f"\n❌ ERROR: Could not save the file. Reason: {e}")
