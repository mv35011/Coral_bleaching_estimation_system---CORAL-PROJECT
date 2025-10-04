import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import argparse


def train_model(data_path):
    """
    Loads data, trains a CPU-based GradientBoostingRegressor model, evaluates it,
    and saves the trained model to a file.
    """
    print("--- Starting Model Training with scikit-learn (CPU) ---")
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded {len(df)} rows from '{data_path}'")
    except FileNotFoundError:
        print(f"❌ ERROR: Processed data file not found at '{data_path}'")
        return
    features = [
        'sea_surface_temp_c', 'hotspot_c', 'degree_heating_week_c_weeks',
        'sst_anomaly_c', 'bleaching_alert_area', 'bleaching_alert_area_7d_max',
        'year', 'month', 'day_of_year', 'week_of_year'
    ]
    target = 'bleaching_risk_percent'

    X = df[features]
    y = df[target]

    print(f"\nFeatures being used for training: {features}")
    print(f"Target variable: {target}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nData split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")
    print("\nTraining the GradientBoostingRegressor model on CPU...")
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        verbose=1
    )
    gbr.fit(X_train, y_train)

    print("\n✅ Model training complete.")
    print("\n--- Model Evaluation ---")
    y_pred = gbr.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("------------------------")
    model_filename = 'coral_bleaching_model.pkl'
    try:
        joblib.dump(gbr, model_filename)
        print(f"\n✅ Trained model successfully saved to '{model_filename}'")
    except Exception as e:
        print(f"\n❌ ERROR: Could not save the model. Reason: {e}")


if __name__ == "__main__":
    input_file_path = 'coral_data_PROCESSED.csv'
    train_model(input_file_path)
    """
    parser = argparse.ArgumentParser(description="Train a coral bleaching prediction model using scikit-learn.")
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the processed input CSV file (e.g., coral_data_PROCESSED.csv)'
    )
    args = parser.parse_args()

    train_model(args.input_file)
    """

