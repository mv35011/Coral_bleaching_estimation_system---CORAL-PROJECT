import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os


PROCESSED_DATA_PATH = 'app/coral_data_PROCESSED.csv'
NEW_MODEL_PATH = 'app/coral_bleaching_model.pkl'

def retrain_model(data_path, model_path):
    """
    Loads data, re-trains the model in the current environment,
    and saves the compatible model file.
    """
    print("--- Starting Model Re-Training for Compatibility ---")

    # 1. Load the processed data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded {len(df)} rows from '{data_path}'")
    except FileNotFoundError:
        print(f"❌ ERROR: Processed data file not found at '{data_path}'")
        print("Ensure 'coral_data_PROCESSED.csv' is inside the 'app' folder.")
        return

    # 2. Define Features (X) and Target (y)
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

    # 3. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nData split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # 4. Initialize and Train the Model (using the exact method as before)
    print("\nTraining the GradientBoostingRegressor model on CPU...")
    # These are the same fast parameters from the original script
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        n_iter_no_change=10,  # Stop if no improvement
        verbose=1
    )
    
    gb_model.fit(X_train, y_train)

    print("\n✅ Model training complete.")

    # 5. Evaluate the model (so you can confirm it's the same)
    print("\n--- Model Evaluation ---")
    y_pred = gb_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("------------------------")

    # 6. Save the new, compatible model
    try:
        joblib.dump(gb_model, model_path)
        print(f"\n✅ New, compatible model successfully saved to '{model_path}'")
    except Exception as e:
        print(f"\n❌ ERROR: Could not save the model. Reason: {e}")

if __name__ == "__main__":
    retrain_model(PROCESSED_DATA_PATH, NEW_MODEL_PATH)
