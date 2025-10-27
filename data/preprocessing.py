# xgboost-api/data/preprocessing.py
import pandas as pd
import numpy as np # Import numpy just in case, though not used here directly

# List ALL possible intersection names from your training data
ALL_INTERSECTION_NAMES = [
    'Intersection_100 Feet Road', 'Intersection_Anil Kumble Circle', 'Intersection_Ballari Road',
    'Intersection_CMH Road', 'Intersection_Hebbal Flyover', 'Intersection_Hosur Road',
    'Intersection_ITPL Main Road', 'Intersection_Jayanagar 4th Block', 'Intersection_Marathahalli Bridge',
    'Intersection_Sarjapur Road', 'Intersection_Silk Board Junction', 'Intersection_Sony World Junction',
    'Intersection_South End Circle', 'Intersection_Trinity Circle', 'Intersection_Tumkur Road',
    'Intersection_Yeshwanthpur Circle'
    # Add any others if needed
]

# --- REVERTED EXPECTED FEATURES ---
# Use simple time features + one-hot encoded intersections
EXPECTED_MODEL_FEATURES = [
    'hour', 'day_of_week', 'month', 'is_weekend'
] + ALL_INTERSECTION_NAMES
# --- END REVERT ---


def preprocess_data(input_df):
    """
    Applies SIMPLE feature engineering (raw time + OHE).
    Input: A pandas DataFrame with 'DateTime' and 'JunctionName' columns.
    Output: A pandas DataFrame ready for the default XGBoost model.
    """
    df = input_df.copy()

    # 1. Convert 'DateTime'
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    if df['DateTime'].isnull().any():
        print("Warning: Could not parse DateTime.")
        df.fillna(pd.Timestamp('1970-01-01'), inplace=True) # Simple fill

    # 2. Extract original time features
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # --- Removed Cyclical Features ---

    # 3. One-Hot Encode the 'JunctionName'
    if 'JunctionName' not in df.columns:
        raise ValueError("Input data must contain a 'JunctionName' column.")

    for col_name in ALL_INTERSECTION_NAMES:
        df[col_name] = 0 # Initialize all OHE columns to 0

    junction_col_name = df['JunctionName'].iloc[0] # Get name from input
    if junction_col_name in ALL_INTERSECTION_NAMES:
        df[junction_col_name] = 1 # Set the correct OHE column to 1
    else:
        print(f"Warning: Received unknown JunctionName '{junction_col_name}'.")

    # 4. Return only the expected features in the correct order
    missing_cols = [col for col in EXPECTED_MODEL_FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Preprocessing failed. Missing expected feature columns: {missing_cols}")

    final_features_df = df[EXPECTED_MODEL_FEATURES]

    return final_features_df

# --- Example Usage (for testing this script locally) ---
if __name__ == '__main__':
    sample_data = {
        'DateTime': ['2025-10-18 16:30:00'],
        'JunctionName': ['Intersection_Trinity Circle']
    }
    sample_df = pd.DataFrame(sample_data)
    processed_df = preprocess_data(sample_df)
    print("Sample processed data for API (Simple Features):")
    print(processed_df)
    print("\nColumns:", processed_df.columns.tolist())
    print("\nShape:", processed_df.shape)