from IPython import get_ipython
get_ipython().magic('reset -sf')

#load packages 
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from joblib import dump


lidar = ["cover_2m_c", "cover_mean_c", 
         "p_mean_c", "p95_c", "p50_c",
         "p25_c", "sd_c", "rumple_c", "cov_c", "density"]

data = pd.read_csv("D:/paper2/high_low_rasters/update_structure_csv/model_data/subbec_modeldata_01_28.csv")

# List of unique subbec values
unique_subbecs = data['subbec'].unique()

# Loop through each unique subbec value
for subbec in unique_subbecs:
    print(f"Processing subbec: {subbec}")
    
    # Filter data for the current subbec
    subset_data = data[data['subbec'] == subbec]
    
    # Remove rows with missing values
    subset_data_cleaned = subset_data.dropna()

    # Skip if there's not enough data for training/testing
    if len(subset_data_cleaned) < 10:  # Adjust threshold as needed
        print(f"Not enough data for subbec: {subbec}. Skipping...")
        continue

    # Split into training and testing data
    np.random.seed(123)
    train, test = train_test_split(subset_data_cleaned, test_size=0.3)

    # Iterate through each LiDAR variable as the target
    for target_column in lidar:
        print(f"  Training model for target: {target_column}")
        
        # Separate features (X) and target (y)
        train_x = train.iloc[:, list(range(16, 34))]  # Replace with dynamic column selection if needed
        train_y = train[target_column]

        test_x = test.iloc[:, list(range(16, 34))]
        test_y = test[target_column]

        # Create and fit the Random Forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_x, train_y)
        
        # Predict on the test set
        y_pred = rf.predict(test_x)

        # Calculate R² score
        r2 = r2_score(test_y, y_pred)
        print(f"    R² for {target_column}: {r2}")

        # Save the model
        model_directory = f"D:/paper2/model_outputs/models_high/{subbec}/"
        os.makedirs(model_directory, exist_ok=True)
        model_path = os.path.join(model_directory, f"{target_column}.joblib")
        dump(rf, model_path)

        # Save predictions
        pred_directory = f"D:/paper2/model_outputs/preds_high/{subbec}/"
        os.makedirs(pred_directory, exist_ok=True)
        pred_path = os.path.join(pred_directory, f"{target_column}.csv")
        
        # Create output DataFrame and save as CSV
        output_df = pd.DataFrame({
            'ob': test_y,
            'pred': y_pred
        })
        output_df.to_csv(pred_path, index=False)

print("Processing complete!")
