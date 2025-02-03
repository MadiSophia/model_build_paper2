import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from joblib import dump

# Attributes of interest 
lidar = ["cover_2m_c", "p95_c", "sd_c"]

# Load data
data = pd.read_csv("D:/paper2/high_low_rasters/1_31_clean_model/model_data/model_data16.csv")

# Extract unique subbec groups
data['subbec_group'] = data['subbec'].apply(lambda x: x.split('-')[-1])
# Count occurrences of each subbec_group
group_counts = data['subbec_group'].value_counts()

# Filter out groups with fewer than 100 occurrences
valid_groups = group_counts[group_counts >= 100].index

# Get the unique subbec groups that meet the condition
filtered_groups = data[data['subbec_group'].isin(valid_groups)]

# Get the list of valid unique groups
unique_groups = filtered_groups['subbec_group'].unique()

# Define variable columns
var = list(range(19, 37))

# Remove rows with missing values
data_cleaned = data.dropna()

# Hyperparameter tuning for KNN (try different n_neighbors)
neighbors_range = range(1, 21)  # Try values from 1 to 20

# Train separate models for each subbec group
for group in unique_groups:
    print(f"Training model for {group}")
    group_data = data_cleaned[data_cleaned['subbec_group'] == group]
    
    # Split into training and testing data
    np.random.seed(123)
    train, test = train_test_split(group_data, test_size=0.3)

    # Prepare training and testing data
    train_x = train.iloc[:, var]
    test_x = test.iloc[:, var]
    train_y = train[lidar]
    test_y = test[lidar]

    # --- Random Forest for Feature Selection ---
    rf = RandomForestRegressor(n_estimators=100, random_state=123)
    rf.fit(train_x, train_y)

    # Get feature importances
    feature_importances = rf.feature_importances_

    # Sort feature importances and select top features based on cumulative importance
    important_features_idx = np.argsort(feature_importances)[::-1]  # Sort in descending order
    total_importance = np.sum(feature_importances)
    
    # Try different thresholds based on cumulative importance
    best_threshold = None
    best_score = -np.inf
    selected_var_names_best = []

    thresholds = np.linspace(0.05, 1.0, 20)  # Adjust from 5% to 100% of total importance
    for threshold in thresholds:
        cumulative_importance = 0
        selected_var = []

        # Select features whose cumulative importance exceeds the threshold
        for i in important_features_idx:
            cumulative_importance += feature_importances[i]
            selected_var.append(i)
            if cumulative_importance >= threshold * total_importance:
                break

        selected_var_names = train_x.columns[selected_var].to_list()

        # Cross-validate the model with selected features
        train_x_selected = train_x[selected_var_names]
        test_x_selected = test_x[selected_var_names]

        # Evaluate KNN performance for different n_neighbors
        best_n_neighbors = 1
        best_knn_score = -np.inf

        for n_neighbors in neighbors_range:
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            score = np.mean(cross_val_score(knn, train_x_selected, train_y, cv=5, scoring='r2'))  # Use R² score for evaluation
            
            if score > best_knn_score:
                best_knn_score = score
                best_n_neighbors = n_neighbors

        # If this threshold gives better performance, update the best threshold and features
        if best_knn_score > best_score:
            best_score = best_knn_score
            best_threshold = threshold
            selected_var_names_best = selected_var_names

    print(f"Best threshold for {group}: {best_threshold}")
    print(f"Best selected features for {group}: {selected_var_names_best}")

    # Final KNN model using the selected features and best n_neighbors
    train_x_selected = train_x[selected_var_names_best]
    test_x_selected = test_x[selected_var_names_best]

    # Train the KNN model with best n_neighbors
    knn = KNeighborsRegressor(n_neighbors=best_n_neighbors)
    
    print(f"N used {group}: {best_n_neighbors}")
    knn.fit(train_x_selected, train_y)

    # Make predictions
    y_pred = knn.predict(test_x_selected)

    # Evaluate performance
    r2_scores = {col: r2_score(test_y[col], y_pred[:, i]) for i, col in enumerate(lidar)}
    print("R2 scores:", r2_scores)

    # Save the model
    model_directory = f"D:/paper2/high_low_rasters/2_03_knn/models/{group}/"
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, f"knn_model_{group}.joblib")
    dump(knn, model_path)

    # Save predictions
    output_df = pd.DataFrame(y_pred, columns=lidar)
    output_df.insert(0, "ob_index", test.index)

    pred_directory = f"D:/paper2/high_low_rasters/2_03_knn/preds_csv/{group}/"
    os.makedirs(pred_directory, exist_ok=True)
    pred_path = os.path.join(pred_directory, f"knn_predictions_{group}.csv")
    output_df.to_csv(pred_path, index=False)

