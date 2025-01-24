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


######################################
#add in other variables 
#attributes of interest 
lidar = ["cover_2m_c", "cover_mean_c", 
          "p_mean_c", "p95_c", "p50_c",
          "p25_c", "sd_c", "rumple_c", "cov_c", "density"]
data = pd.read_csv("D:/paper2/model_outputs/data/high_pred.csv")



##########################################################

# Remove rows with missing values
data_cleaned = data.dropna()


# Separate into training and testing data
np.random.seed(123)
train, test = train_test_split(data, test_size=0.3)


#################################################
#We might remove high variables 

# for loop to build model 

i = 0  
for i in range(len(lidar)):
    #select lidar
    target_column = lidar[i]
    print(lidar[i])
    
    #seperate into training and testing data
    train_x = train.iloc[:, list(range(16, 34))]
    train_y = train[target_column]

    test_x = test.iloc[:, list(range(16, 34))]
    test_y = test[target_column]
    
    
    #create base random forest 
    # First create the base model to tune
    rf = RandomForestRegressor(n_estimators= 100, random_state= 42)
    rf.fit(train_x, train_y)
    
    ###################################################
    y_pred = rf.predict(test_x)

    y_pred_df = pd.DataFrame(y_pred, columns= ['pred'])

    r2  = r2_score(test_y, y_pred_df['pred'])
    print("R2:" , r2)
    
    #look at directory 
    model_directory = "D:/paper2/model_outputs/models_high/"
    
    #check if exists
    os.makedirs(model_directory, exist_ok=True)
    
    model_path = os.path.join(model_directory, f"{target_column}.joblib")
    
    dump(rf, model_path)
    
    ################################################################
    output_df = pd.DataFrame({
        'ob': test_y,
        'pred': y_pred
    })
    
    #look at directory 
    pred_directory = "D:/paper2/model_outputs/preds_high/"
    
    pred_path = os.path.join(pred_directory, f"{target_column}.csv")
    
    output_df.to_csv(pred_path, index=False)  # index=False avoids writing row numbers






































# Define file paths
data_path = "D:/paper2/updated_model_data/test_01_21.csv"
output_path = "D:/new_preds/cover_2m.csv"
model_path = "D:/new_models/cover_2m.joblib"

# Read the dataset
data = pd.read_csv(data_path)

# Remove rows with missing values
data_cleaned = data.dropna()

# Sample 5,000 data points (adjust if fewer rows are available)
sampled_data = data_cleaned.sample(n=5000, random_state=42)

# Split the data into training and testing sets
train, test = train_test_split(sampled_data, test_size=0.3, random_state=42)

# Define features (X) and target (y)
target_column = "cover_2m_c"  # Replace with the actual column name for the target variable
feature_columns = data.columns[13:31]  # Select columns from index 13 to 31

x_train = train[feature_columns]
y_train = train[target_column]
x_test = test[feature_columns]
y_test = test[target_column]

# Train the base Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mad = mean_absolute_error(y_test, y_pred)

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAD: {mad:.3f}")

# Save predictions to CSV
os.makedirs(os.path.dirname(output_path), exist_ok=True)
predictions_df = pd.DataFrame({
    "Index": test.index,
    "Observed": y_test,
    "Predicted": y_pred
})
predictions_df.to_csv(output_path, index=False)

# Save the model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
dump(model, model_path)

print(f"Predictions saved to {output_path}")
print(f"Model saved to {model_path}")
