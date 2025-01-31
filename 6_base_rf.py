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



          
data = pd.read_csv("D:/paper2/high_low_rasters/1_31_clean_model/model_data/model_data16.csv")




# Create a copy of your dataset
filtered_d_model_no_outliers = data.copy()


    
#data_df = data.sample(n=10000, random_state=42) 
#change dependent on parameters 
var = list(range(19, 37) )

#model 8 is optimized , 10 is really similar

##########################################################

# Remove rows with missing values
data_cleaned = filtered_d_model_no_outliers.dropna()


# Separate into training and testing data
np.random.seed(123)
#sampled_rows = data_cleaned.sample(n=10000, random_state=42)
train, test = train_test_split(data_cleaned, test_size=0.3)


#################################################
#We might remove high variables 

# for loop to build model 
i =0
 
for i in range(len(lidar)):
    #select lidar
    target_column = lidar[i]
    print(lidar[i])
    
    #seperate into training and testing data
    train_x = train.iloc[:, var]
    train_y = train[target_column]

    test_x = test.iloc[:, var]
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
    
    #look at directory  change if neeeded 
    model_directory = "D:/paper2/high_low_rasters/1_31_clean_model/models/"
    
    #check if exists
    os.makedirs(model_directory, exist_ok=True)
    
    model_path = os.path.join(model_directory, f"{target_column}.joblib")
    
    dump(rf, model_path)
    
    ################################################################
    output_df = pd.DataFrame({
        'ob': test_y,
        'pred': y_pred
    })
    
    #look at directory  change if needed 
    pred_directory = "D:/paper2/high_low_rasters/1_31_clean_model/preds_csv/"
    
    pred_path = os.path.join(pred_directory, f"{target_column}.csv")
    
    output_df.to_csv(pred_path, index=False)  # index=False avoids writing row numbers






































