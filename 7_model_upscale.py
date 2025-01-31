import os
import rasterio
import os
import rasterio
import numpy as np
from joblib import load

#from model 11 

# Load the trained model
model = load("D:/paper2/high_low_rasters/1_31_clean_model/models/cover_2m_c.joblib")

# Define input and output directories
input_dir = "D:/paper2/stack/"  # Directory containing input rasters
output_dir = "D:/paper2/high_low_rasters/1_31_clean_model/preds_raster/cover_2m/"  # Directory to save output rasters
os.makedirs(output_dir, exist_ok=True)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        # Extract the year from the filename
        try:
            year = int(filename.split("beast")[1][:4])  # Assumes 'beastYYYY-MM-DD.tif'
        except (IndexError, ValueError):
            print(f"Skipping {filename}: unable to parse year.")
            continue

        # Skip files before 2013
        if year < 2013:
            print(f"Skipping {filename}: year {year} is before 2013.")
            continue

        raster_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred.tif")

        # Skip if the output file already exists
        if os.path.exists(output_path):
            print(f"Skipping {filename}, output file already exists.")
            continue

        try:
            with rasterio.open(raster_path) as src:
                # Read raster data and metadata
                raster_data = src.read()  # Shape: (bands, rows, cols)
                profile = src.profile  # Metadata (CRS, transform, etc.)
                nodata_value = profile.get('nodata', None)

            # Get dimensions and reshape for model input
            n_bands, n_rows, n_cols = raster_data.shape
            feature_matrix = raster_data.reshape(n_bands, -1).T  # Shape: (pixels, bands)

            # Mask no-data values
            if nodata_value is not None:
                valid_mask = ~np.any(raster_data == nodata_value, axis=0).flatten()
                valid_features = feature_matrix[valid_mask]
            else:
                valid_features = feature_matrix
                valid_mask = np.ones(feature_matrix.shape[0], dtype=bool)

            # Predict using the model
            elev95_pred = model.predict(valid_features)

            # Reinsert predictions into the raster shape
            full_predictions = np.full(feature_matrix.shape[0], np.nan)
            full_predictions[valid_mask] = elev95_pred
            result = full_predictions.reshape(n_rows, n_cols)

            # Update profile for single-band output
            profile.update(count=1, dtype=np.float32, nodata=np.nan)

            # Write only the first band to the output
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(result.astype(np.float32), 1)  # Write the first band

            print(f"Processed {filename}, saved to {output_path}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
