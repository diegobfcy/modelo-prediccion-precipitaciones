# predict.py

import numpy as np
import tensorflow as tf
import xarray as xr
import sys # Import sys to potentially read command line arguments
from config import (
    DATA_PATH, PREC_VARIABLE_NAME, TIME_DIM, Y_DIM, X_DIM,
    SEQUENCE_LENGTH, MODEL_SAVE_PATH, PROCESSED_DATA_DIR # PROCESSED_DATA_DIR needed if you saved a scaler
)
import matplotlib.pyplot as plt
import os # Needed for scaler loading

# Optional: Load the scaler if you used one during preprocessing
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# try:
#     scaler = joblib.load(PROCESSED_DATA_DIR + 'scaler.pkl')
#     print("Scaler loaded.")
# except FileNotFoundError:
#     print("Warning: Scaler file not found. Predictions will be in the scaled range if scaling was applied.")
#     scaler = None


if __name__ == "__main__":

    # --- Define the target time index to predict ---
    # You can set this manually, or read it from command line arguments
    # Example: Predict the month at index 530 (the one after the last data point if data is 0-529)
    # Or predict a month within the dataset, e.g., index 500 (this is like predicting a test sample)

    # Option 1: Set the target index manually here
    TARGET_TIME_INDEX = 530 # Predict the month after the last one (if data is 0-529)
    # TARGET_TIME_INDEX = 500 # Predict the month at index 500 (requires data up to index 499)


    # Option 2: Read the target index from command line arguments
    # You would run the script like: python predict.py 530
    # try:
    #     if len(sys.argv) > 1:
    #         TARGET_TIME_INDEX = int(sys.argv[1])
    #         print(f"Predicting precipitation for time index: {TARGET_TIME_INDEX}")
    #     else:
    #         print("No target time index provided. Predicting the month after the last data point.")
    #         # Load data info to get the number of time steps
    #         ds_info = xr.open_dataset(DATA_PATH, decode_times=False)
    #         num_total_time_steps = len(ds_info[TIME_DIM])
    #         ds_info.close()
    #         TARGET_TIME_INDEX = num_total_time_steps
    #         print(f"Defaulting to predicting time index: {TARGET_TIME_INDEX} (the one after the last data point)")
    # except ValueError:
    #      print("Error: Please provide a valid integer for the target time index.")
    #      sys.exit(1)
    # except FileNotFoundError:
    #      print(f"Error: Data file not found at {DATA_PATH}. Cannot determine last time index.")
    #      sys.exit(1)
    # ---------------------------------------------


    print(f"Loading data from: {DATA_PATH}")
    try:
        # Load the entire dataset needed for prediction
        ds = xr.open_dataset(DATA_PATH, decode_times=False)

        if PREC_VARIABLE_NAME not in ds.variables:
            raise ValueError(f"Variable '{PREC_VARIABLE_NAME}' not found in the dataset.")

        # Ensure data is in (T, Y, X) order
        precipitation_data = ds[PREC_VARIABLE_NAME].transpose(TIME_DIM, Y_DIM, X_DIM).values
        num_total_data_steps = precipitation_data.shape[0]

        # --- Apply the same preprocessing as training to the historical sequence ---
        # Handle NaNs if preprocessing did
        if np.isnan(precipitation_data).any():
             print("Warning: NaN values found in full dataset. Filling with 0.")
             precipitation_data = np.nan_to_num(precipitation_data, nan=0.0)

        # Optional: Apply scaling if you used one
        # if scaler is not None:
        #     original_shape = precipitation_data.shape
        #     precipitation_data = scaler.transform(precipitation_data.reshape(-1, 1)).reshape(original_shape)


        # Determine the slice needed for the input sequence
        # The input sequence covers time steps from (TARGET_TIME_INDEX - SEQUENCE_LENGTH) to (TARGET_TIME_INDEX - 1)
        start_index = TARGET_TIME_INDEX - SEQUENCE_LENGTH
        end_index = TARGET_TIME_INDEX # This is exclusive slicing, so it goes up to TARGET_TIME_INDEX - 1

        # --- Validate the target time index and required historical data ---
        if TARGET_TIME_INDEX > num_total_data_steps:
             # You can predict the next step (index == num_total_data_steps), but not steps beyond that
             if TARGET_TIME_INDEX == num_total_data_steps and start_index >= 0:
                  print(f"Predicting the next time step (index {TARGET_TIME_INDEX}) after the end of the data (index {num_total_data_steps - 1}).")
             else:
                 raise ValueError(f"Target time index ({TARGET_TIME_INDEX}) is beyond the available data ({num_total_data_steps}). Cannot predict this far.")

        if start_index < 0:
            raise ValueError(f"Not enough preceding data to create a sequence of length {SEQUENCE_LENGTH} for target index {TARGET_TIME_INDEX}. Need data from index {start_index} onwards, but data starts at index 0.")


        # Get the required historical sequence
        historical_sequence = precipitation_data[start_index : end_index]

        # Reshape the historical sequence to match the model's input shape (1, sequence_length, Y, X, 1)
        # Add batch dimension (1) and channel dimension (1)
        input_sequence_for_prediction = np.expand_dims(historical_sequence, axis=0) # Add batch dimension
        input_sequence_for_prediction = np.expand_dims(input_sequence_for_prediction, axis=-1) # Add channel dimension

        print(f"Shape of the input sequence for prediction: {input_sequence_for_prediction.shape}")
        print(f"Using data from time index {start_index} to {end_index - 1} to predict time index {TARGET_TIME_INDEX}.")


        # Load the trained model
        print(f"Loading trained model from: {MODEL_SAVE_PATH}")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")

        # Make the prediction for the next time step
        print(f"Making prediction for time index {TARGET_TIME_INDEX}...")
        next_month_prediction_scaled = model.predict(input_sequence_for_prediction)

        # The prediction is likely in the shape (1, Y, X, 1)
        # Remove batch and channel dimensions
        next_month_prediction = next_month_prediction_scaled[0, :, :, 0]

        # --- Inverse scale the prediction if a scaler was used ---
        # if scaler is not None:
        #     # Reshape for inverse transform (Y*X, 1) -> inverse transform -> reshape back (Y, X)
        #     spatial_shape = next_month_prediction.shape
        #     next_month_prediction = scaler.inverse_transform(next_month_prediction.reshape(-1, 1)).reshape(spatial_shape)

        # Ensure predicted values are not negative if precipitation should be >= 0
        next_month_prediction[next_month_prediction < 0] = 0


        print(f"Prediction complete for time index {TARGET_TIME_INDEX}.")

        # Visualize the prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(next_month_prediction, cmap='Blues')
        plt.title(f"Predicted Precipitation for Time Index {TARGET_TIME_INDEX}")
        plt.colorbar(label='Precipitation')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        ds.close() # Close the dataset

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH} or model file not found at {MODEL_SAVE_PATH}.")
    except ValueError as ve:
        print(f"Prediction error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")