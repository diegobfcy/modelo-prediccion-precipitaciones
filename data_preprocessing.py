# data_preprocessing.py

import xarray as xr
import numpy as np
import os
from config import (
    DATA_PATH, PREC_VARIABLE_NAME, TIME_DIM, Y_DIM, X_DIM,
    SEQUENCE_LENGTH, TRAIN_SIZE, VAL_SIZE, PROCESSED_DATA_DIR,
    X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH,
    X_TEST_PATH, Y_TEST_PATH
)

def create_sequences(data, sequence_length):
    """
    Creates input-output sequences for the ConvLSTM model.

    Args:
        data (np.ndarray): The input data array of shape (T, Y, X).
        sequence_length (int): The number of time steps in each input sequence.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Input sequences of shape (samples, sequence_length, Y, X, 1).
            - y (np.ndarray): Output next frames of shape (samples, Y, X, 1).
    """
    X, y = [], []
    # Ensure data has a channel dimension for ConvLSTM (even if it's 1)
    data_with_channel = np.expand_dims(data, axis=-1) # Shape (T, Y, X, 1)

    for i in range(len(data) - sequence_length):
        # Get the sequence of past frames
        seq_x = data_with_channel[i : (i + sequence_length)]
        # Get the next frame (the target)
        seq_y = data_with_channel[i + sequence_length]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    print(f"Loading data from: {DATA_PATH}")
    try:
        # Load the NetCDF file
        ds = xr.open_dataset(DATA_PATH, decode_times=False) # Use decode_times=False as in your example

        # Extract the precipitation data as a NumPy array
        if PREC_VARIABLE_NAME not in ds.variables:
            raise ValueError(f"Variable '{PREC_VARIABLE_NAME}' not found in the dataset.")

        # Ensure the data is in the desired order (T, Y, X)
        # xarray should handle this based on coordinate names, but good to be aware
        precipitation_data = ds[PREC_VARIABLE_NAME].transpose(TIME_DIM, Y_DIM, X_DIM).values

        print(f"Original data shape: {precipitation_data.shape}")

        # --- Data Preprocessing Steps ---
        # 1. Handle potential NaNs (Simple fill with 0 for demonstration)
        #    You might need a more sophisticated method depending on your data
        if np.isnan(precipitation_data).any():
            print("Warning: NaN values found. Filling with 0.")
            precipitation_data = np.nan_to_num(precipitation_data, nan=0.0)

        # 2. Optional: Data Scaling (Add scaling here if needed)
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # # Reshape for scaler (T*Y*X, 1) -> fit -> reshape back (T, Y, X)
        # original_shape = precipitation_data.shape
        # precipitation_data_scaled = scaler.fit_transform(precipitation_data.reshape(-1, 1)).reshape(original_shape)
        # precipitation_data = precipitation_data_scaled
        # # Save the scaler object if you need to inverse transform predictions later
        # import joblib
        # joblib.dump(scaler, PROCESSED_DATA_DIR + 'scaler.pkl')


        # Create sequences for the ConvLSTM model
        print(f"Creating sequences with length: {SEQUENCE_LENGTH}")
        X, y = create_sequences(precipitation_data, SEQUENCE_LENGTH)

        print(f"Input sequences shape (X): {X.shape}")
        print(f"Output frames shape (y): {y.shape}")

        # Split data into training, validation, and testing sets
        # The number of samples available after creating sequences is X.shape[0]
        total_samples = X.shape[0]
        if TRAIN_SIZE + VAL_SIZE > total_samples: # Check if sizes exceed total samples
             raise ValueError("TRAIN_SIZE + VAL_SIZE exceeds the total number of samples available.")
        if TRAIN_SIZE + VAL_SIZE == total_samples and total_samples > 0:
             print("Warning: No samples left for the test set.")
        elif TRAIN_SIZE + VAL_SIZE >= total_samples and total_samples == 0:
             raise ValueError("No samples could be created with the given sequence length.")


        train_end = TRAIN_SIZE
        val_end = train_end + VAL_SIZE
        # test_end is the end of the available samples

        print(f"Splitting data: Train={train_end} samples, Val={val_end - train_end} samples, Test={total_samples - val_end} samples") # Corrected VAL_end to val_end

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:] # Slice till the end for test set

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


        # Create directory for processed data if it doesn't exist
        if not os.path.exists(PROCESSED_DATA_DIR):
            os.makedirs(PROCESSED_DATA_DIR)
            print(f"Created directory: {PROCESSED_DATA_DIR}")

        # Save the processed data arrays
        print("Saving processed data...")
        np.save(X_TRAIN_PATH, X_train)
        np.save(Y_TRAIN_PATH, y_train)
        np.save(X_VAL_PATH, X_val)
        np.save(Y_VAL_PATH, y_val)
        np.save(X_TEST_PATH, X_test)
        np.save(Y_TEST_PATH, y_test)
        print("Processed data saved successfully.")

        ds.close() # Close the dataset

    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
    except ValueError as ve:
        print(f"Data processing error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")