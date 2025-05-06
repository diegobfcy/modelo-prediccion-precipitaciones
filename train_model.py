# train_model.py
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_definition import build_convlstm_model
from config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, X_VAL_PATH, Y_VAL_PATH,
    MODEL_SAVE_PATH, LOSS_FUNCTION, OPTIMIZER, BATCH_SIZE, EPOCHS,
    SEQUENCE_LENGTH, Y_DIM, X_DIM, DATA_PATH # Needed for input shape
)
import xarray as xr # Needed to get Y and X dimensions

if __name__ == "__main__":
    print("Loading processed data...")
    try:
        X_train = np.load(X_TRAIN_PATH)
        y_train = np.load(Y_TRAIN_PATH)
        X_val = np.load(X_VAL_PATH)
        y_val = np.load(Y_VAL_PATH)
        print("Processed data loaded successfully.")

        # Get the spatial dimensions from the original data for input shape
        ds_info = xr.open_dataset(DATA_PATH, decode_times=False)
        y_dim_size = len(ds_info[Y_DIM])
        x_dim_size = len(ds_info[X_DIM])
        ds_info.close()

        input_shape = (SEQUENCE_LENGTH, y_dim_size, x_dim_size, 1) # (sequence_length, Y, X, channels)

        # Build the model
        print("Building the ConvLSTM model...")
        model = build_convlstm_model(input_shape)
        model.summary()

        # Compile the model
        model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER)
        print(f"Model compiled with loss='{LOSS_FUNCTION}' and optimizer='{OPTIMIZER}'.")

        # Define callbacks
        # EarlyStopping to stop training when validation loss stops improving
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        # ModelCheckpoint to save the best model based on validation loss
        model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)

        # Create directory for saving the model if it doesn't exist
        model_dir = os.path.dirname(MODEL_SAVE_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created directory: {model_dir}")

        # Train the model
        print("Starting model training...")
        history = model.fit(X_train, y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, model_checkpoint])

        print("Training finished.")
        print(f"Best model saved to: {MODEL_SAVE_PATH}")

        # Optional: Plot training history
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss during Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError as fnf:
        print(f"Error loading processed data: {fnf}. Make sure you ran data_preprocessing.py first.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")