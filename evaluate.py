# evaluate.py

import numpy as np
import tensorflow as tf
from config import (
    X_TEST_PATH, Y_TEST_PATH, MODEL_SAVE_PATH,
)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Loading test data...")
    try:
        X_test = np.load(X_TEST_PATH)
        y_test = np.load(Y_TEST_PATH)
        print("Test data loaded successfully.")

        #ver cuanto ocupa en memoria
        print(f"X_test shape: {X_test.shape}, size: {X_test.nbytes / (1024**2):.2f} MB")
        print(f"y_test shape: {y_test.shape}, size: {y_test.nbytes / (1024**2):.2f} MB")

        # Load the trained model
        print(f"Loading trained model from: {MODEL_SAVE_PATH}")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")

        # Evaluate the model on the test data
        print("Evaluating model on test data...")
        loss = model.evaluate(X_test, y_test, batch_size=4, verbose=0)
        print(f"Test Loss (MSE): {loss}")

        # --- Optional: Visualize some predictions vs actual ---
        print("\nGenerating sample predictions for visualization...")
        # Predict the first few examples in the test set
        num_examples_to_show = min(3, X_test.shape[0]) # Don't exceed available samples
        if num_examples_to_show > 0:
            sample_predictions = model.predict(X_test[:num_examples_to_show])

            for i in range(num_examples_to_show):
                plt.figure(figsize=(14, 6))

                # Actual Precipitation
                plt.subplot(1, 2, 1)
                # Remove the channel dimension for plotting (shape (Y, X))
                actual_frame = y_test[i, :, :, 0]
                plt.imshow(actual_frame, cmap='Blues')
                plt.title(f"Actual Precipitation (Test Sample {i+1})")
                plt.colorbar(label='Precipitation')
                plt.xlabel('X')
                plt.ylabel('Y')


                # Predicted Precipitation
                plt.subplot(1, 2, 2)
                 # Remove the channel dimension for plotting (shape (Y, X))
                predicted_frame = sample_predictions[i, :, :, 0]
                 # Ensure predicted values are not negative if precipitation should be >= 0
                predicted_frame[predicted_frame < 0] = 0
                plt.imshow(predicted_frame, cmap='Blues')
                plt.title(f"Predicted Precipitation (Test Sample {i+1})")
                plt.colorbar(label='Precipitation')
                plt.xlabel('X')
                plt.ylabel('Y')

                plt.tight_layout()
                plt.show()
        else:
            print("Not enough samples in test set to show examples.")



    except FileNotFoundError as fnf:
        print(f"Error loading data or model: {fnf}. Make sure you have run data_preprocessing.py and train_model.py.")
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")