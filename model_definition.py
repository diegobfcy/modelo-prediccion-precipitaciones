# model_definition.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization
from config import (
    SEQUENCE_LENGTH, Y_DIM, X_DIM, FILTERS, KERNEL_SIZE,
    ACTIVATION, PADDING, OUTPUT_KERNEL_SIZE,DATA_PATH
)

def build_convlstm_model(input_shape):
    """
    Builds a ConvLSTM model for precipitation prediction.

    Args:
        input_shape (tuple): The shape of the input sequences
                             (sequence_length, Y, X, channels).

    Returns:
        tf.keras.models.Sequential: The defined Keras model.
    """
    model = Sequential()

    # Input layer expects shape (samples, sequence_length, Y, X, channels)
    # input_shape is (sequence_length, Y, X, channels)
    model.add(ConvLSTM2D(filters=FILTERS,
                         kernel_size=KERNEL_SIZE,
                         activation=ACTIVATION,
                         padding=PADDING,
                         return_sequences=True, # Return sequences for the next ConvLSTM layer
                         input_shape=input_shape))
    model.add(BatchNormalization())

    # You can add more ConvLSTM layers if needed
    model.add(ConvLSTM2D(filters=FILTERS, # Using same filters, can be changed
                         kernel_size=KERNEL_SIZE, # Using same kernel size, can be changed
                         activation=ACTIVATION,
                         padding=PADDING,
                         return_sequences=False)) # Only return the output of the last time step
    model.add(BatchNormalization())

    # Add layers to output the final predicted grid
    # Use a 1x1 convolution to get the single precipitation channel
    model.add(Conv2D(filters=1,
                     kernel_size=OUTPUT_KERNEL_SIZE,
                     activation='linear', # Linear activation for continuous output
                     padding=PADDING))

    return model

if __name__ == '__main__':
    # Example of how to use the function
    # Assuming input data will have shape (samples, SEQUENCE_LENGTH, 198, 133, 1)
    # So the input_shape for the first layer is (SEQUENCE_LENGTH, 198, 133, 1)
    # You need the spatial dimensions (Y, X) from your data
    try:
        # A way to get spatial dimensions without loading all data again, if config has them
        # Or load a small part of the data
        import xarray as xr
        ds_info = xr.open_dataset(DATA_PATH, decode_times=False)
        y_dim_size = len(ds_info[Y_DIM])
        x_dim_size = len(ds_info[X_DIM])
        ds_info.close()

        input_shape = (SEQUENCE_LENGTH, y_dim_size, x_dim_size, 1) # 1 channel for precipitation
        model = build_convlstm_model(input_shape)
        model.summary()
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Cannot determine spatial dimensions for model summary.")
    except Exception as e:
        print(f"An error occurred while building the model summary: {e}")