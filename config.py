# config.py

# --- Data Configuration ---
DATA_PATH = "C:/Users/HP/Documents/InteligenciaArtificial/modelo-prediccion-prueba/data/data.nc"
PREC_VARIABLE_NAME = 'Prec'
# Assuming time dimension is named 'T', Y is 'Y', X is 'X' based on your output
TIME_DIM = 'T'
Y_DIM = 'Y'
X_DIM = 'X'

# --- Preprocessing Configuration ---
SEQUENCE_LENGTH = 12  # Number of past months to use for prediction
TRAIN_SIZE = 400      # Number of initial time steps for training (before considering sequence_length)
VAL_SIZE = 50         # Number of time steps for validation
# Test size is calculated automatically

# --- Model Configuration ---
# ConvLSTM layer parameters
FILTERS = 16
KERNEL_SIZE = (3, 3)
ACTIVATION = 'relu'
PADDING = 'same'
# Output layer parameters
OUTPUT_KERNEL_SIZE = (1, 1) # Common for 1x1 convolution to get the final channel

# --- Training Configuration ---
LOSS_FUNCTION = 'mse' # Mean Squared Error
OPTIMIZER = 'adam'
BATCH_SIZE = 1
EPOCHS = 50 # You may need to adjust this based on training progress

# --- Paths for saving/loading ---
PROCESSED_DATA_DIR = './processed_data/'
X_TRAIN_PATH = PROCESSED_DATA_DIR + 'X_train.npy'
Y_TRAIN_PATH = PROCESSED_DATA_DIR + 'y_train.npy'
X_VAL_PATH = PROCESSED_DATA_DIR + 'X_val.npy'
Y_VAL_PATH = PROCESSED_DATA_DIR + 'y_val.npy'
X_TEST_PATH = PROCESSED_DATA_DIR + 'X_test.npy'
Y_TEST_PATH = PROCESSED_DATA_DIR + 'y_test.npy'
MODEL_SAVE_PATH = './trained_model/precipitation_convlstm_model.keras' # Use .keras extension