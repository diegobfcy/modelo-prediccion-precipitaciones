import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Listar dispositivos físicos GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA Available: True")
    print("GPU Devices:", gpus)
else:
    print("CUDA Available: False")
    print("No GPU devices detected.")
