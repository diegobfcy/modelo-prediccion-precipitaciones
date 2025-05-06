import os
import sys

# Agrega las rutas correctas a cudart64_110.dll y cudnn64_8.dll (ajusta según sea necesario)
os.add_dll_directory(r"C:\Users\HP\anaconda3\envs\tf_cuda_env\Library\bin")

import tensorflow as tf
