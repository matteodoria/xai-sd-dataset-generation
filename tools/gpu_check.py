import sys
import tensorflow as tf

print("python :", sys.executable)
print("tf     :", tf.__version__)
print("GPU    :", tf.config.list_physical_devices('GPU'))