import tensorflow as tf
import tensorflow_io as tfio
import sys

print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"TensorFlow I/O: {tfio.__version__}")

try:
    print("Checking tfio.audio.decode_aac...")
    print(tfio.audio.decode_aac)
    print("Success!")
except AttributeError:
    print("tfio.audio.decode_aac NOT FOUND")

try:
    print("Checking tfio.experimental.audio.decode_aac...")
    print(tfio.experimental.audio.decode_aac)
    print("Success!")
except AttributeError:
    print("tfio.experimental.audio.decode_aac NOT FOUND")


file_contents = tf.io.read_file("downloads/audioset/balanced_train_segments/__0OQemumqg.m4a")
# decode_aac returns (Time, Channels)
tf.print(111)
audio = tfio.audio.decode_aac(file_contents)