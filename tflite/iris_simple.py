import numpy as np
#import tensorflow as tf
import tflite_runtime.interpreter as tflite


# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
#input_shape = input_details[0]['shape']
new_specie = np.array([7.9,3.8,6.4,2.0])
input_data = np.array(np.expand_dims(new_specie, axis=0), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)