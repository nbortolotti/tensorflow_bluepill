import tensorflow as tf

#converter = tf.lite.TFLiteConverter.from_saved_model("folder pbtxt & pb")
#tflite_model = converter.convert()


converter = tf.lite.TFLiteConverter.from_keras_model_file('prescription_three/iris_model.h5')
tfmodel = converter.convert()
open ("iris_lite.tflite" , "wb") .write(tfmodel)