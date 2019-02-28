import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def checking_eager():
    check = tf.executing_eagerly()
    print(check)


# functions
# checking_eager()

