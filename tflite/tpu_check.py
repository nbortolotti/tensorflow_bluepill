import os
import time
import argparse
import tensorflow as tf
from edgetpu.basic import edgetpu_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# if tf.test.is_gpu_available():
#    with tf.device("gpu:0"):
#        v = tf.Variable(tf.random_normal([1000, 1000]))
#        v = None  # v no longer takes up GPU memory


def measure(x, steps):
    tf.matmul(x, x)
    start = time.time()
    for i in range(steps):
        x = tf.matmul(x, x)
    _ = x.numpy()
    end = time.time()
    return end - start

t = (2000, 2000)
steps = 100


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow performance')
    pa.add_argument('--device', dest='device', required=False, help='input gpu or cpu parameter')

    args = pa.parse_args()

    #if args.device == "cpu":
    #    with tf.device("/cpu:0"):
    #        print("CPU: {} secs".format(measure(tf.random.normal(t), steps)))
    #elif args.device == "gpu":
    #    if tf.test.is_gpu_available():
    #        with tf.device("/gpu:0"):
    #            print("GPU: {} secs".format(measure(tf.random.normal(t), steps)))
    #    else:
    #        print("it's not possible to detect a GPU")
    #else:
    #    print("please define a device")
    edge_tpus = edgetpu_utils.ListEdgeTpuPaths(
        edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
    if len(edge_tpus) <= 1:
        print('This demo requires at least two Edge TPU available.')

    print("TPU: {} secs".format(measure(tf.random.normal(t), steps)))