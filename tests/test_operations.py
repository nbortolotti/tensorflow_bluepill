import os
import sys
import tensorflow as tf
import unittest

sys.path.insert(0, os.path.abspath('.'))

from prescription_one.main_operations import tf_operations


class TestOperations(tf.test.TestCase):
    def test_constant_tensor_pill(self):
        constant_test = tf.constant([[2., 3.], [7., 11.]])
        result = tf.constant([[9., 10.], [14., 18.]])

        obj1 = tf_operations(constant_test)

        self.assertAllEqual(result, obj1.constant_tensor_pill(obj1.tf_constant))

    def test_square_tensor_pill(self):
        constant_test = tf.constant([[2., 3.], [7., 11.]])
        result = tf.constant([[4., 9.], [49., 121.]])

        obj1 = tf_operations(constant_test)

        self.assertAllEqual(result, obj1.square_tensor_pill(obj1.tf_constant))


if __name__ == '__main__':
    unittest.main()
