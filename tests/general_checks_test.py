import os
import sys
import tensorflow as tf
from unittest import TestCase

sys.path.insert(0, os.path.abspath('.'))

from device_check import measure
from prescription_one.tensor_scalars import add_pill


class GeneralDeviceCheck(TestCase):
    def test_cpu(self):
        t = (500, 500)
        steps = 50

        result = measure(tf.random.normal(t), steps)
        self.assertLess(result, 10.)


class general_scalars(TestCase):
    def test_add(self):
        x = 3
        y = 5

        result = add_pill(x, y)

        self.assertTrue(tf.assert_equal(result, 8))
