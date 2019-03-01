import os
import sys
import tensorflow as tf
from unittest import TestCase

sys.path.insert(0, os.path.abspath('.'))

from device_check import measure


class GeneralDeviceCheck(TestCase):
    def test_cpu(self):
        t = (500, 500)
        steps = 50

        result = measure(tf.random.normal(t), steps)
        self.assertLess(result, 10.)
