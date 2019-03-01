import os
import sys
import tensorflow as tf
from unittest import TestCase
from device_check import measure

sys.path.insert(0, os.path.abspath('.'))

class GeneralDeviceCheck(TestCase):
    def test_cpu(self):
        t = (500, 500)
        steps = 50

        result = measure(tf.random.normal(t), steps)
        self.assertLess(result, 10.)

