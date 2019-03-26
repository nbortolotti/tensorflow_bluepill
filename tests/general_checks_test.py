import os
import sys
import tensorflow as tf
from unittest import TestCase

sys.path.insert(0, os.path.abspath('.'))

from device_check import measure
from prescription_one.tensor_scalars import add_pill


class GeneralDeviceCheck(TestCase):
    def test_cpu(self):
        result = measure(tf.random.normal((500, 500)), 50)
        self.assertLess(result, 10.)


class general_scalars(TestCase):
    def test_add(self):
        self.assertEqual(tf.print(add_pill(3, 5)), tf.print(tf.add(8, 0)))
