import os
import argparse
import tensorflow as tf
import csv


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def add_pill(arrays_content):
    sum_operation = tf.add(arrays_content[0], arrays_content[1], name='add')
    print(sum_operation)

def div_pill(arrays_content):
    div_operation = tf.divide(arrays_content[0], arrays_content[1], name='div')
    print(div_operation)


def read_csv_array(filename):
    path = os.path.dirname(__file__)
    full_path = path + filename

    results = []
    with open(full_path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        for row in reader:  # each row is a list
            results.append(row)

    return results


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow constants and functions')
    pa.add_argument('--operation', dest='operation', required=False, help='operation')

    args = pa.parse_args()

    #add_pill(read_csv_array('/../support/arrays.csv'))
    div_pill(read_csv_array('/../support/arrays.csv'))

