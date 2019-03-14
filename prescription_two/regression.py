from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers


# path = os.path.dirname(__file__)
# full_path = path + '/../support/heart.csv'

# full_path = os.path.dirname(os.getcwd() + '/support/heart.csv')

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()


def extract_information():
    path = os.path.dirname(os.getcwd())
    full_path = path + '/support/winequality-red.csv'

    column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

    raw_dataset = pd.read_csv(full_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=",", skipinitialspace=True)

    return raw_dataset


# dataset = raw_dataset.copy()

# ds_view = dataset.tail()
# print(ds_view)

# print(dataset.isna().sum())


def plot_analysis_dataset(train_dataset):
    sns.pairplot(train_dataset[["fixed acidity", "chlorides", "citric acid"]], diag_kind="kde")
    plt.show()


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


# example_result

# Display training progress by printing a single dot for each completed epoch
class PrintWait(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('*', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [fixed acidity]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$fixed acidity^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


def train_model(normed_train_data, train_labels, hist):
    EPOCHS = 1000

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintWait()])

    if hist:
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        plot_history(history)

    # return history


# history = model.fit(
#   normed_train_data, train_labels,
#   epochs=EPOCHS, validation_split = 0.2, verbose=0,
#   callbacks=[PrintDot()])


def evaluate_model(normalized_data, labels):
    loss, mae, mse = model.evaluate(normalized_data, labels, verbose=0)
    print("Testing set Mean Abs Error: {:5.2f} fixed acidity".format(mae))


def predict_model(normalized_data):
    test_predictions = model.predict(normalized_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [fixed acidity]')
    plt.ylabel('Predictions [fixed acidity]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [fixed acidity]")
    _ = plt.ylabel("Count")

    plt.show()


if __name__ == '__main__':
    dataset = extract_information()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    ## plot_analysis_dataset(train_dataset)

    ## correlation_heatmap(ds_view)
    train_stats = train_dataset.describe()
    train_stats.pop("fixed acidity")
    train_stats = train_stats.transpose()
    # train_stats

    train_labels = train_dataset.pop('fixed acidity')
    test_labels = test_dataset.pop('fixed acidity')


    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']


    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    model = build_model()
    print(model.summary())

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)

    train_model(normed_train_data, train_labels, True)

    evaluate_model(normed_test_data, test_labels)

    predict_model(normed_test_data)
