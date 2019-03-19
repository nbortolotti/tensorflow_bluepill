import tensorflow as tf
import pandas as pd
import numpy as np

ds_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

dataset_path = tf.keras.utils.get_file(ds_url.split('/')[-1], ds_url)
dataset_csv = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=";", skipinitialspace=True, header=0)

dataset_final = dataset_csv[['alcohol', 'chlorides', 'quality']]

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
categories = pd.cut(dataset_final['quality'], bins, labels=group_names)
dataset_final['quality'] = categories.replace({"bad": 0, "good": 1})

train_dataset = dataset_final.sample(frac=0.8, random_state=0)
test_dataset = dataset_final.drop(train_dataset.index)

train_features, train_categories = train_dataset, train_dataset.pop("quality")
test_features, test_categories = test_dataset, test_dataset.pop("quality")

#y_categorical = tf.contrib.keras.utils.to_categorical(train_categories, num_classes=2)
#test_y_categorical = tf.contrib.keras.utils.to_categorical(test_categories, num_classes=2)

# Build dataset
#dataset = tf.data.Dataset.from_tensor_slices((train_features.values, y_categorical))
dataset = tf.data.Dataset.from_tensor_slices((train_features.values, train_categories.values))
dataset = dataset.batch(32)
dataset = dataset.shuffle(1000)
dataset = dataset.repeat()

#dataset_test = tf.data.Dataset.from_tensor_slices((test_features.values, test_y_categorical))
dataset_test = tf.data.Dataset.from_tensor_slices((test_features.values, test_categories.values))
dataset_test = dataset_test.batch(32)
dataset_test = dataset_test.shuffle(1000)
dataset_test = dataset_test.repeat()

model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation='relu', input_dim=2),
  tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dense(2, activation=tf.nn.sigmoid),
])

model.summary()

# op = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# optimizer='adam'
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
model.fit(dataset, steps_per_epoch=32, epochs=300, callbacks=[early_stop], verbose=1)

loss, accuracy = model.evaluate(dataset_test, steps=32)

print("loss:%f" % (loss))
print("accuracy: %f" % (accuracy))


new_vine = np.array([[10.0, 0.065]])
prediction = model.predict_classes(new_vine)
if prediction[0] == 1:
  print("Good")
else:
  print("Bad")