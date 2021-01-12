import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from tensorflow import keras

# Check if gpu is working
tf.config.list_physical_devices("GPU")

# Loading dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Visualisung Data
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Normalise data. From 0-255 to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Setting up model
model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

# Compile model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train model
model.fit(train_images, train_labels, epochs=10)

# Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

# Example Prediction
predictions = model.predict(test_images)
print(predictions[0])
print(numpy.argmax(predictions[0]))
print(test_labels[0])