import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from loadData import *
from proportion import * 

#plotStat()

dataxls = loadXLS()

lstTrain, lstVal, lstTest, yTrain, yVal, yTest = lstTrainValTestCDR(dataxls=dataxls)

dataTrain, dataVal, dataTest = loadData(dataxls=dataxls)


def get_model(width=256, height=256, depth=128):
    "Build a 3D convolutional neural network model."

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model
model = get_model(width=256, height=256, depth=128)
model.summary()

# Compile model
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    metrics=["acc"],
)

# Fit model
model.fit(
    x=dataTrain,
    y=yTrain,
    epochs=10,
    shuffle=True,
    verbose=1,
)

# Results
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

plt.show()    