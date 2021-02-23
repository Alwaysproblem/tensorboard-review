# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow.keras import Input, layers
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split as tvsplit
# disable logging warning and error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.run_functions_eagerly(True)
# tf.config.experimental_run_functions_eagerly(True) tensorflow <= 2.2


# %%
sample_n = 100
epochs = 50


# %%
meana = np.array([1, 1])
cova = np.array([[0.1, 0],[0, 0.1]])

meanb = np.array([2, 2])
covb = np.array([[0.1, 0],[0, 0.1]])

x_red = np.random.multivariate_normal(mean=meana, cov = cova, size=sample_n)
x_green = np.random.multivariate_normal(mean=meanb, cov = covb, size=sample_n)

y_red = np.array([1] * sample_n)
y_green = np.array([0] * sample_n)

plt.scatter(x_red[:, 0], x_red[:, 1], c = 'red' , marker='.', s = 30)
plt.scatter(x_green[:, 0], x_green[:, 1], c = 'green', marker='.', s = 30)
# plt.show()

X = np.concatenate([x_red, x_green]).astype(np.float32)
y = np.concatenate([y_red, y_green]).astype(np.float32)


# %%
X_train, X_test, y_train, y_test = tvsplit(X, y)


# %%
class CustomModel(tf.keras.models.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

#%%
def logstic(input_size=2, hidden_size = 5, output_size=1):
    Inputs = Input(shape=(input_size,), name="Inputs")
    linear1 = layers.Dense(hidden_size)(Inputs)
    Outputs = layers.Dense(output_size, activation=tf.keras.activations.sigmoid)(linear1)
    model = CustomModel(inputs=Inputs, outputs=Outputs, name="Logistic")
    return model
model = logstic()

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])
model.summary()


# %%
epochs = 200
model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test))
#%%[markdown]

# - "accuracy" ->  `tf.keras.metrics.BinaryAccuracy()`, `tf.keras.metrics.CategoricalAccuracy()`
# https://stackoverflow.com/questions/45632549/why-is-the-accuracy-for-my-keras-model-always-0-when-training

#%%
loss_tracker = tf.keras.metrics.Mean(name="losses")
mae_metric = tf.keras.metrics.BinaryAccuracy()

class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute our own loss
            loss = tf.keras.losses.mean_squared_error(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_metric.update_state(y, y_pred)
        return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mae_metric]
    
    def test_step(self, data):
            # Unpack the data
            x, y = data
            # Compute predictions
            y_pred = self(x, training=False)
            loss = tf.keras.losses.mean_squared_error(y, y_pred)
            # Updates the metrics tracking the loss
            # self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss_tracker.update_state(loss)
            # Update the metrics.
            # self.compiled_metrics.update_state(y, y_pred)
            mae_metric.update_state(y, y_pred)
            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            return {m.name: m.result() for m in self.metrics}

#%%
def logstic(input_size=2, hidden_size = 5, output_size=1):
    Inputs = Input(shape=(input_size,), name="Inputs_1")
    linear1 = layers.Dense(hidden_size)(Inputs)
    Outputs = layers.Dense(output_size, activation=tf.keras.activations.sigmoid)(linear1)
    model = CustomModel(inputs=Inputs, outputs=Outputs, name="Logistic_1")
    return model
model = logstic()

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(),)
model.summary()


# %%
epochs = 200
model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test))