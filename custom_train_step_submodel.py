# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow.keras import Input, layers
import matplotlib.pyplot as plt
import numpy as np
import time
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
class Logistic(tf.keras.models.Model):
    def __init__(self, input_size=2, hidden_size = 5, output_size=1, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.inputs_ = tf.keras.Input(shape=(input_size,), dtype=tf.float32, name = "Inputs")
        self._set_input_layer(self.inputs_)
        self.dense = layers.Dense(hidden_size, name = "linear")
        self.outlayer = layers.Dense(output_size, 
                        activation = 'sigmoid', name = "out_layer")
        # self.inputs = tf.nest.map_structure()
        
        self.build()

    def _set_input_layer(self, inputs):
        """add inputLayer to model and display InputLayers in model.summary()

        Args:
            inputs ([dict]): the result from `tf.keras.Input`
        """
        if isinstance(inputs, dict):
            self.inputs_layer = {n: tf.keras.layers.InputLayer(input_tensor=i, name=n) 
                                    for n, i in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            self.inputs_layer = [tf.keras.layers.InputLayer(input_tensor=i, name=i.name) 
                                    for i in inputs]
        elif tf.is_tensor(inputs):
            self.inputs_layer = tf.keras.layers.InputLayer(input_tensor=inputs, name=inputs.name)
    
    def build(self):
        super(Logistic, self).build(self.inputs_.shape if tf.is_tensor(self.inputs_) else self.inputs_)
        # super(Logistic, self).build((None, 2))
        _ = self.call(self.inputs_)
    
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

    def call(self, X):
        X = self.dense(X)
        Y = self.outlayer(X)
        return Y


# %%
model = Logistic()
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])
model.summary()

#%%
# tensorboard
# logdir = "logs_custom_train_step_submodel" + os.path.sep + datetime.now().strftime("""%Y%m%d-%H%M%S""")
# callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
# ]
# %%
epochs = 200
model.fit(x=X_train, y=y_train, epochs=epochs, validation_data=(X_test, y_test))
# model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_test, y_test))


# %%



