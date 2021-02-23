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
        self.dense = layers.Dense(hidden_size, name = "linear")
        self.outlayer = layers.Dense(output_size, 
                        activation = 'sigmoid', name = "out_layer")
        self.build(input_shape=(None, self.input_size))

    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(input_shape, tuple):
            inputs = tf.keras.Input(shape=input_shape[1:])
        elif isinstance(input_shape, list):
            try:
                inputs = [tf.keras.Input(shape = (i[1:],)) 
                                        for i in input_shape]
            except TypeError:
                print("User Input_shape for build function is not right.")
        else:
            return
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model.")
        
        _ = self.call(inputs)
    
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
logdir = "logs_custom_train_step_submodel" + os.path.sep + datetime.now().strftime("""%Y%m%d-%H%M%S""")
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
]
# %%
epochs = 200
model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_test, y_test))


# %%



