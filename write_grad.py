# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#%%
import tensorflow as tf
from tensorflow.keras import Input, layers
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split as tvsplit
# disable logging warning and error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.experimental_run_functions_eagerly(True)
#%%
sample_n = 100
epochs = 200
#%%
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


#%%
X_train, X_test, y_train, y_test = tvsplit(X, y)

#%%
def logstic(input_size=2, hidden_size = 5, output_size=1):
    Inputs = Input(shape=(input_size,), name="Inputs")
    linear1 = layers.Dense(hidden_size)(Inputs)
    Outputs = layers.Dense(output_size, activation=tf.keras.activations.sigmoid)(linear1)
    model = tf.keras.Model(inputs=Inputs, outputs=Outputs, name="Logistic")
    return model
model = logstic()
#%%
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=["accuracy"])
model.summary()


#%%
# TODO: need to be modified for gradient viz or waiting for a new version
class GradientCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir, console, **kwargs):
        super().__init__(**kwargs)
        self.console = console
        self.witer = None
        self.logdir = logdir
    
    def on_train_begin(self, logs=None):
        self.witer = tf.summary.create_file_writer(self.logdir)

    def on_train_end(self, logs=None):
        self.witer.close()

    # @tf.function
    def on_epoch_begin(self, epoch, logs=None):
        weights = self.model.trainable_weights
        loss = self.loss(self.model(self.models.inputs),)
        optimizer = self.model.optimizer
        gradients = optimizer.get_gradients(loss, weights)
        print(gradients)
        for t in gradients:
            if self.console:
                print('Tensor: {}'.format(t.name))
                print('{}\n'.format(K.get_value(t)[:10]))
            else:
                with self.witer.as_default():
                    tf.summary.histogram(t.name, data=t, step=epoch)
#%%
# tensorboard
logdir = "logs_grad" + os.path.sep + "standard" + os.path.sep + datetime.now().strftime("""%Y%m%d-%H%M%S""")
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images = True),
    GradientCallback(logdir, True)
]
#%%
model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks, validation_data=(X_test, y_test))