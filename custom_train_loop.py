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
from sklearn.model_selection import train_test_split as tvsplit
# disable logging warning and error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#%%
sample_n = 100
epochs = 50
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
class Logistic(tf.keras.models.Model):
    def __init__(self, input_size=2, hidden_size = 5, output_size=1, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(hidden_size, name = "linear")
        self.outlayer = layers.Dense(output_size, 
                        activation = 'sigmoid', name = "out_layer")
        super().build(input_shape=(None, input_size))
    
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 2), dtype=tf.float32)])
    # @tf.function
    def call(self, X):
        X = self.dense(X)
        Y = self.outlayer(X)
        return Y

#%%
# tensorboard
logdir = "logs" + os.path.sep + "custom" + os.path.sep + datetime.now().strftime("""%Y%m%d-%H%M%S""")
#%%
model = Logistic()
model.summary()
#%%
optimizer=tf.keras.optimizers.Adam()
loss=tf.keras.losses.BinaryCrossentropy()
metrics=tf.keras.metrics.AUC()

#%%
# @tf.function # in graph mode
def losses(y_true, y_pred, sample_weight=None, loss_obj=loss):
    return loss_obj(y_true, y_pred, sample_weight)

#%%
print(f"the loss: {losses(y_test, model(X_test)).numpy()}")

#%%
# @tf.function # in graph mode
def Metrics(y_true, y_pred, sample_weight=None, metrics=metrics):
    metrics.update_state(y_true, y_pred, sample_weight)
    return metrics.result()
#%%
# @tf.function # in graph mode
def grad(model, inputs, labels):
    
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss_value = losses(labels, pred)
        labels = tf.expand_dims(labels, axis=1)
        metr = Metrics(labels, pred)
    
    return loss_value, tape.gradient(loss_value, model.trainable_variables), metr

#%%
closs, grad_v, met = grad(model, X_test, y_test)
print(f"loss value: {closs.numpy()}, grad value: {grad_v}, metrics: {met}.")

# %%
# @tf.function # in graph mode
def train_on_batch(model, inputs, labels, opt=optimizer):
    closs, cgrad, cmetric = grad(model, inputs, labels)
    opt.apply_gradients(zip(
        cgrad,
        model.trainable_variables
    ))
    return closs, cmetric

#%%
writer = tf.summary.create_file_writer(logdir)

#%%
with writer.as_default():
    for e in range(epochs):
        # write graph to tensorboard 
        tf.summary.trace_on(graph=True, profiler=False)

        for ind in range(len(X_train)):
            loss_value, cmetric = train_on_batch(model, X_train[ind][None, :], 
                    np.expand_dims(y_train[ind], axis=0))
        # export the profiler files every step but I don't know why it can not show graph after training.
        tf.summary.trace_export(name="logitic_graph", step=e, profiler_outdir=logdir)
        tf.summary.scalar("logloss", loss_value.numpy(), step=e)
        tf.summary.scalar("metrics", cmetric.numpy(), step=e)
        for v in model.trainable_variables:
            tf.summary.histogram(v.name, v, step=e)
        writer.flush()
        # tf.print(f"Epochs {e}: loss {loss_value.numpy()}, metric:{cmetric.numpy()}") # in graph mode
        print(f"Epochs {e}: loss {loss_value.numpy()}, metric:{cmetric.numpy()}")