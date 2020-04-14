#%%
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, layers

#%%
writer = tf.summary.create_file_writer("./try")

with writer.as_default():
    tf.summary.image("image", np.random.rand(3, 5, 6, 1), step=0)

#%%
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.summary()
#%%
for v in model.trainable_variables:
    print(v.shape)