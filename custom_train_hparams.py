#%%
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorboard.plugins.hparams import api as hp
import numpy as np
#%%
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#%%
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

opt_dict = {
  "adam": tf.keras.optimizers.Adam(),
  "sgd": tf.keras.optimizers.SGD()
}

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

#%%
# def train_test_model(hparams):
#   model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
#     tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
#     tf.keras.layers.Dense(10, activation=tf.nn.softmax),
#   ])
#   model.compile(
#       optimizer=hparams[HP_OPTIMIZER],
#       loss='sparse_categorical_crossentropy',
#       metrics=['accuracy'],
#   )

#   model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
#   _, accuracy = model.evaluate(x_test, y_test)
#   return accuracy
class mnist_model(tf.keras.models.Model):
    def __init__(self, hparams, input_shape ,**kwargs):
        super().__init__(**kwargs)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu)
        self.dropout = layers.Dropout(hparams[HP_DROPOUT])
        self.outLayer = layers.Dense(10, activation=tf.nn.softmax)
        self.build(input_shape)

    def call(self, X):
        X = self.flatten(X)
        X = self.dense(X)
        X = self.dropout(X)
        y = self.outLayer(X)

        return y

hp_example = {
    HP_NUM_UNITS : 32,
    HP_DROPOUT : 0.5,
    HP_OPTIMIZER: opt_dict["adam"]
}

model = mnist_model(hp_example, (None, 28, 28))
model.summary()
#%%
loss_obj = tf.keras.losses.sparse_categorical_crossentropy
opt = tf.keras.optimizers.Adam
metric = tf.keras.metrics.Accuracy()

def train_on_batch(hparams, model, x_train, y_train):
  
  with tf.GradientTape() as tape:
      losses = loss_obj(y_train, model(x_train))
  dw = tape.gradient(losses, model.trainable_variables)
  hparams[HP_OPTIMIZER].apply_gradients(zip(dw, model.trainable_variables))
  return tf.reduce_mean(losses)

def train_on_epoch(hparams, model, x_train, y_train, batch_size=32):
  eloss = 0
  for i in range(0, len(x_train), batch_size):
    if i + batch_size > len(x_train):
      x, y = x_train[i:], y_train[i:]
    else:
      x, y = x_train[i:i + batch_size], y_train[i:i+batch_size]
    bloss = train_on_batch(hparams, model, x, y)
    eloss += bloss
  return eloss / (len(x_train) // batch_size + 1)

def train_test_model(hparams):
  metric.reset_states()
  model = mnist_model(hp_example, (None, 28, 28))
  eloss = train_on_epoch(hparams, model, 
          x_train.astype(np.float32), y_train.astype(np.float32))
  tf.print(f"the loss of this epoch is {eloss}")
  y_pred_logits = tf.math.argmax(model(x_test), axis=-1)
  metric.update_state(y_test, y_pred_logits)
  acc = metric.result()
  tf.print(f"the accuracy is {acc}")
  return acc

train_test_model(hp_example)
#%%
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial with json format
    hparams[HP_OPTIMIZER] = opt_dict[hparams[HP_OPTIMIZER]] # this will fix the optimizer does not convert to json
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


#%%
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning/' + run_name, hparams)
      session_num += 1

# %%
