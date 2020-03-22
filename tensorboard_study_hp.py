from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
import numpy as np
print(tf.__version__)
vocab_size = 10000

imdb=keras.datasets.imdb
(train_x, train_y), (test_x, text_y)=keras.datasets.imdb.load_data(num_words=10000)
word_index  = imdb.get_word_index()
word2id = {k:(v+3) for k, v in word_index.items()}
word2id['<PAD>'] = 0
word2id['<START>'] = 1
word2id['<UNK>'] = 2
word2id['<UNUSED>'] = 3
id2word = {v:k for k, v in word2id.items()}


# 句子末尾padding
train_x = keras.preprocessing.sequence.pad_sequences(
    train_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)

test_x = keras.preprocessing.sequence.pad_sequences(
    test_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)

hp_num_units = hp.HParam('num_units',hp.Discrete([16,32,64]))
hp_optimizers = hp.HParam('optimizers',hp.Discrete(['adam','RMSprop','sgd']))
Metric_acc = 'accuracy'

with tf.summary.create_file_writer('./log/text_hp/hparam_tuning').as_default():
    hp.hparams_config(
        hparams = [hp_num_units,hp_optimizers],
        metrics = [hp.Metric(Metric_acc,display_name= 'Acc')]
    )

def train(logdir, hparams):
    model = keras.Sequential()
    model.add(layers.Embedding(vocab_size, hparams[hp_num_units], name='embed'))
    model.add(layers.GlobalAveragePooling1D(name='pool'))
    model.add(layers.Dense(hparams[hp_num_units], activation='relu', name='relu_layer'))
    model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
    #     model.summary()
    model.compile(optimizer=hparams[hp_optimizers],
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    x_val = train_x[:10000]
    x_train = train_x[10000:]

    y_val = train_y[:10000]
    y_train = train_y[10000:]

    history = model.fit(x_train, y_train,
                        epochs=4, batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=[hp.KerasCallback(logdir, hparams),
                                        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, profile_batch = 3)])
    _, acc = model.evaluate(test_x, text_y)
    return acc

def run(logdir,hparams):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams)
        accuracy = train(logdir,hparams)
        tf.summary.scalar(Metric_acc, accuracy, step=1)

session_num = 0

for num_units in hp_num_units.domain.values:
    for optimizer in hp_optimizers.domain.values:
        hparams = {
              hp_num_units: num_units,
              hp_optimizers: optimizer}
        run_name =  'run-%d' % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run('log/text_hp/hparam_tuning/' + run_name, hparams)
        session_num += 1

