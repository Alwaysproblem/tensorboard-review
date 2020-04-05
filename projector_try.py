#%%
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
print(tf.__version__)
vocab_size = 10000

#%%
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

logs_name = "logs"

#%%
with open(f'./{logs_name}/word.tsv','w',encoding='utf-8') as f:
    for i in range(len(word2id)):
        f.write('{}\n'.format(id2word[i]))

#%%
import os
class ProjectorCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir, embeddings_freq=0, embeddings_metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata
        self.logdir = logdir
        self.embeddings = []
        self.writer = tf.summary.create_file_writer(self.logdir)

    def set_model(self, model):
        super().set_model(model)
        from tensorflow.python.keras.layers import embeddings
        try:
            from tensorboard.plugins import projector
        except ImportError:
            raise ImportError('Failed to import TensorBoard. Please make sure that '
                            'TensorBoard integration is complete."')
        config = projector.ProjectorConfig()
        for layer in self.model.layers:
            if isinstance(layer, embeddings.Embedding):
                embedding = config.embeddings.add()
                embedding.tensor_name = layer.embeddings.name
                self.embeddings.append(layer.embeddings)

                if self.embeddings_metadata is not None:
                    if isinstance(self.embeddings_metadata, str):
                        embedding.metadata_path = self.embeddings_metadata
                        self.embeddings_metadata = None
                    else:
                        if layer.name in self.embeddings_metadata:
                            embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

        if self.embeddings_metadata:
            raise ValueError('Unrecognized `Embedding` layer names passed to '
                            '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                            'argument: ' + str(self.embeddings_metadata.keys()))

        class DummyWriter(object):
            """Dummy writer to conform to `Projector` API."""

            def __init__(self, logdir):
                self.logdir = logdir

            def get_logdir(self):
                return self.logdir

        writer = DummyWriter(self.logdir)
        projector.visualize_embeddings(writer, config)

    # def on_train_begin(self, log=None):
    #     #TODO: start the writer
    #     self.writer = tf.summary.create_file_writer(self.logdir)
    #     self.writer.as_default()

    # def on_train_end(self, log=None):
    #     self.writer.close()

    def save_embedding(self, epoch):
        embeddings_ckpt = os.path.join(self.logdir, 'keras_embedding-ckpt')
        # self.model.save_weights(embeddings_ckpt)
        saver = tf.compat.v1.train.Saver(self.embeddings)
        saver.save(sess=None, global_step=epoch, save_path=embeddings_ckpt)

    def on_epoch_end(self, epoch, logs=None):
        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self.save_embedding(epoch)


PCB = ProjectorCallback(f"./{logs_name}", 1, {"embed": "word.tsv"})

#%%

model = keras.Sequential()
model.add(layers.Embedding(vocab_size, 16, name='embed'))
model.add(layers.GlobalAveragePooling1D(name='pool'))
model.add(layers.Dense(64, activation='relu', name='relu_layer'))
model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))

#%%
model.summary()
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
x_val = train_x[:10000]
x_train = train_x[10000:]
y_val = train_y[:10000]
y_train = train_y[10000:]

#%%
history = model.fit(x_train, y_train,
                    epochs=4, batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1,
                    callbacks=[ 
                        PCB ,
                        # tf.keras.callbacks.TensorBoard(f"./{logs_name}/",
                        #      histogram_freq=1, profile_batch = 3,
                        #      embeddings_freq = 1)
                    ])

# %%
