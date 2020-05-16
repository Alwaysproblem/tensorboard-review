#%%
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import tensorflow_datasets as tfds
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


# # 句子末尾padding
# train_x = keras.preprocessing.sequence.pad_sequences(
#     train_x, value=word2id['<PAD>'],
#     padding='post', maxlen=256
# )

# test_x = keras.preprocessing.sequence.pad_sequences(
#     test_x, value=word2id['<PAD>'],
#     padding='post', maxlen=256
# )

(train_data, test_data), info = tfds.load(
    "imdb_reviews/subwords8k",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True,
    as_supervised=True,
)
encoder = info.features["text"].encoder

# shuffle and pad the data.
train_x = train_data.shuffle(1000).padded_batch(
    10, padded_shapes=((None,), ())
)
test_x = test_data.shuffle(1000).padded_batch(
    10, padded_shapes=((None,), ())
)


logdir_name = 'log-base'
#%%
log_dir=f'./{logdir_name}'
os.makedirs(log_dir + '/train', exist_ok=True)


with open(f'./{logdir_name}/train/word.tsv','w',encoding='utf-8') as f:
    for i in range(len(word2id)):
        f.write('{}\n'.format(id2word[i]))

#%%
import os
class ProjectorCallback(tf.keras.callbacks.TensorBoard):

    def _configure_embeddings(self):
        from tensorflow.python.keras.layers import embeddings
        try:
            from tensorboard.plugins import projector
        except ImportError:
            raise ImportError('Failed to import TensorBoard. Please make sure that '
                            'TensorBoard integration is complete."')
        config = projector.ProjectorConfig()
        for ind, layer in enumerate(self.model.layers):
            if isinstance(layer, embeddings.Embedding):
                embedding = config.embeddings.add()
                tensor_name = layer.embeddings.name.split('/')[-1].split(':')[0]
                embedding.tensor_name = f"layer_with_weights-{ind}/{tensor_name}/.ATTRIBUTES/VARIABLE_VALUE"

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

        writer = DummyWriter(self._log_write_dir + '/train')
        projector.visualize_embeddings(writer, config)


PCB = ProjectorCallback(f"./{logdir_name}", histogram_freq = 1, profile_batch = 3,
                        embeddings_freq = 1, 
                        embeddings_metadata={"embed": "word.tsv"})



#%%

model = keras.Sequential()
model.add(layers.Embedding(vocab_size, 16, name='embed'))
model.add(layers.GlobalAveragePooling1D(name='pool'))
model.add(layers.Dense(64, activation='relu', name='relu_layer'))
model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
model.summary()
model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
#%%
history = model.fit(train_x,
                    epochs=4, batch_size=512,
                    validation_data=test_x,
                    verbose=1,
                    callbacks=[ 
                        PCB ,
                        # tf.keras.callbacks.TensorBoard(f"./{logdir_name}/",
                        #      histogram_freq=1, profile_batch = 3, )
                        #     #  embeddings_freq = 1)
                    ])

# %%
