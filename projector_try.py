# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins import projector
import numpy as np
print(tf.__version__)


# %%
imdb=keras.datasets.imdb
(train_x, train_y), (test_x, text_y)=keras.datasets.imdb.load_data(num_words=10000)


# %%
print("Training entries: {}, labels: {}".format(len(train_x), len(train_y)))


# %%
print(train_x[0])


# %%
print('len: ',len(train_x[0]), len(train_x[1]))


# %%
word_index  = imdb.get_word_index()


# %%
word2id = {k:(v+3) for k, v in word_index.items()}
word2id['<PAD>'] = 0
word2id['<START>'] = 1
word2id['<UNK>'] = 2
word2id['<UNUSED>'] = 3
id2word = {v:k for k, v in word2id.items()}
def get_words(sent_ids):
    return ' '.join([id2word.get(i, '?') for i in sent_ids])

sent = get_words(train_x[0])
print(sent)


# %%
print(len(word2id))

# %%
with open('./logs/text_classify/word.tsv','w',encoding='utf-8') as f:
    for i in range(len(word2id)):
        f.write('{}\n'.format(id2word[i]))

# %%
# 句子末尾padding
train_x = keras.preprocessing.sequence.pad_sequences(
    train_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)
test_x = keras.preprocessing.sequence.pad_sequences(
    test_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)
print(train_x[0])
print('len: ',len(train_x[0]), len(train_x[1]))


# %%
vocab_size = 10000
model = keras.Sequential()
model.add(layers.Embedding(vocab_size, 16,name='embed'))
model.add(layers.GlobalAveragePooling1D(name='pool'))
model.add(layers.Dense(16, activation='relu',name='relu_layer'))
model.add(layers.Dense(1, activation='sigmoid',name='output_layer'))
model.summary()


# %%
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])


# %%
layer_embed = model.layers[0]


# %%
layer_embed.name

#%%
def register_embedding(meta_data_fname, log_dir, tensor_name):
    config = projector.ProjectorConfig()
    for i in range(3):
        embedding = config.embeddings.add()
        embedding.tensor_name = tensor_name
        embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir,config)

class ProjectorCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, embeddings_metadata_fname, log_dir, tensor_name, **kwargs)
        super().__init__(**kwargs)
        
        self.meta_data_fname
    def on_train_begin(self, logs=None):
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = tensor_name
        embedding.metadata_path = meta_data_fname
        projector.visualize_embeddings(log_dir,config)
    def on_epoch_end(self, epoch, logs=None):
        pass

# %%
# check_path = './model_save/text_classify/model.ckpt'
# callback = [
#             tf.keras.callbacks.TensorBoard(log_dir='./logs/text_classify',histogram_freq=5000,
#             write_graph=True, write_images=False,update_freq='epoch',
#             embeddings_freq=1,
#             embeddings_metadata ={layer_embed.name:'word.tsv'}),
#             tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,save_freq=10000)
#             ]


# %%
x_val = train_x[:10000]
x_train = train_x[10000:]

y_val = train_y[:10000]
y_train = train_y[10000:]

history = model.fit(x_train,y_train,
                   epochs=5, batch_size=512,
                   validation_data=(x_val, y_val),
                   verbose=1,callbacks = callback)

result = model.evaluate(test_x, text_y)
print(result)

# %%



