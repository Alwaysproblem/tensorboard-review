#%%
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.keras.layers as layers
print(tf.__version__)

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
with open('./logs/word.tsv','w',encoding='utf-8') as f:
    for i in range(len(word2id)):
        f.write('{}\n'.format(id2word[i]))

#%%
# 句子末尾padding
train_x = keras.preprocessing.sequence.pad_sequences(
    train_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)
test_x = keras.preprocessing.sequence.pad_sequences(
    test_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)
#%%
vocab_size = 10000
model = keras.Sequential()
model.add(layers.Embedding(vocab_size, 16,name='embed'))
model.add(layers.GlobalAveragePooling1D(name='pool'))
model.add(layers.Dense(16, activation='relu',name='relu_layer'))
model.add(layers.Dense(1, activation='sigmoid',name='output_layer'))
model.summary()

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

x_val = train_x[:10000]
x_train = train_x[10000:]

y_val = train_y[:10000]
y_train = train_y[10000:]

#%%
dataloader = tf.data.Dataset.from_tensor_slices((x_train,y_train))
dataloader = dataloader.batch(521)

from tensorboard.plugins import projector

def register_embedding( meta_data_fname, log_dir):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = 'embed/embedding:0'
    embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir,config)

def train_step(model,dataloader):
    # tensor_embeddings = []
    for i in range(1):
        for x_data,y_data in dataloader:
            history = model.fit(x_data,y_data,
                                validation_data=(x_val, y_val))

        print('epochs is {}'.format(i))
        tf.summary.scalar('loss',history.history['loss'][-1],step=i)
        tf.summary.scalar('val_loss',history.history['val_loss'][-1],step=i)
    
    # tensor_embeddings.append(tf.Variable(model.variables[0],name='embed/embedding'))
    # saver = tf.compat.v1.train.Saver(tensor_embeddings)
    # saver.save(sess=None, global_step=i, save_path='./logs/embed.ckpt')
    # model.save_weights('./logs/embed.ckpt')
    tf.train.Checkpoint(model=model).save(file_prefix='./logs/embed.ckpt')

summary_file  = tf.summary.create_file_writer('./logs/')
register_embedding('word.tsv','./logs/')
with summary_file.as_default():
    train_step(model,dataloader)
# summary_file.close()