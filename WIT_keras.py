# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import functools

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import Input, layers
from IPython.display import display, dis

# %%
# Creates a tf feature spec from the dataframe and columns specified.
def create_feature_spec(df, columns=None):
    feature_spec = {}
    if columns == None:
        columns = df.columns.values.tolist()
    for f in columns:
        if df[f].dtype is np.dtype(np.int64):
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
        elif df[f].dtype is np.dtype(np.float64):
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
        else:
            feature_spec[f] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    return feature_spec


# %%
def create_feature_columns(columns, feature_spec, df):
    ret = []
    for col in columns:
        if feature_spec[col].dtype is tf.int64 or feature_spec[col].dtype is tf.float32:
            ret.append(tf.feature_column.numeric_column(col))
        else:
            ret.append(tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(col, list(df[col].unique()))))
    return ret


# %%
def tfexamples_input_fn(examples, feature_spec, label, mode=tf.estimator.ModeKeys.EVAL,
                       num_epochs=None, 
                       batch_size=64):
    def ex_generator():
        for i in range(len(examples)):
            yield examples[i].SerializeToString()
    dataset = tf.data.Dataset.from_generator(
      ex_generator, tf.dtypes.string, tf.TensorShape([]))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda tf_example: parse_tf_example(tf_example, label, feature_spec))
    dataset = dataset.repeat(num_epochs)
    return dataset


# %%
def parse_tf_example(example_proto, label, feature_spec):
    parsed_features = tf.io.parse_example(serialized=example_proto, features=feature_spec)
    target = parsed_features.pop(label)
    return parsed_features, target


# %%
# Converts a dataframe into a list of tf.Example protos.
def df_to_examples(df, columns=None):
    examples = []
    if columns == None:
        columns = df.columns.values.tolist()
    for _, row in df.iterrows():
        example = tf.train.Example()
        for col in columns:
            if df[col].dtype is np.dtype(np.int64):
                example.features.feature[col].int64_list.value.append(int(row[col]))
            elif df[col].dtype is np.dtype(np.float64):
                example.features.feature[col].float_list.value.append(row[col])
            elif row[col] == row[col]:
                example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
        examples.append(example)
    return examples


# %%
def make_label_column_numeric(df, label_column, test):
  df[label_column] = np.where(test(df[label_column]), 1, 0)


# %%
dataframe = pd.read_csv("heart.csv")
dataframe.head()


# %%
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# %%
feature_spec = create_feature_spec(dataframe, )


# %%
train_ds = df_to_examples(train)
val_ds = df_to_examples(val)
test_ds = df_to_examples(test)

train_ds = tfexamples_input_fn(train_ds, feature_spec,num_epochs=5 , label = "target")
val_ds = tfexamples_input_fn(val_ds, feature_spec ,"target")
test_ds = tfexamples_input_fn(test_ds, feature_spec ,"target")


# %%
for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )


# %%
feature_columns = []

# numeric cols
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
  feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age = feature_column.numeric_column("age")
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
      'thal', ['fixed', 'normal', 'reversible'])
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)


# %%
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# %%
col = ["age", "ca", "chol", "cp", "exang", "fbs", "oldpeak", "restecg", "sex", "slope",
 "thal", "thalach", "trestbps",]

def get_model(df):
  input_dic = {}
  for c in col:
    if df[c].dtype is np.dtype(np.int64):
      input_dic[c] = Input(shape=(1,), dtype=tf.dtypes.int64, name = c)
    elif df[c].dtype is np.dtype(np.float64):
      input_dic[c] = Input(shape=(1,), dtype=tf.dtypes.float32, name = c)
    else:
      input_dic[c] = Input(shape=(1,), dtype=tf.dtypes.string, name = c)

  x = feature_layer(input_dic)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dense(128, activation='relu')(x)
  y = layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=input_dic, outputs=y)

  return model

# model = tf.keras.Sequential([
#   feature_layer,
#   layers.Dense(128, activation='relu'),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(1)
# ])
model = get_model(dataframe)


# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


# %%
model.fit(train_ds, steps_per_epoch = 1,
          validation_data=val_ds,
          validation_steps=20,
          epochs=5)


# %%
test_example = df_to_examples(val)


# %%
def predict(example):
    exp_dataset = tfexamples_input_fn(example, feature_spec, "target", num_epochs = 1)
    def pop_label(x, y):
        return x
    exp_dataset = exp_dataset.map(pop_label)
    pred = model.predict(exp_dataset)
    return pred

predict(test_example)


# %%
num_datapoints = 2000  #@param {type: "number"}
tool_height_in_px = 1000  #@param {type: "number"}

from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget

# Setup the tool with the test examples and the trained classifier
config_builder = WitConfigBuilder(test_example[0:num_datapoints]).set_custom_predict_fn(predict)


# %%
display(WitWidget(config_builder, height=tool_height_in_px))


