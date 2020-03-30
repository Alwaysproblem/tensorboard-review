# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## What-If Tool and SHAP on COMPAS keras model
# 
# This notebook shows:
# - Training of a keras model on the [COMPAS](https://www.kaggle.com/danofer/compass) dataset.
# - Explanation of inference results using [SHAP](https://github.com/slundberg/shap).
# - Use of What-If Tool on the trained model, including SHAP values.
# 
# For ML fairness background on COMPAS see:
# 
# - https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
# - https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
# - http://www.crj.org/assets/2017/07/9_Machine_bias_rejoinder.pdf
# 
# This notebook trains a model to mimic the behavior of the COMPAS recidivism classifier and uses the SHAP library to provide feature importance for each prediction by the model. We can then analyze our COMPAS proxy model for fairness using the What-If Tool, and explore how important each feature was to each prediction through the SHAP values.
# 
# The specific binary classification task for this model is to determine if a person belongs in the "Low" risk class according to COMPAS (negative class), or the "Medium" or "High" risk class (positive class). We then analyze it with the What-If Tool for its ability to predict recidivism within two years of arrest.
# 
# A simpler version of this notebook that doesn't make use of the SHAP explainer can be found [here](https://colab.research.google.com/github/pair-code/what-if-tool/blob/master/WIT_COMPAS.ipynb).
# 
# Copyright 2019 Google LLC.
# SPDX-License-Identifier: Apache-2.0

# %%
#@title Read training dataset from CSV {display-mode: "form"}
import pandas as pd
import numpy as np
import tensorflow as tf
import witwidget
import os
import pickle

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.utils import shuffle

df = pd.read_csv('https://storage.googleapis.com/what-if-tool-resources/computefest2019/cox-violent-parsed_filt.csv')


# %%
# Preprocess the data

# Filter out entries with no indication of recidivism or no compass score
df = df[df['is_recid'] != -1]
df = df[df['decile_score'] != -1]

# Rename recidivism column
df['recidivism_within_2_years'] = df['is_recid']

# Make the COMPASS label column numeric (0 and 1), for use in our model
df['COMPASS_determination'] = np.where(df['score_text'] == 'Low', 0, 1)

df = pd.get_dummies(df, columns=['sex', 'race'])

# Get list of all columns from the dataset we will use for model input or output.
input_features = ['sex_Female', 'sex_Male', 'age', 'race_African-American', 'race_Caucasian', 'race_Hispanic', 'race_Native American', 'race_Other', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']

to_keep = input_features + ['recidivism_within_2_years', 'COMPASS_determination']

to_remove = [col for col in df.columns if col not in to_keep]
df = df.drop(columns=to_remove)

input_columns = df.columns.tolist()
labels = df['COMPASS_determination']
df.head()


# %%
# Create data structures needing for training and testing.
# The training data doesn't contain the column we are predicting,
# 'COMPASS_determination', or the column we are using for evaluation of our
# trained model, 'recidivism_within_2_years'.
df_for_training = df.drop(columns=['COMPASS_determination', 'recidivism_within_2_years'])
train_size = int(len(df_for_training) * 0.8)

train_data = df_for_training[:train_size]
train_labels = labels[:train_size]

test_data_with_labels = df[train_size:]


# %%
# Create the model

# This is the size of the array we'll be feeding into our model for each example
input_size = len(train_data.iloc[0])

model = Sequential()
model.add(Dense(200, input_shape=(input_size,), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')


# %%
# Train the model
model.fit(train_data.values, train_labels.values, epochs=4, batch_size=32, validation_split=0.1)


# %%
# Create a SHAP explainer by passing a subset of our training data
import shap
explainer = shap.DeepExplainer(model, train_data.values[:200])


# %%
# Explain predictions of the model on the first 5 examples from our training set
# to test the SHAP explainer.
shap_values = explainer.shap_values(train_data.values[:5])
shap_values


# %%
# Column indices to strip out from data from WIT before passing it to the model.
columns_not_for_model_input = [
    test_data_with_labels.columns.get_loc("recidivism_within_2_years"),
    test_data_with_labels.columns.get_loc("COMPASS_determination")
]

# Return model predictions and SHAP values for each inference.
def custom_predict_with_shap(examples_to_infer):
  # Delete columns not used by model
  model_inputs = np.delete(
      np.array(examples_to_infer), columns_not_for_model_input, axis=1).tolist()

  # Get the class predictions from the model.
  preds = model.predict(model_inputs)
  preds = [[1 - pred[0], pred[0]] for pred in preds]

  # Get the SHAP values from the explainer and create a map of feature name
  # to SHAP value for each example passed to the model.
  shap_output = explainer.shap_values(np.array(model_inputs))[0]
  attributions = []
  for shap in shap_output:
    attrs = {}
    for i, col in enumerate(df_for_training.columns):
      attrs[col] = shap[i]
    attributions.append(attrs)
  ret = {'predictions': preds, 'attributions': attributions}
  return ret


# %%
#@title Show model results and SHAP values in WIT
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder
num_datapoints = 1000  #@param {type: "number"}


examples_for_shap_wit = test_data_with_labels.values.tolist()
column_names = test_data_with_labels.columns.tolist()


# %%
examples_for_shap_wit[0]


# %%
column_names


# %%
custom_predict_with_shap(examples_for_shap_wit[1:2])


# %%
config_builder = WitConfigBuilder(
    examples_for_shap_wit[:num_datapoints],
    feature_names=column_names).set_custom_predict_fn(
  custom_predict_with_shap).set_target_feature('recidivism_within_2_years')

ww = WitWidget(config_builder, height=800)
ww

# %% [markdown]
# #### What-If Tool exploration ideas
# 
# - Organize datapoints by "inference score" (can do this through binning or use of scatter plot) to see points ordered by how likely they were determined to re-offend.
#   - Select a point near the boundary line (where red points turn to blue points)
#   - Find the nearest counterfactual to see a similar person with a different decision. What is different?
#   - Look at the partial dependence plots for the selected person. What changes in what features would change the decision on this person?
# - Explore the attribution values provided by SHAP.
#   - For a variety of selected datapoints, look at which features have the highest positive attribution values. These are making the model predict higher risk for this person.
#   - Look at which features have the lowest negative attribution values as well. These are making the model predict lower risk for this person.
#   - How well do these attribution scores line up with the partial dependence plots for those datapoints?
#   - Use the attribution scores in the datapoints visualizations to look for interesting patterns. As one example, you could set the scatter X-axis to "attributions__age" and the scatter Y-axis to "attributions__priors_count" with the points colored by "Inference score" to investigate the relationship between feature importance of those two features and how those relate to the score the model gives for each datapoint being "High risk".
# - In "Performance and Fairness" tab, slice the dataset by different features (such as race or sex)
#   - Look at the confusion matrices for each slice - How does performance compare in those slices? What from the training data may have caused the difference in performance between the slices? What root causes could exist?
#   - Use the threshold optimization buttons to optimize positive classification thresholds for each slice based on any of the possible fairness constraints - How different do the thresholds have to be to achieve that constraint? How varied are the thresholds depending on the fairness constraint chosen?
# 
# - In the "Performance + Fairness" tab, change the cost ratio so that you can optimize the threshold based off of a non-symmetric cost of false positives vs false negatives. Then click the "optimize threshold" button and see the effect on the confusion matrix. 
#   - Slice the dataset by a feature, such as sex or race. How has the new cost ratio affected the disparity in performance between slices? Click the different threshold optimization buttons to see how the changed cost ratio affects the disparity given different fairness constraints.
# 
# 
# %% [markdown]
# #### Further exploration ideas
# 
# - Edit the training data so that race fields are not included as a feature and train a new model with this data as input (make sure to create a new explainer and a new custom prediction function that filters race out of model input and uses the right explainer and model).
# - Load the new model with set_compare_custom_predict_fn and compare it with the original model.
#   - HINT: You'll need to make edits in 3 separate code cells.
#   - Is there still a racial disparity in model results? If so, what could be the causes?
#   - How did the SHAP attributions change?

