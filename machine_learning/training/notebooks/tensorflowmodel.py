# Databricks notebook source
##################################################################################
# Model Training Notebook
#
# This notebook shows an example of a Model Training pipeline using Delta tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``tensorrec/resources/model-workflow-resource.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - MLflow registered model name to use for the trained model. Will be created if it
# *                                   doesn't exist.
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

#%pip install -r ../../requirements.txt

# COMMAND ----------

!pip install mlflow==1.24.0
!pip install tensorflow==2.10.0
!pip install -q tensorflow-recommenders
!pip install keras==2.10.0



# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# # Path to the Hive-registered Delta table containing the training data.
# dbutils.widgets.text(
#     "training_data_path",
#     "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
#     label="Path to the training data",
# )

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-tensorrec-experiment",
    label="MLflow experiment name",
)
# MLflow registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev-tensorrec-model", label="Model Name"
)

# COMMAND ----------

dbutils.widgets.text("NUM_USERS", "100")
dbutils.widgets.text("NUM_ITEMS", "100")
dbutils.widgets.text("NUM_INTERACTIONS", "1000")
NUM_USERS = int(dbutils.widgets.get("NUM_USERS"))
NUM_ITEMS = int(dbutils.widgets.get("NUM_ITEMS"))
NUM_INTERACTIONS = int(dbutils.widgets.get("NUM_INTERACTIONS"))

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

def gen_data(NUM_USERS, NUM_ITEMS, NUM_INTERACTIONS):
    # Set seed to 42
    np.random.seed(42)
    
    # generate random integers for user_id, item_id & label
    user_id = np.random.randint(NUM_USERS, size=NUM_INTERACTIONS)
    item_id = np.random.randint(NUM_ITEMS, size=NUM_INTERACTIONS)
    label = np.ones(NUM_INTERACTIONS, dtype="int")

    df = pd.DataFrame({"user_id": user_id, "item_id": item_id, "label": label}, columns=['user_id', 'item_id', 'label'])
    sdf = spark.createDataFrame(df)

    df = df.drop_duplicates(subset=["item_id"])
    item_df = df[['item_id']]
    item_df = spark.createDataFrame(item_df)
    item_df = item_df.withColumn('item_id', expr('explode(array_repeat(item_id,4))'))

    vals = np.arange(0, NUM_ITEMS)
    df2 = pd.DataFrame(vals, columns=['user_id'])
    user_df = spark.createDataFrame(df2)
    user_df = user_df.union(user_df)
    user_df = user_df.union(user_df)

    label = np.zeros(int(NUM_ITEMS * 4), dtype="int")
    label_zeros = pd.DataFrame(label, columns=['label'])

    item_df = item_df.toPandas()
    user_df = user_df.toPandas()
    df_concat = pd.concat([item_df, user_df, label_zeros], axis=1)

    sdf_concat = spark.createDataFrame(df_concat)

    sdf_merged = sdf_concat.union(sdf)

    sdf_merged = (sdf_merged.sort(sdf_merged.label.desc()))
    sdf_final = sdf_merged.dropDuplicates(['user_id', 'item_id'])

    return sdf_final


# COMMAND ----------

spark_df = gen_data(NUM_USERS, NUM_ITEMS, NUM_INTERACTIONS)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
#input_table_path = dbutils.widgets.get("training_data_path")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")


# COMMAND ----------

# DBTITLE 1, Load raw data
import mlflow

mlflow.set_experiment(experiment_name)

training_df = spark_df
training_df.display()

# COMMAND ----------

# DBTITLE 1, Helper function
from mlflow.tracking import MlflowClient
import mlflow.pyfunc


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

training_df = training_df.toPandas()

# COMMAND ----------

training_df['item_id'] = training_df['item_id'].astype("string")
training_df['user_id'] = training_df['user_id'].astype("string")

# COMMAND ----------

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
tns_data = tf.data.Dataset.from_tensor_slices(dict(training_df))

# COMMAND ----------

for x in tns_data.take(1).as_numpy_iterator():
    pprint.pprint(x)

# COMMAND ----------

data_size = tns_data.cardinality().numpy()

# COMMAND ----------

interactions = tns_data.map(lambda x: {
    "item_id": x['item_id'],
    "user_id": x['user_id'],
})
items = tns_data.map(lambda x: x['item_id'])

# COMMAND ----------

items = items.apply(tf.data.experimental.unique())

# COMMAND ----------

tf.random.set_seed(42)
shuffled = interactions.shuffle(data_size, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(round(data_size*.8))
test = shuffled.skip(round(data_size*.8)).take(round(data_size*.2))

# COMMAND ----------

unique_item_ids = training_df['item_id'].unique().to_numpy()
unique_user_ids = training_df['user_id'].unique().to_numpy()

unique_item_ids[:10]

# COMMAND ----------

embedding_dimension = 32

# COMMAND ----------

user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

# COMMAND ----------

item_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_item_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dimension)
])

# COMMAND ----------

import tensorflow_recommenders as tfrs

# COMMAND ----------

metrics = tfrs.metrics.FactorizedTopK(
  candidates=items.batch(128).map(item_model)
)

# COMMAND ----------

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

# COMMAND ----------

class ItemlensModel(tfrs.Model):
    
    def __init__(self, user_model, item_model):
        super().__init__()
        self.item_model: tf.keras.Model = item_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the item features and pass them into the item model,
        # getting embeddings back.
        positive_item_embeddings = self.item_model(features["item_id"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_item_embeddings)

# COMMAND ----------

# Instantiate the model.
model = ItemlensModel(user_model, item_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# COMMAND ----------

batch_size = round((data_size*.8)/10/8)*8

# COMMAND ----------

# shuffle, batch, and cache the training and evaluation data.
cached_train = train.shuffle(data_size).batch(batch_size).cache()
cached_test = test.batch(int(batch_size/2)).cache()

# COMMAND ----------

# train the model:
model.fit(cached_train, epochs=3)

# COMMAND ----------

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends items out of the entire items dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((items.batch(100), items.batch(100).map(model.item_model)))
)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# COMMAND ----------

# Save the index.# Save the index.
tf.saved_model.save(index, "/dbfs/FileStore/tensorrec/model")

# COMMAND ----------

# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load("/dbfs/FileStore/tensorrec/model")

# COMMAND ----------

conda_env = {
    'channels': ['conda-forge'],
    'dependencies': [
        'python=3.8.10',
        'pip',
      {'pip':['tensorflow==2.10.0','tensorflow_datasets==4.6.0','tensorflow_recommenders==v0.7.2','pyspark==3.2.1', 'pandas==1.2.4', 'keras==2.10.0', 'numpy==1.20.1', 'cloudpickle==1.6.0', 'mlflow==1.24.0']}
    ],
    'name': 'mlflow-env'
}

# COMMAND ----------

from tensorflow.python.saved_model import signature_constants
tag=[tf.saved_model.SERVING]
key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
 
retrieval_model_path='retrieval'
 
mlflow.end_run()
with mlflow.start_run():
    model_info = mlflow.tensorflow.log_model(tf_saved_model_dir="/dbfs/FileStore/tensorrec/model", tf_meta_graph_tags=tag, tf_signature_def_key=key, artifact_path=retrieval_model_path, registered_model_name=model_name, conda_env=conda_env)
    artifact_uri_retrieval = mlflow.get_artifact_uri()

# COMMAND ----------

# DBTITLE 1, Train model
# import mlflow
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# import mlflow.lightgbm

# # Collect data into a Pandas array for training. Since the timestamp columns would likely
# # cause the model to overfit the data, exclude them to avoid training on them.
# columns = [col for col in training_df.columns if col not in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']]
# data = training_df.toPandas()[columns]

# train, test = train_test_split(data, random_state=123)
# X_train = train.drop(["fare_amount"], axis=1)
# X_test = test.drop(["fare_amount"], axis=1)
# y_train = train.fare_amount
# y_test = test.fare_amount

# mlflow.lightgbm.autolog()
# train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
# test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

# param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
# num_rounds = 100

# # Train a lightGBM model
# model = lgb.train(param, train_lgb_dataset, num_rounds)

# COMMAND ----------

# DBTITLE 1, Log model and return output.



# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)

# COMMAND ----------


