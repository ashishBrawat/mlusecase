# Databricks notebook source
##################################################################################
# Model Training Notebook
#
# This notebook shows an example of a Model Training pipeline using Delta tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``dlops_keras/resources/model-workflow-resource.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the trained model in Unity Catalog. 
#  
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------
# DBTITLE 1, Notebook arguments

# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "training_data_path",
    "/databricks-datasets/nyctaxi-with-zipcodes/subsampled",
    label="Path to the training data",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-dlops-keras-experiment",
    label="MLflow experiment name",
)
# Unity Catalog registered model name to use for the trained model.
dbutils.widgets.text(
    "model_name", "dev.dlops_keras.dlops-keras-model", label="Full (Three-Level) Model Name"
)

# COMMAND ----------
# DBTITLE 1,Define input and output variables

input_table_path = dbutils.widgets.get("training_data_path")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------
# DBTITLE 1, Set experiment

import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------
# DBTITLE 1, Load raw data

training_df = spark.read.format("delta").load(input_table_path)
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

# MAGIC %md
# MAGIC Train a Keras model on built-in data.

# COMMAND ----------
# DBTITLE 1, Imports

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow

# COMMAND ----------
# DBTITLE 1, Load and preprocess data - California housing prediction dataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

cal_housing = fetch_california_housing()

# Split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(
    cal_housing.data,
    cal_housing.target,
    test_size=0.2
)

# Save test dataset for inference:
test_dataset_name = 'test_data_california'
# X_test_save = pd.DataFrame(X_test, columns=["data"])
# X_test_save.write.format("delta").mode("overwrite").saveAsTable(test_dataset_name)

# COMMAND ----------
# DBTITLE 1, Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------
# DBTITLE 1, Model Creation and compilation

def create_model():
  model = Sequential()
  model.add(Dense(20, input_dim=8, activation="relu"))
  model.add(Dense(20, activation="relu"))
  model.add(Dense(1, activation="linear"))

  return model

model = create_model()
model.compile(
    loss="mse",
    optimizer="Adam",
    metrics=["mse"]
)

# COMMAND ----------
# DBTITLE 1, Define callbacks and tensorboard logging

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Replace path with your path.
path = 'FileStore'
experiment_log_dir = f"/dbfs/{path}/tb"
checkpoint_path = f"/dbfs/{path}/keras_checkpoint_weights.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)

history = model.fit(
    X_train, y_train, validation_split=.2, epochs=35,
    callbacks=[tensorboard_callback, model_checkpoint, early_stopping]
)

# COMMAND ----------
# DBTITLE 1, Evaluate model

model.evaluate(X_test, y_test)

# COMMAND ----------
# DBTITLE 1, Hyperparameter tuning with Hyperopt and MLflow

def create_model(n):
  model = Sequential()
  model.add(Dense(int(n["dense_l1"]), input_dim=8, activation="relu"))
  model.add(Dense(int(n["dense_l2"]), activation="relu"))
  model.add(Dense(1, activation="linear"))

  return model

# COMMAND ----------
# DBTITLE 1, Hyperparameter tuning with Hyperopt and MLflow - 2. Define objective function

from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials


def runNN(n):
    # Import tensorflow
    import tensorflow as tf

    # Log run information with mlflow.tensorflow.autolog()
    mlflow.tensorflow.autolog()

    model = create_model(n)

    # Select optimizer
    optimizer_call = getattr(tf.keras.optimizers, n["optimizer"])
    optimizer = optimizer_call(learning_rate=n["learning_rate"])

    # Compile model
    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mse"]
    )

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, verbose=2)

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    obj_metric = score[0]

    return {"loss": obj_metric, "status": STATUS_OK}

# COMMAND ----------
# DBTITLE 1, Hyperparameter tuning with Hyperopt and MLflow - 3. Perform Hyperparameter tuning

space = {
    "dense_l1": hp.quniform("dense_l1", 10, 30, 1),
    "dense_l2": hp.quniform("dense_l2", 10, 30, 1),
    "learning_rate": hp.loguniform("learning_rate", -5, 0),
    "optimizer": hp.choice("optimizer", ["Adadelta", "Adam"])
}
spark_trials = SparkTrials()
with mlflow.start_run():
    best_hyperparam = fmin(
        fn=runNN,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        trials=spark_trials
    )

# COMMAND ----------
# DBTITLE 1, Hyperparameter tuning with Hyperopt and MLflow - 4. Retrain model with best Hyperparameters

import hyperopt

print(hyperopt.space_eval(space, best_hyperparam))
first_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l1"]
second_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l2"]
learning_rate = hyperopt.space_eval(space, best_hyperparam)["learning_rate"]
optimizer = hyperopt.space_eval(space, best_hyperparam)["optimizer"]

# Get optimizer and update with learning_rate value
optimizer_call = getattr(tf.keras.optimizers, optimizer)
optimizer = optimizer_call(learning_rate=learning_rate)

def create_new_model():
  model = Sequential()
  model.add(Dense(first_layer, input_dim=8, activation="relu"))
  model.add(Dense(second_layer, activation="relu"))
  model.add(Dense(1, activation="linear"))

  return model


best_model = create_new_model()
best_model.compile(
    loss="mse",
    optimizer=optimizer,
    metrics=["mse"]
)

# When autolog() is active, MLflow does not automatically end a run
mlflow.end_run()

# Retrain model:
import matplotlib.pyplot as plt


mlflow.tensorflow.autolog()
with mlflow.start_run() as run:
    history = best_model.fit(X_train, y_train, epochs=35, callbacks=[early_stopping])

    # Save the run information to register the model later
    kerasURI = run.info.artifact_uri

    # Evaluate model on test dataset and log result
    mlflow.log_param("eval_result", best_model.evaluate(X_test, y_test)[0])

    # Plot predicted vs known values for a quick visual check of the model and log the plot as an artifact
    keras_pred = best_model.predict(X_test)
    plt.plot(y_test, keras_pred, "o", markersize=2)
    plt.xlabel("observed value")
    plt.ylabel("predicted value")
    plt.savefig("kplot.png")

    mlflow.log_artifact("kplot.png")

# input_example = X_train[0]
from mlflow.models import infer_signature
signature = infer_signature(X_train, y_train)
mlflow.tensorflow.log_model(
    model,
    artifact_path="tf_keras_model",
    signature=signature,
    registered_model_name=model_name
)
# model_uri = f"models:/{model_name}/{model_version}"
# model_version = mlflow.register_model(
#     model_uri,
#     model_name
# )

# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# import mlflow.lightgbm
#
# # Collect data into a Pandas array for training. Since the timestamp columns would likely
# # cause the model to overfit the data, exclude them to avoid training on them.
# columns = [col for col in training_df.columns if col not in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']]
# data = training_df.toPandas()[columns]
#
# train, test = train_test_split(data, random_state=123)
# X_train = train.drop(["fare_amount"], axis=1)
# X_test = test.drop(["fare_amount"], axis=1)
# y_train = train.fare_amount
# y_test = test.fare_amount
#
# mlflow.lightgbm.autolog()
# train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
# test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)
#
# param = {"num_leaves": 32, "objective": "regression", "metric": "rmse"}
# num_rounds = 100
#
# # Train a lightGBM model
# model = lgb.train(param, train_lgb_dataset, num_rounds)

# COMMAND ----------
# DBTITLE 1, Log model and return output.

# Take the first row of the training dataset as the model input example.
# input_example = X_train.iloc[[0]]

# Log the trained model with MLflow
# mlflow.lightgbm.log_model(
#     model,
#     artifact_path="lgb_model",
#     # The signature is automatically inferred from the input example and its predicted output.
#     input_example=input_example,
#     registered_model_name=model_name
# )

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
