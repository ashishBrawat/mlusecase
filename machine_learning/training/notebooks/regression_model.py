# Databricks notebook source
# MAGIC %md # AutoML regression 
# MAGIC ## Requirements
# MAGIC Databricks Runtime for Machine Learning 8.3 or above.

## COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

##################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook shows an example of a Model Training pipeline using Databricks Feature Store tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``ml_usecase/resources/model-workflow-resource.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the trained model in Unity Catalog. 
#  
##################################################################################

# COMMAND ----------

# MAGIC %md ## Train a model with feature store

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text(
    "training_data_path",
    "dev_ashish.ashish_mlops.regression_sample",
    label="Path to the training data",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/Users/ashish.b.rawat@koantekorg.onmicrosoft.com/dev-ml_usecase-experiment", #/Users/ashish.b.rawat@koantekorg.onmicrosoft.com/dev_ashish-ashish_mlops-experiment
    label="MLflow experiment name",
)
# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev_ashish.ml_usecase.ml_usecase-model", label="Full (Three-Level) Model Name"
)

# features table name
dbutils.widgets.text(
    "features_table",
    "dev_ashish.ml_usecase.features",
    label="Features Table",
)


# COMMAND ----------

# MAGIC %md
# MAGIC Define input and output variables

# COMMAND ----------

input_table_path = dbutils.widgets.get("training_data_path")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

experiment_name

# COMMAND ----------

model_name

# COMMAND ----------

train_df=spark.read.table(input_table_path)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, expr, rand


# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# Adding a unique ID column to the DataFrame
train_df = train_df.withColumn("id", monotonically_increasing_id())

# COMMAND ----------

# ## inference_data_df includes wine_id (primary key), quality (prediction target), and a real time feature
inference_data_df = train_df.select("id", "Target", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

# MAGIC %md Use a `FeatureLookup` to build a training dataset that uses the specified `lookup_key` to lookup features from the feature table and the online feature `real_time_measurement`. If you do not specify the `feature_names` parameter, all features except the primary key are returned.

# COMMAND ----------

features_table = dbutils.widgets.get("features_table")


# COMMAND ----------

table_name=features_table
table_name

# COMMAND ----------

import uuid

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
from sklearn.model_selection import train_test_split


# COMMAND ----------

fs = feature_store.FeatureStoreClient()


# COMMAND ----------

spark.read.table(table_name).columns

# COMMAND ----------

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="Target", exclude_columns=["id"])
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("Target", axis=1)
    y = training_pd["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X,X_train, X_test, y_train, y_test, training_set,training_pd

# Create the train and test datasets
X,X_train, X_test, y_train, y_test, training_set,training_pd = load_data(table_name, "id")
X_train.head()

# COMMAND ----------

# MAGIC %md # Training
# MAGIC The following command starts an AutoML run. You must provide the column that the model should predict in the `target_col` argument.  
# MAGIC When the run completes, you can follow the link to the best trial notebook to examine the training code. This notebook also includes a feature importance plot.

# COMMAND ----------

from databricks import automl
summary = automl.regress(training_pd, target_col="Target", timeout_minutes=5)

# COMMAND ----------

# MAGIC %md The following command displays information about the AutoML output.

# COMMAND ----------

help(summary)

# COMMAND ----------

import mlflow

# COMMAND ----------

experimentID=summary.experiment.experiment_id
run_id=summary.best_trial.mlflow_run_id
best_model=mlflow.sklearn.load_model(summary.best_trial.model_path)

# COMMAND ----------

# import mlflow
# logged_model = 'runs:/732a801c7b0d4de389d031789d2f152b/model'
# best_model=mlflow.sklearn.load_model(logged_model)

# COMMAND ----------

best_model

# COMMAND ----------

from databricks import feature_store
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

from mlflow.models.signature import infer_signature

# COMMAND ----------

experiment_name

# COMMAND ----------

import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(experiment_name)


# COMMAND ----------

experiment

# COMMAND ----------

model_name

# COMMAND ----------

# Log the trained model with MLflow and package it with feature lookup information.
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:#'2031585130442181'
    model_info =fs.log_model(
        best_model,
        artifact_path="artificats",
        flavor=mlflow.sklearn,
        training_set=training_set,
        input_example = X_train[:5],
        signature     = infer_signature(X_train, y_train),
        registered_model_name=model_name,
    )
    artifact_uri_model = mlflow.get_artifact_uri()

# COMMAND ----------

def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

from mlflow.tracking import MlflowClient


# COMMAND ----------

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)

# COMMAND ----------

# model_info

# COMMAND ----------

#predictions_df = fs.score_batch('runs:/21e29de1fae846bdb7c9862f3fa9b30e/artificats', inference_data_df) # batch_input_df => id + additional_column


# COMMAND ----------

# predictions_df.display()

# COMMAND ----------

  # # If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
  # # the cause may be that a model already exists with the name "wine_quality". Try using a different name.
  # model_name = "regression_auto_model"
  # try:
  #   model_version = mlflow.register_model(f"runs:/{model_info.run_id}/artificats", model_name)
  # except:
  #   model_version = mlflow.register_model(f"runs:/ebdd57cf21254ab08394df39d7157649/artificats", model_name)
  # # Registering the model takes a few seconds, so add a small delay
  # import time
  # time.sleep(15)

  # from mlflow.tracking import MlflowClient

  # # Initializing the MlflowClient
  # client = MlflowClient()

  
  # # Transition the previous version to Archived (if applicable)
  # previous_version = int(model_version.version) - 1
  # if previous_version >= 1:
  #     client.transition_model_version_stage(
  #         name=model_name,
  #         version=str(previous_version),
  #         stage="Archived",
  #     )

  # # Transition the current version to Production
  # client.transition_model_version_stage(
  #     name=model_name,
  #     version=model_version.version,
  #     stage="Production",
  # )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving Model as a mircoservices

# COMMAND ----------

# from mlflow.utils.databricks_utils import get_databricks_host_creds
# import requests

# COMMAND ----------

# # gather other inputs the API needs
# serving_host = spark.conf.get("spark.databricks.workspaceUrl")
# creds = get_databricks_host_creds()


 
# my_json = {
#   "name": 'prediction',
#   "config": {
#    "served_models": [{
#      "model_name": model_version.name,
#      "model_version": model_version.version,
#      "workload_size": "Small",
#      "scale_to_zero_enabled": True
#    }]
#  }
# }
# my_json

# token=  creds.token
# # With the token, you can create our authorization header for our subsequent REST calls
# headers = {
#     "Authorization": f"Bearer {token}",
#     "Content-Type": "application/json"
#   }
# instance=serving_host

# COMMAND ----------

# def func_create_endpoint(model_serving_endpoint_name):
#   #get endpoint status
#   endpoint_url = f"https://{instance}/api/2.0/serving-endpoints"
#   url = f"{endpoint_url}/{model_serving_endpoint_name}"
#   r = requests.get(url, headers=headers)
#   if "RESOURCE_DOES_NOT_EXIST" in r.text:  
#     print("Creating this new endpoint: ", f"https://{instance}/serving-endpoints/{model_serving_endpoint_name}/invocations")
#     re = requests.post(endpoint_url, headers=headers, json=my_json)
#   else:
#     new_model_version = (my_json['config'])['served_models'][0]['model_version']
#     print("This endpoint existed previously! We are updating it to a new config with new model version: ", new_model_version)
#     # update config
#     url = f"{endpoint_url}/{model_serving_endpoint_name}/config"
#     re = requests.put(url, headers=headers, json=my_json['config']) 
#     # wait till new config file in place
#     import time,json
#     #get endpoint status
#     url = f"https://{instance}/api/2.0/serving-endpoints/{model_serving_endpoint_name}"
#     retry = True
#     total_wait = 0
#     while retry:
#       r = requests.get(url, headers=headers)
#       assert r.status_code == 200, f"Expected an HTTP 200 response when accessing endpoint info, received {r.status_code}"
#       endpoint = json.loads(r.text)
#       if "pending_config" in endpoint.keys():
#         seconds = 10
#         print("New config still pending")
#         if total_wait < 6000:
#           #if less the 10 mins waiting, keep waiting
#           print(f"Wait for {seconds} seconds")
#           print(f"Total waiting time so far: {total_wait} seconds")
#           time.sleep(10)
#           total_wait += seconds
#         else:
#           print(f"Stopping,  waited for {total_wait} seconds")
#           retry = False  
#       else:
#         print("New config in place now!")
#         retry = False
#   assert re.status_code == 200, f"Expected an HTTP 200 response, received {re.status_code}"

# func_create_endpoint('prediction')

# COMMAND ----------

# MAGIC %md # Next steps
# MAGIC - Explore the notebooks and experiments linked above.
# MAGIC - If the metrics for the best trial notebook look good, skip directly to the inference section.
# MAGIC - If you want to improve on the model generated by the best trial:
# MAGIC   - Go to the notebook with the best trial and clone it.
# MAGIC   - Edit the notebook as necessary to improve the model. For example, you might try different hyperparameters.
# MAGIC   - When you are satisfied with the model, note the URI where the artifact for the trained model is logged. Assign this URI to the `model_uri` variable in Cmd 12.

# COMMAND ----------

# MAGIC %md # Inference
# MAGIC You can use the model trained by AutoML to make predictions on new data. The examples below demonstrate how to make predictions on data in pandas DataFrames, or register the model as a Spark UDF for prediction on Spark DataFrames.

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## pandas DataFrame

# COMMAND ----------

model_uri = summary.best_trial.model_path
# model_uri = "<model-uri-from-generated-notebook>"

# COMMAND ----------

# X_test=training_pd.sample(50,random_state=42)

# COMMAND ----------

# import mlflow

# # Prepare test dataset
# y_test = X_test["Target"]
# X_test = X_test.drop("Target", axis=1)

# # Run inference using the best model
# model = mlflow.pyfunc.load_model(model_uri)
# predictions = model.predict(X_test)
# X_test["Target_predicted"] = predictions
# display(X_test)

# COMMAND ----------


