# Databricks notebook source
####################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook shows an example of a Model Training pipeline using Databricks Feature Store tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``clustering_demo_mlops/resources/model-workflow-resource.yml``
#
# Parameters:
# * env (required):                 - Environment the notebook is run in (staging, or prod). Defaults to "staging".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - Three-level name (<catalog>.<schema>.<model_name>) to register the trained model in Unity Catalog. 
#  
####################################################################################

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

# dbutils.widgets.removeAll()

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
    "dbfs:/FileStore/demo_data/Iris.csv",
    label="Path to the training data",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-clustering_demo_mlops-experiment",
    label="MLflow experiment name",
)
# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev.clustering_demo_mlops.clustering_demo_mlops-model", label="Full (Three-Level) Model Name"
)

# Sepal features table name
dbutils.widgets.text(
    "sepal_features_table",
    "dev.clustering_demo_mlops.iris_sepal_features",
    label="Sepal Features Table",
)

# Petal features table name
dbutils.widgets.text(
    "petal_features_table",
    "dev.clustering_demo_mlops.iris_petal_features",
    label="Petal Features Table",
)

dbutils.widgets.text(
    "output_prediction_ref", 
    "dev.clustering_demo_mlops.predictions_ref", 
    "Prediction Output Table Path"
    )


############ RETRAINING ################
dbutils.widgets.text(
    "model_retraining",
    "no",
    label="Retraining the Model?",
)

dbutils.widgets.text(
    "retraining_data_path",
    "dbfs:/FileStore/demo_data/Iris_resampled.csv",
    label="Retraining Data Path",
)

# COMMAND ----------

# DBTITLE 1,Define input and output variables
input_table_path = dbutils.widgets.get("training_data_path")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
output_prediction_ref = dbutils.widgets.get("output_prediction_ref")
model_retraining = dbutils.widgets.get("model_retraining")
retraining_data_path = dbutils.widgets.get("retraining_data_path")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1, Load raw data
raw_data = spark.read.format("csv").load(input_table_path, header=True, inferSchema=True)
if model_retraining == "yes":
    retraining_data = spark.read.format("csv").load(retraining_data_path, header=True, inferSchema=True)
    raw_data = raw_data.union(retraining_data)
raw_data.display()

# COMMAND ----------

# DBTITLE 1, Helper functions
# from datetime import timedelta, timezone
# import math
import mlflow.pyfunc
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from mlflow.tracking import MlflowClient


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

# DBTITLE 1,Count of rows
raw_data.count()

# COMMAND ----------

# DBTITLE 1, Create FeatureLookups
from databricks.feature_store import FeatureLookup
import mlflow

sepal_features_table = dbutils.widgets.get("sepal_features_table")
petal_features_table = dbutils.widgets.get("petal_features_table")

sepal_feature_lookups = [
    FeatureLookup(
        table_name=sepal_features_table,
        feature_names=["log_SepalLengthCm", "log_SepalWidthCm", "sepal_area"],
        lookup_key=["Id"],
        # timestamp_lookup_key=["rounded_pickup_datetime"],
    ),
]

petal_feature_lookups = [
    FeatureLookup(
        table_name=petal_features_table,
        feature_names=["log_PetalLengthCm", "log_PetalWidthCm", "petal_area"],
        lookup_key=["Id"],
    ),
]

# COMMAND ----------

# DBTITLE 1, Create Training Dataset
from databricks import feature_store

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()

# Since the rounded timestamp columns would likely cause the model to overfit the data
# unless additional feature engineering was performed, exclude them to avoid training on them.
exclude_columns = ["PetalLengthCm", "PetalWidthCm", "SepalLengthCm", "SepalWidthCm"]

fs = feature_store.FeatureStoreClient()

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
    raw_data,
    feature_lookups=sepal_feature_lookups + petal_feature_lookups,
    label="Species",
    exclude_columns=exclude_columns,
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the Feature Store, like `dropoff_is_weekend`
training_df.display()

# COMMAND ----------

training_cols = training_df.columns[2:-1]
training_cols

# COMMAND ----------

# training_df_pd = training_df.toPandas()
# training_df_pd

# COMMAND ----------

# MAGIC %md
# MAGIC Train a K-Means clustering model on the data returned by `TrainingSet.to_df`, then log the model with `FeatureStoreClient.log_model`. The model will be packaged with feature metadata.

# COMMAND ----------

# DBTITLE 1, Train model
# Spark imports
from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Feature engineering and scaling
assembler = VectorAssembler(inputCols=training_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# Enabling AutoLogger
mlflow.autolog()

# Define number of clusters
k = 3

# Create and train KMeans model using Pipeline
kmeans = KMeans(featuresCol="scaled_features", 
                k=k)

# Define the model pipeline and fit on the data
pipeline = Pipeline(stages=[assembler, scaler, kmeans])
model = pipeline.fit(training_df)

# Evaluate model
evaluator = ClusteringEvaluator(metricName="silhouette", featuresCol="scaled_features")
predictions_df = model.transform(training_df)
silhouette = evaluator.evaluate(predictions_df)
print(f"Silhouette score: {silhouette}")

# COMMAND ----------

# DBTITLE 1, Log model and return output.
# Log the trained model with MLflow and package it with feature lookup information.
fs.log_model(
    model,
    artifact_path="model_packaged",
    flavor=mlflow.spark,
    training_set=training_set,
    registered_model_name=model_name,
)

mlflow.end_run()


# Writing the prediction data to Delta table as a Reference Table
(predictions_df.drop("features", "scaled_features")
 .write.format("delta").mode("overwrite")
 .saveAsTable(output_prediction_ref)
 )

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)

# COMMAND ----------

# DBTITLE 1, Log model and return output.
# # Log the trained model with MLflow and package it with feature lookup information.
# fs.log_model(
#     model,
#     artifact_path="model_packaged",
#     flavor=mlflow.lightgbm,
#     training_set=training_set,
#     registered_model_name=model_name,
# )

# # The returned model URI is needed by the model deployment notebook.
# model_version = get_latest_model_version(model_name)
# model_uri = f"models:/{model_name}/{model_version}"
# dbutils.jobs.taskValues.set("model_uri", model_uri)
# dbutils.jobs.taskValues.set("model_name", model_name)
# dbutils.jobs.taskValues.set("model_version", model_version)
# dbutils.notebook.exit(model_uri)
