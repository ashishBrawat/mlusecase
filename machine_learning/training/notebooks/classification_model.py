# Databricks notebook source
##################################################################################
# Model Training Notebook
#
# This notebook shows an example of a Model Training pipeline using Delta tables.
# It is configured and can be executed as the "Train" task in the model_training_job workflow defined under
# ``classification_mlops/resources/model-workflow-resource.yml``
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
    "/databricks-datasets/adult/adult.data",
    label="Path to the training data",
)

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/dev-classification_mlops-experiment",
    label="MLflow experiment name",
)
# Unity Catalog registered model name to use for the trained model.
dbutils.widgets.text(
    "model_name", "dev.classification_mlops.classification_mlops-model",
    label="Full (Three-Level) Model Name"
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

# training_df = spark.read.format("delta").load(input_table_path)
schema = '''
  age DOUBLE,
  workclass STRING,
  fnlwgt DOUBLE,
  education STRING,
  education_num DOUBLE,
  marital_status STRING,
  occupation STRING,
  relationship STRING,
  race STRING,
  sex STRING,
  capital_gain DOUBLE,
  capital_loss DOUBLE,
  hours_per_week DOUBLE,
  native_country STRING,
  income STRING
'''
training_df = spark.read.format(
    "csv"
).schema(
    schema
).option(
    "header",
    "true"
).load(
    input_table_path
)
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
# MAGIC Train a LightGBM model on the data, then log and register the model with MLflow.

# COMMAND ----------
# DBTITLE 1, Train model

import mlflow
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import mlflow.lightgbm

# # Collect data into a Pandas array for training. Since the timestamp columns would likely
# # cause the model to overfit the data, exclude them to avoid training on them.
# columns = [col for col in training_df.columns if col not in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']]

# Generate features on the fly and train model:
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from mlflow.models import ModelSignature, infer_signature
import numpy as np

with mlflow.start_run():
    input_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
    output_cols = ['workclass_indexed', 'education_indexed', 'marital_status_indexed', 'occupation_indexed', 'relationship_indexed', 'race_indexed', 'sex_indexed', 'native_country_indexed', 'income_indexed']
    feature_cols = ['age', 'hours_per_week', 'education_indexed', 'race_indexed', 'sex_indexed', 'workclass_indexed', 'occupation_indexed', 'marital_status_indexed', 'relationship_indexed', 'native_country_indexed']
    indexer = StringIndexer(inputCols = input_cols, outputCols = output_cols)
    indexed_df = indexer.fit(training_df).transform(training_df)
    feature_assembler = VectorAssembler(
        inputCols = feature_cols,
        outputCol = 'features'
    )
    feature_df = feature_assembler.transform(indexed_df)
    training_data = feature_df.select('features', 'income_indexed')
    train_data, test_data = training_data.randomSplit([0.70,0.30])
    train_data.display()
    model = LogisticRegression(labelCol='income_indexed').fit(train_data)

    # Show summary stats:
    summary = model.summary

    # Predict on test dataset:
    predictions = model.evaluate(test_data)
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol='prediction',
        labelCol = 'income_indexed'
    )
    model_evaluations = evaluator.evaluate(
        predictions.predictions,
        {evaluator.metricName: "areaUnderPR"}
    )
    mlflow.log_metric("areaUnderCurve", model_evaluations)
    # featureassembler_edu = VectorAssembler(inputCols = ['age', 'hours_per_week', 'race_indexed', 'sex_indexed', 'workclass_indexed', 'occupation_indexed', 'marital_status_indexed','relationship_indexed', 'native_country_indexed', 'income_indexed'], outputCol = 'features')
    # output_edu = featureassembler_edu.transform(df2_lr)
    # finalized_data_edu = output_edu.select('features', 'education_indexed')
    # model_edu = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol='education_indexed')
    # train_data_edu, test_data_edu = finalized_data_edu.randomSplit([0.70,0.30])
    # model_edu = model_edu.fit(train_data_edu)
    # summary_edu = model_edu.summary
    # predictions_edu = model_edu.evaluate(test_data_edu)
    # evaluator_edu = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol = 'education_indexed')
    # print(evaluator_edu.evaluate(predictions_edu.predictions))

    # COMMAND ----------
    # DBTITLE 1, Log model and return output.

    # Take the first row of the training dataset as the model input example.
    input_example = training_data.toPandas()[['features', 'income_indexed']].iloc[[0]]

    # Log the trained model with MLflow
    mlflow.spark.log_model(
        model,
        artifact_path="income_evaluator",
        # The signature is automatically inferred from the input example and its predicted output.
        # input_example=input_example,
        registered_model_name=model_name,
        signature=infer_signature(
            np.array([test_data.features.values]),
            np.array([model_evaluations]))
    )

# The returned model URI is needed by the model deployment notebook.
model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.jobs.taskValues.set("model_name", model_name)
dbutils.jobs.taskValues.set("model_version", model_version)
dbutils.notebook.exit(model_uri)
