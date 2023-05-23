# Databricks notebook source
# DBTITLE 1,Paths
root_path = '/tmp/airbnb'
source_file = 's3://db-gtm-industry-solutions/data/rcg/airbnb/listings.csv'

# COMMAND ----------

# DBTITLE 1,Database settings
database = "xgboost_serving"

spark.sql(f"create database if not exists {database}")

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
model_name = "sfo_airbnb_price"
model_name_lgb = "sfo_airbnb_price__lgb"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/xgboost_serving'.format(username))

# COMMAND ----------

# DBTITLE 1,Set Host and Personal Access Token
import os
os.environ['DATABRICKS_URL'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) 
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

config = {}
config['root_path'] = root_path
config['database'] = database
config['model_name'] = model_name
config['model_name_lgb'] = model_name_lgb
config['source_file'] = source_file
config['serving_endpoint_name'] = serving_endpoint_name = 'sfo_airbnb_price'

# COMMAND ----------

print("Defined: ", config)
