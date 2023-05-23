# Databricks notebook source
# MAGIC %md The purpose of this notebook is to demonstrate how to train an XGBoost model in a distributed manner using Spark and then deploy it for lightweight model serving.  This notebook is available at https://github.com/databricks-industry-solutions/xgboost-serving-enabler.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In the world of Big Data, we are often managing large data sets that cannot fit into the memory of a single server. This becomes a barrier to the training of many Machine Learning models that require all the data on which they are to be trained to be loaded into a single pandas dataframe or numpy array.
# MAGIC
# MAGIC To overcome this limitation, many popular models, such as those in the XGBoost family of models, have implemented capabilities allowing them to process data in Spark dataframes.  Spark dataframes overcome the memory limitations of a single server by allowing large datasets to be distributed over the combined resources of the multiple servers that comprise a Spark cluster. When models implement support for Spark dataframes, all or portions of the work they perform against the data can be distributed in a similar manner.
# MAGIC
# MAGIC While this capability overcomes a key challenge to successfully training a model on a large dataset, it creates a dependency in the fitted model on the availability of a Spark cluster at the time of model deployment.  While in batch inference scenarios where the model is used as part of a Spark workflow, this dependency is no big deal.  But in real-time inference scenarios where individual records are typically sent to a model (often hosted behind a REST API) for scoring, this dependency can create overhead on the model host that slows response rates.
# MAGIC
# MAGIC To overcome this challenge, we may wish to train a model in a distributed manner and then transfer the information learned during training to a non-distributed version of the model.  Such an approach allows us to eliminate the dependency on Spark during inference but does require us to carefully transfer learned information from one model to another.  In this notebook, we'll demonstrate how this might be done, focusing on an XGBoost model.
# MAGIC
# MAGIC **NOTE** Throughout the remainder of this blog, we will refer to the distributed model as the *pyspark model* and the non-distributed model as the *sklearn model*.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn
from pyspark.sql.types import *

import mlflow
import os
import requests
import numpy as np
import pandas as pd
import json
import time

# for spark model
from xgboost.spark import SparkXGBRegressor
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.linalg import Vectors

# for nonspark model
import xgboost as xgb
import sklearn
from sklearn.impute import SimpleImputer as sklearn_SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder as sklearn_LabelEncoder
from sklearn.preprocessing import OneHotEncoder as sklearn_OneHotEncoder
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from sklearn.metrics import mean_squared_error

# COMMAND ----------

# DBTITLE 1,Setup configurations we use throughout the accelerator
# MAGIC %run ./util/notebook-config

# COMMAND ----------

# MAGIC %md ##Step 1: Access the Data
# MAGIC
# MAGIC To demonstrate this technique, we will make use of an Airbnb dataset providing details about various properties for short-term lease in the San Francisco area and their lease price.  This dataset has been made available for [download](https://www.kaggle.com/datasets/jeploretizo/san-francisco-airbnb-listings?resource=download) from the Kaggle website.  For this notebook, we provide the data at a publicly accessible cloud storage path `config['source_file']`:

# COMMAND ----------

# DBTITLE 1,Verify the Source Data File
display(
  dbutils.fs.ls(config['source_file'])
  )

# COMMAND ----------

# MAGIC %md The dataset consists of one record per listing as of a specific point in time.  A large number of attributes, many of which are not needed for our purposes, are included in the dataset.  Some of these attributes contain multi-line text (enclosed in double quotes) that we will need to carefully read (as shown below).
# MAGIC
# MAGIC As we read the data, we will reduce the size of the dataset by selecting only those fields we intend to use to train our model.  We will also eliminate anomalous records with invalid prices or unrealistic minimum stays:

# COMMAND ----------

# DBTITLE 1,Access the Raw Data
raw_df = (
  spark
    .read
    .csv(
      path=config['source_file'],
      header=True,
      inferSchema=True,
      escape='"', # multi-line descriptions wrapped in double-quotes
      multiLine=True
      )
    .select( # we only need some of the available columns
      'host_is_superhost',
      'cancellation_policy',
      'instant_bookable',
      'host_total_listings_count',
      'neighbourhood_cleansed',
      'latitude',
      'longitude',
      'property_type',
      'room_type',
      'accommodates',
      'bathrooms',
      'bedrooms',
      'beds',
      'bed_type',
      'minimum_nights',
      'number_of_reviews',
      'review_scores_rating',
      'review_scores_accuracy',
      'review_scores_cleanliness',
      'review_scores_checkin',
      'review_scores_communication',
      'review_scores_location',
      'review_scores_value',
      'price'
      )
    .withColumn('price', fn.translate('price', '$,', '').cast(DoubleType())) # remove unnecessary chars from price and convert to double
    .filter('price > 0.0') # remove rows with invalid price
    .filter('minimum_nights <= 365') # remove rows with invalid min nights
    )

display(raw_df)

# COMMAND ----------

# MAGIC %md We can see from this data that we have a combination of categorical and continuous variables which we will need to transform in preparation for model training.  
# MAGIC
# MAGIC One of these transformations is handling the NULL values in many of the numerical attributes found in the dataset. While we can make use of imputation to replace NULL values with appropriate alternative values, we found one small difference between how the pyspark Imputer transform and the sklearn SimpleImputer transform approach this.  Somehow, the pyspark Imputer is keeping track of which values were replaced without indicator fields.  And while the sklearn SimpleImputer supports indicator fields, the use of this setting creates a divergence in the data flowing through the pipeline that yields slightly different results.
# MAGIC
# MAGIC What we found solves this problem is manually creating indicator fields prior to data being submitted to either pipeline.  When we do this, we get consistent results from the two solutions.  While we will tackle this outside the pipeline, we will need to remember to add this dataset transformation step to our model prior to deployment:
# MAGIC
# MAGIC **NOTE** While this step seems highly specific, it illustrates that you will want to carefully examine the results of both the pyspark and the sklearn models when performing this kind of work.  If there are differences in behaviours with transformations, you will need to figure out steps to resolve them that may not always be intuitive.

# COMMAND ----------

# DBTITLE 1,Identify Categorical and Numerical Fields
# identify categorical features
categorical_cols = [field for (field, dtype) in raw_df.dtypes if dtype == 'string']

# identify numerical features
numerical_cols = [field for (field, dtype) in raw_df.dtypes if ((dtype in ['double','int']) & (field != 'price') )]

# present fields
print(f"Categorical: {categorical_cols}")
print(f"Numerical:   {numerical_cols}")

# COMMAND ----------

# DBTITLE 1,Add NULL Indicator Fields for All Numerical Fields
raw_df_w_na = raw_df
for c in numerical_cols:
    raw_df_w_na = raw_df_w_na.withColumn(f"{c}_na", fn.expr(f"CASE WHEN {c} IS NULL THEN 1 ELSE 0 END"))

display(raw_df_w_na)

# COMMAND ----------

# MAGIC %md Before tackling the remaining transformations, we will first split our data into training and testing sets:

# COMMAND ----------

# DBTITLE 1,Split the Data into Training and Testing Sets
# split the data
(train_df, test_df) = raw_df_w_na.randomSplit([.8, .2], seed=42)

# print count of records in each set
print(f"Training instances: {train_df.cache().count()}")
print(f"Testing instances:  {test_df.cache().count()}")

# COMMAND ----------

# MAGIC %md ##Step 2: Train the PySpark Model
# MAGIC
# MAGIC For our model to make use of the categorical features, we will need to perform a one-hot encoding by which each unique value (except for one) is mapped to a binary field.  This is done using the Spark MLlib [OneHotEncoder](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html).  This OneHotEncoder expects incoming categorical values to be integer values so that we'll need to use the Spark MLlib [StringIndexer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html) to map unique values to integers as follows:

# COMMAND ----------

# DBTITLE 1,Define Transformations for Categorical Features
# convert categorical strings into integer indexes
index_cols = [c + 'Index' for c in categorical_cols]
string_indexer = StringIndexer(
  inputCols=categorical_cols, 
  outputCols=index_cols,  
  stringOrderType='alphabetAsc', 
  handleInvalid='keep'
  )

# apply one-hot encoding to integer indexes
ohe_cols = [c + 'OHE' for c in categorical_cols] 
ohe_encoder = OneHotEncoder(
  inputCols=index_cols, 
  outputCols=ohe_cols, 
  handleInvalid='error', 
  dropLast=True
  )

# COMMAND ----------

# MAGIC %md Next, we need to ensure our numerical columns do not contain any missing values.  If you are familiar with other implementations of the XGBoost models, you may be curious why we need to impute missing values here.  The reason is not because of the Spark-distributed XGBoost model but instead the VectorAssembler we will use in a bit.  The VectorAssembler requires that all incoming features have values assigned to them:

# COMMAND ----------

# DBTITLE 1,Define Transformations for Numerical Fields
imputer = Imputer(
  strategy='median', 
  inputCols=numerical_cols, 
  outputCols=numerical_cols
  )

# COMMAND ----------

# MAGIC %md Our transformed values, along with the missing value indicator fields, are then passed to the Spark MLlib [VectorAssembler](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html) to bring all the features together as a vector, aka a field of *pyspark.ml.linalg.Vector* type.
# MAGIC
# MAGIC If you are familiar with the use of XGBoost models, you may be curious why it is we need to assemble a vector at this point. With traditional XGBoost models, data are often passed in directly from pandas dataframes (or numpy arrays). While the use of a pandas dataframe gives the sense that XGBoost is directly engaging specific fields, it is in fact viewing each record as an array.  In Spark MLlib, an array and a vector are very similar constructs.  The explicit use of a vector with Spark-distributed XGBoost models makes clear how data is being handed to the model and aligns the approach with what is typically used with Spark MLlib:

# COMMAND ----------

# DBTITLE 1,Define Vector Assembler
assembler_cols =  ohe_cols + numerical_cols + [f"{c}_na" for c in numerical_cols]  #this order needs to match order when building sklearn pipeline
vec_assembler = VectorAssembler(
  inputCols=assembler_cols,
  outputCol='features'
  )

# COMMAND ----------

# MAGIC %md We can then configure the [Spark-distributed XGBoost Regressor](https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html#sparkxgbregressor) as follows.  Please note that the *num_workers* parameter reflects the number of executors available in our Spark cluster:

# COMMAND ----------

# DBTITLE 1,Define Model
xgb_regressor = SparkXGBRegressor(
  num_workers=sc.defaultParallelism, # default parallelism provides number of available executors
  label_col='price'
  )

# COMMAND ----------

# MAGIC %md All this logic is then wrapped up in a pipeline so that raw data can be passed to the model for transformation prior to model training and inference:

# COMMAND ----------

# DBTITLE 1,Assemble Transformation + Model Pipeline
pyspark_pipeline = Pipeline(stages=[string_indexer, ohe_encoder, imputer, vec_assembler, xgb_regressor])

# COMMAND ----------

# MAGIC %md With the model pipeline assembled, we can now train it across the Databricks cluster. Please note that we are not performing the hyperparameter tuning that's typically done with XGBoost models as our focus is on deployment patterns.  If you'd like to see an example of how hyperparameter tuning can be performed with Spark-distributed XGBoost models, please check out the sample notebook provided [here](https://docs.databricks.com/machine-learning/train-model/xgboost-spark.html):

# COMMAND ----------

# DBTITLE 1,Train the Model
pyspark_trained = pyspark_pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md With model training completed, we might perform model evaluation as follows:

# COMMAND ----------

# DBTITLE 1,Generate Predictions
pred_df = pyspark_trained.transform(test_df)
display(
  pred_df
    .select('features','prediction','price')
  )

# COMMAND ----------

# DBTITLE 1,Score the Predictions
# define evaluation
evaluator = RegressionEvaluator(
  metricName='rmse',
  labelCol='price',
  predictionCol='prediction'
  )

# evaluate predictions
pyspark_rmse = evaluator.evaluate(pred_df)

# return the evaluation metric
print(f'RMSE: {pyspark_rmse}')

# COMMAND ----------

# MAGIC %md ##Step 3: Create a Sklearn Model
# MAGIC
# MAGIC At this point, we have a model trained using distributed computing that's ready for persistence and deployment for batch inference scenarios. But our goal is to prepare a model for real-time inference without a dependency on Spark.  We can achieve this by extracting the information learned during model training, aka the *booster*, from our trained pyspark model and loading that data into a non-distributed sklearn model.  To do this, let's start by accessing our pyspark model and it's booster data:

# COMMAND ----------

# DBTITLE 1,Retrieve Booster from Distributed Model
# retrieve the trained model
pyspark_model = pyspark_trained.stages[-1]

#get booster from trained model
booster = pyspark_model.get_booster()

# display booster info
display(booster.trees_to_dataframe())

# COMMAND ----------

# MAGIC %md The booster metadata can then be transferred to a sklearn (non-distributed) XGBoost model as follows:

# COMMAND ----------

# DBTITLE 1,Load Booster to Sklearn Model
#save booster to temp file
dbutils.fs.mkdirs('dbfs:/tmp') # ensure target folder exists in dbfs file system
booster.save_model('/dbfs/tmp/jbooster.json') # access target folder as part of local file system

#instantiate new xgboost model
sklearn_model = xgb.XGBRegressor()

#load booster to new xgboost model instance
sklearn_model.load_model(fname='/dbfs/tmp/jbooster.json')

# COMMAND ----------

# MAGIC %md With our model properly configured, we now need to turn our attention to the transformations that preceded it in the pipeline.  Just as a reminder, these transformations are:</p>
# MAGIC
# MAGIC 0. String Indexer
# MAGIC 1. OneHot Encoder
# MAGIC 2. Imputer
# MAGIC 3. Vector Assembler
# MAGIC
# MAGIC We will use metadata in the trained transformations to create sklearn equivalent transformations with the same logic and value-mappings.
# MAGIC
# MAGIC First up are the categorical feature transformations.  We will retrieve the string labels captured by our StringIndexer and convert these into classes 

# COMMAND ----------

# DBTITLE 1,Define Transformations for Categorical Features
# extract lists of labels used by string indexer as class definitions
classes = [
  sklearn_LabelEncoder().fit(arr).classes_ for arr in pyspark_trained.stages[0].labelsArray
  ] 

# apply class definitions to one hot encoder
sklearn_ohe = sklearn_OneHotEncoder(
  categories=[c.tolist() for c in classes],
  handle_unknown='ignore'
  )

# COMMAND ----------

# MAGIC %md Next, we'll leverage the instructions employed by our Imputer to configure an equivalent transformation:

# COMMAND ----------

# DBTITLE 1,Define Transformations for Numerical Fields
# get replacement values for numerical columns learned by pyspark imputer
defaults = pyspark_trained.stages[2].surrogateDF.collect()[0]

# build up per-column imputers with appropriate imputer 
sklearn_imputers=[]
for c in numerical_cols:
  sklearn_imputers += [(
    f"{c}_imputer", # name the imputer <column_name>_imputer
    sklearn_SimpleImputer( # define imputer to use a constant value
      missing_values=pyspark_trained.stages[2].getMissingValue(),
      strategy='constant',
      fill_value=defaults[c]
      ),
    [c] # apply to this column (as a list)
    )]

# COMMAND ----------

# DBTITLE 1,Assemble Transformations
# assemble column transformation logic
col_trans = ColumnTransformer(
  [('sklearn_ohe', sklearn_ohe, categorical_cols)] + sklearn_imputers,
  remainder='passthrough'
  )

# run at least one record through transforms so that they are recognized as "fit"
col_trans.fit(
  train_df.limit(1).drop('price').toPandas()
  ) 

# COMMAND ----------

# MAGIC %md With our transforms configured, this would be a good time for us to send a record through both the pyspark and the sklearn pipelines to ensure that they are returning the same values in **EXACTLY** the same order as one another.  If any values are out of order, the sklearn model, configured using a booster from the pyspark model, will apply its logic to the wrong fields. 
# MAGIC
# MAGIC To help with this, we'll present each set of features as a sparse vector.  Be sure to note the index number associated with each value as you make your comparisons:

# COMMAND ----------

# DBTITLE 1,Apply Feature Transforms to Both Sets of Transforms
# retrieve a sample row
sample_df = train_df.orderBy(fn.rand()).limit(1).drop('price')
sample_pd = sample_df.toPandas()

# apply transformations with pyspark pipeline
pyspark_features = pyspark_trained.transform(sample_df).select('features')

# apply transformations with sklearn transforms
sklearn_features = col_trans.transform(sample_pd)

# COMMAND ----------

# DBTITLE 1,Compare Transformed Features
# compare results
print(
  pyspark_features.collect()[0],
  )
print('\n')
print(
  'sklearn    ', # print this charstrings to help align string representations
  str(
    [Vectors.sparse(
      len(list(sklearn_features[0])),
      {k:v for k, v in enumerate(list(sklearn_features[0])) if v != 0}
      )]
    )
  )

# COMMAND ----------

# MAGIC %md Now that we are confident we have our inputs in the same order, we can complete the setup of the sklearn pipeline:

# COMMAND ----------

# DBTITLE 1,Assemble Transformation + Model Pipeline
# assemble sklearn pipeline with model
sklearn_pipeline = sklearn_Pipeline(steps=[('col_trans', col_trans), ('booster', sklearn_model)])

# COMMAND ----------

# MAGIC %md Evaluating our model using the test dataframe, we can verify we are getting the same evaluation metric as above, confirming we've successfully transferred the learned information form the pyspark pipeline to the sklearn pipeline:

# COMMAND ----------

# DBTITLE 1,Evaluate Pipeline
# pull test data to pandas dataframe
test_pd = test_df.toPandas()

# get price as y
y = test_pd['price']

# predict price as yhat
yhat = sklearn_pipeline.predict(
  test_pd.drop(['price'],axis=1)
  )

# calculate rmse using y and yhat
sklearn_rmse = mean_squared_error(
  y,
  yhat,
  squared=False
  )

# display results
print(f'RMSE (spark):   {pyspark_rmse}')
print(f'RMSE (sklearn): {sklearn_rmse}')

# COMMAND ----------

# MAGIC %md ##Step 4: Deploy the Core Model for Real-Time Inference
# MAGIC
# MAGIC Now we can turn our attention to model deployment. Our goal will be to prepare our model for deployment to Databricks [model serving](https://docs.databricks.com/machine-learning/model-serving/index.html) which will present our model behind a REST API for real-time inference.
# MAGIC
# MAGIC To support this, we'll need to create a custom wrapper for our model.  We wouldn't typically need this but we do have some additional data transformation steps, *i.e.* the creation of indicator fields for our numerical values, that we need to address as the model is called.  The definition of our wrapper is as follows:
# MAGIC
# MAGIC **NOTE** In the wrapper, you will notice we are explicitly identifying the fields that will drive the creation of the *_na* indicator fields.  We tried using data type checking but weren't getting consistent results most likely due to how data types were being translated as the incoming data was translated back into a pandas dataframe.

# COMMAND ----------

# DBTITLE 1,Define Class Wrapper for Model
class modelWrapper(mlflow.pyfunc.PythonModel):

  def __init__(self, model):
    self.model = model

  # initialize model
  def load_context(self, context):
    self.model=self.model

  # define prediction logic
  def predict(self, context, df):

    # import required libraries
    import pandas as pd
    import numpy as np

    # copy input df ahead of modification 
    _df = df.copy(deep=True)

    # for numerical fields in dataframe 
    for c in numerical_cols:
      # add an indicator field to dataframe
      _df[f"{c}_na"] =  np.where(_df[c] is np.nan, 1, 0)
    
    # get prediction
    result = self.model.predict(_df)

    return result

# wrap the sklearn model
wrapped_model = modelWrapper(sklearn_pipeline)

# COMMAND ----------

# MAGIC %md Next we need to ensure the host that runs our model is aware of our dependency on XGBoost:

# COMMAND ----------

# DBTITLE 1,Define Environment Configuration
# get base environment configuration
conda_env = mlflow.pyfunc.get_default_conda_env()

# define packages required by model
packages = [
  f'xgboost=={xgb.__version__}',
  f'scikit-learn=={sklearn.__version__}'
  ]

# add required packages to environment configuration
conda_env['dependencies'][-1]['pip'] += packages

print(
  conda_env
  )

# COMMAND ----------

# MAGIC %md And now we can log our model to mlflow:

# COMMAND ----------

# DBTITLE 1,Log Model to MLflow
with mlflow.start_run() as run:

    mlflow.pyfunc.log_model(
        artifact_path='model',
        python_model=wrapped_model,
        conda_env=conda_env,
        registered_model_name=model_name
    )

# COMMAND ----------

# MAGIC %md Our model has been placed in the [mlflow model registry](https://docs.databricks.com/mlflow/model-registry.html) where it can be reviewed and moved into various stages for appropriate use in our environment.  We would typically have a process for evaluation and rollout that would manage the movement between stages, but for the purposes of this demonstration, we'll simply elevate our model to production status:

# COMMAND ----------

# DBTITLE 1,Move Model to Production Status
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
model_version = client.search_model_versions(f"name='{model_name}'")[0].version

# move model version to production
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='production'
  )      

# COMMAND ----------

# MAGIC %md We can now retrieve our production-ready model and send a record to it to test its functionality:

# COMMAND ----------

# DBTITLE 1,Retrieve Sample Record to Score
# retrieve a sample row
sample_df = train_df.orderBy(fn.rand()).limit(1).drop('price')
for c in sample_df.columns: # drop indicator fields
  if c[-3:]=='_na': sample_df = sample_df.drop(c)
  
sample_pd = sample_df.toPandas()

# COMMAND ----------

# DBTITLE 1,Test the Model
# retrieve model
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# generate a prediction
loaded_model.predict(sample_pd) 

# COMMAND ----------

# MAGIC %md Now that we are comfortable our logged model is working properly, we can deploy it to model serving. To deploy our model, you can follow the step-by-step UI screenshot guide below, or execute the next code block where we use API to automate the steps in the screenshots.
# MAGIC
# MAGIC ----
# MAGIC **UI-based Guide**
# MAGIC
# MAGIC we need to reconfigure our Databricks workspace for Machine Learning.  We can do this by clicking on the drop-down at the top of the left-hand navigation bar and selecting *Machine Learning*.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/search_change_workspace.png'>
# MAGIC
# MAGIC Once we've done that, we should be able to select *Serving* from that same left-hand navigation bar.
# MAGIC
# MAGIC Within the Serving Endpoints page, click on the *Create Serving Endpoint* button.  Give your endpoint a name, select your model - it may help to start typing the model name to limit the search - and then select the model version.  Select the compute size based on the number of requests expected and select/deselect the *scale to zero* option based on whether you want the service to scale down completely during a sustained period of inactivity.  (Spinning back up from zero does take a little time once a request has been received.)
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/xgb_create_endpoint.PNG' width=80%>
# MAGIC
# MAGIC Click the *Create serving endpoint* button to deploy the endpoint and monitor the deployment process until the *Serving Endpoint State* is *Ready*.  
# MAGIC
# MAGIC ----
# MAGIC **API-based Guide**
# MAGIC
# MAGIC We provide code to create or update model serving endpoints according to the configuration below:

# COMMAND ----------

# MAGIC %run ./util/create-update-serving-endpoint

# COMMAND ----------

# DBTITLE 1,Use the defined function to create or update the endpoint
served_models = [
    {
      "name": "XGBoost",
      "model_name": model_name,
       "model_version": model_version,
       "workload_size": "Small",
       "scale_to_zero_enabled": True
    }
]
traffic_config = {"routes": [{"served_model_name": "XGBoost", "traffic_percentage": "100"}]}

# kick off endpoint creation/update
if not endpoint_exists(config['serving_endpoint_name']):
  create_endpoint(config['serving_endpoint_name'], served_models)
else:
  update_endpoint(config['serving_endpoint_name'], served_models)

# COMMAND ----------

# MAGIC %md Next, we can use the code below to setup a function to query this endpoint.  This code is a slightly modified version of the code accessible through the *Query Endpoint* UI accessible through the serving endpoint page:

# COMMAND ----------

# DBTITLE 1,Define Functions to Query the Endpoint
endpoint_url = f"""{os.environ['DATABRICKS_URL']}/serving-endpoints/{model_name}/invocations"""

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = endpoint_url
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

# COMMAND ----------

# MAGIC %md And now we can test the endpoint:

# COMMAND ----------

# DBTITLE 1,Test the Model Serving Endpoint
score_model( 
   sample_pd
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
