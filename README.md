![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

In the world of Big Data, we are often managing large data sets that cannot fit into the memory of a single server. This becomes a barrier to the training of many Machine Learning models that require all the data on which they are to be trained to be loaded into a single pandas dataframe or numpy array.

To overcome this limitation, many popular models, such as those in the XGBoost family of models, have implemented capabilities allowing them to process data in Spark dataframes.  Spark dataframes overcome the memory limitations of a single server by allowing large datasets to be distributed over the combined resources of the multiple servers that comprise a Spark cluster. When models implement support for Spark dataframes, all or portions of the work they perform against the data can be distributed in a similar manner.

While this capability overcomes a key challenge to successfully training a model on a large dataset, it creates a dependency in the fitted model on the availability of a Spark cluster at the time of model deployment.  While in batch inference scenarios where the model is used as part of a Spark workflow, this dependency is no big deal.  But in real-time inference scenarios where individual records are typically sent to a model (often hosted behind a REST API) for scoring, this dependency can create overhead on the model host that slows response rates.

To overcome this challenge, we may wish to train a model in a distributed manner and then transfer the information learned during training to a non-distributed version of the model.  Such an approach allows us to eliminate the dependency on Spark during inference but does require us to carefully transfer learned information from one model to another.  In this accelerator, we'll demonstrate how this might be done for XGBoost and lightGBM models.

___
<bryan.smith@databricks.com> <sean.owen@databricks.com>

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| SynapseML  | Open-source library that simplifies the creation of massively scalable machine learning (ML) pipelines |  MIT | https://github.com/Microsoft/SynapseML  |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the `RUNME` notebook to any cluster and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. The job configuration is written in the RUNME notebook in json format. 
3. Execute the multi-step-job to see how the pipeline runs. 
4. You might want to modify the samples in the solution accelerator to your need, collaborate with other users and run the code samples against your own data. To do so start by changing the Git remote of your repository  to your organization’s repository vs using our samples repository (learn more). You can now commit and push code, collaborate with other user’s via Git and follow your organization’s processes for code development.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
