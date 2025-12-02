# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img
# MAGIC     src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png"
# MAGIC     alt="Databricks Learning"
# MAGIC   >
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB- Evaluation with Mosaic AI Agent Evaluation
# MAGIC
# MAGIC In this lab, you will have the opportunity to evaluate a RAG chain model **using Mosaic AI Agent Evaluation Framework.**
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC *In this lab, you will complete the following tasks:*
# MAGIC
# MAGIC - **Task 1**: Define a custom Gen AI evaluation metric.
# MAGIC
# MAGIC - **Task 2**: Conduct an evaluation test using the Agent Evaluation Framework.
# MAGIC
# MAGIC - **Task 3**: Analyze the evaluation results through the user interface.

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC
# MAGIC 2. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC
# MAGIC    - Click **More** in the drop-down.
# MAGIC    
# MAGIC    - In the **Attach to an existing compute resource** window, use the first drop-down to select your unique cluster.
# MAGIC
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC
# MAGIC 2. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC
# MAGIC 3. Wait a few minutes for the cluster to start.
# MAGIC
# MAGIC 4. Once the cluster is running, complete the steps above to select your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U -qq databricks-agents databricks-sdk databricks-vectorsearch databricks-langchain langchain==0.3.7 langchain-community==0.3.7 mlflow>=3.0 databricks-feature-engineering --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-04

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab Overview
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation Dataset
# MAGIC
# MAGIC In this lab, you will work with the same dataset utilized in the demos. This dataset contains sample queries along with their corresponding expected responses, which are generated using synthetic data.

# COMMAND ----------

display(DA.eval_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Model
# MAGIC
# MAGIC A RAG chain has been created and registered for use in this lab. The model details are provided below.
# MAGIC
# MAGIC **📌 Note:** If you are using your own workspace to run this lab, you must manually execute **`00 - Build Model / 00-Build Model`**.

# COMMAND ----------

import mlflow

catalog_name = "genai_shared_catalog_03"
schema_name = f"ws_{spark.conf.get('spark.databricks.clusterUsageTags.clusterOwnerOrgId')}"

mlflow.set_registry_uri("databricks-uc")

model_uri = f"models:/{catalog_name}.{schema_name}.rag_app/1"
model_name = f"{catalog_name}.{schema_name}.rag_app"

# COMMAND ----------

model_name

# COMMAND ----------

rag_model = mlflow.langchain.load_model(model_uri)
rag_model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1 - Define A Custom Metric
# MAGIC
# MAGIC For this task, define a custom metric to evaluate whether the generated **"ANSWER"** from the RAG chain is easily readable by a non-expert user.

# COMMAND ----------

# MAGIC %md
# MAGIC We can create our own evaluation metrics using prompting
# MAGIC
# MAGIC **The function `make_genai_metric_from_prompt` creates a custom evaluation metric for GenAI (Generative AI) models in MLflow, based on a prompt you define. It allows you to specify a prompt template, a model to use for evaluation (such as an LLM endpoint), and metadata about the metric. This metric can then be used in MLflow's evaluation workflows to systematically assess model outputs, such as checking for PII, correctness, or other criteria, using LLM-based judgments. This helps automate and standardize the evaluation of GenAI applications**

# COMMAND ----------

from mlflow.metrics.genai import make_genai_metric_from_prompt

## Prompt for LLM as judge to determine if the generated response is easily readable by non-academic or expert readers
eval_prompt = "Your task is to determine whether the generated response easily readable by non-academic or expert readers. This was the content: '{retrieved_context}'"

## Use Llama-3 as LLM
llm="endpoints:/databricks-meta-llama-3-3-70b-instruct"

## Define the metric
is_readable = make_genai_metric_from_prompt(
    name="is_readable",
    judge_prompt=eval_prompt,
    model=llm,
    metric_metadata={"assessment_type": "ANSWER"},
    greater_is_better=True
)

# COMMAND ----------

# Customized Evaluation Metric
is_readable

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2 - Run Evaluation Test
# MAGIC
# MAGIC Next, run an evaluation using the custom metric you defined. Ensure that you select **Mosaic AI Agent Evaluation** as the evaluation type.
# MAGIC

# COMMAND ----------

with mlflow.start_run(run_name="lab_04_agent_evaluation"):
    eval_results = mlflow.evaluate(
        data=DA.eval_df,
        model=model_uri,
        model_type="databricks-agent",
        extra_metrics=[is_readable]
    )

# COMMAND ----------

display(eval_results.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3 - Review Evaluation Results
# MAGIC
# MAGIC Review the evaluation results in the **Experiments** section. Examine the following information regarding this evaluation:
# MAGIC
# MAGIC - Token usage
# MAGIC
# MAGIC - Model metrics
# MAGIC
# MAGIC - Results of the custom metric defined earlier ("readability")

# COMMAND ----------

displayHTML('<img src="/files/tables/agent_ai_customized_evaluation_metrics.png" width="1100"/>')

# COMMAND ----------

# MAGIC %md
# MAGIC ## OPTIONAL - Collect Human Feedback via Databricks Review App
# MAGIC
# MAGIC The Databricks Review App stages the LLM in an environment where expert stakeholders can engage with it—allowing for conversations, questions, and more. This setup enables the collection of valuable feedback on your application, ensuring the quality and safety of its responses.
# MAGIC
# MAGIC **Stakeholders can interact with the application bot and provide feedback on these interactions. They can also offer feedback on historical logs, curated traces, or agent outputs.**
# MAGIC
# MAGIC **🚨 Important Note:**
# MAGIC
# MAGIC This step is **for instructors only**. If you are using your own environment, you can comment out the cells and run them to deploy the model and access the Review App.
# MAGIC
# MAGIC **⚠️ Warning: Permission Required**
# MAGIC
# MAGIC If you are not an instructor and try to run this step without the required permissions, you may encounter the `PermissionDenied` error.
# MAGIC
# MAGIC **How to Proceed:**
# MAGIC - **If you are an instructor**, after running this code, you must grant permissions to users as needed.
# MAGIC - **If you are not an instructor**, do **not** run this step without getting permission from an instructor. Otherwise, you will encounter a permission error and won’t be able to proceed.

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
import mlflow
from databricks import agents

# Deploy the model with the agent framework
deployment_info = agents.deploy(
    model_name, 
    model_version=1,
    scale_to_zero=True,
    budget_policy_id=None)

# Wait for the Review App and deployed model to be ready
w = WorkspaceClient()
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")

while ((w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY) or (w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS)):
    print(".", end="")
    time.sleep(30)

print("\nThe endpoint is ready!", end="")

# COMMAND ----------

print(f"Endpoint URL    : {deployment_info.endpoint_url}")
print(f"Review App URL  : {deployment_info.review_app_url}")

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
try:
    print("\nCleaning up resources...")
    # Delete endpoint
    agents.delete_deployment(model_name=model_name)
    print(f"Deleted agent endpoint: {model_name}")
    # Delete payload table
    base_table_name = model_name.split(".")[-1]  # rag_app_<suffix>
    payload_table_name = f"{catalog_name}.{schema_name}.{base_table_name}_payload"
    # Drop the table
    spark.sql(f"DROP TABLE IF EXISTS {payload_table_name}")
    print(f"Deleted table: {payload_table_name}")
    # Delete feedback model
    feedback_model_name = f"{catalog_name}.{schema_name}.feedback"
    client.delete_registered_model(name=feedback_model_name)
    print(f"Deleted feedback model: {feedback_model_name}")
except:
    print("An error occured while trying to delete resources. Please try to delete resources manually! Delete these resources: Model Serving Endpoint, Payload Table, and Feedback Model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you evaluated a RAG chain using the Mosaic AI Evaluation Framework library. You began by loading the dataset and RAG model. Then, you defined a custom metric and initiated the evaluation process. Finally, you reviewed the results through the user interface.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>