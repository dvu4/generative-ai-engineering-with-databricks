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
# MAGIC # Serving External Models with Model Serving
# MAGIC
# MAGIC **In this demo, we will focus on deploying GenAI applications.**
# MAGIC
# MAGIC Deployment is a key part of operationalizing our LLM-based applications. We will explore deployment options within Databricks and demonstrate how to achieve each one.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Determine the right deployment strategy for your use case.
# MAGIC * Deploy an external model to a Databricks Model Serving endpoint.
# MAGIC * Deploy a custom application to a Databricks Model Serving endpoint.
# MAGIC
# MAGIC **🚨 Important: Deploying custom models necessitates Model Serving with provisioned throughput and involves substantial compute resources. As such, this demonstration is designed to be instructor-led, and the model WILL NOT be deployed in the training workspace.**

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
# MAGIC ## Demo Overview
# MAGIC
# MAGIC In this demo, we will walk through basic deployment capabilities in Databricks. We'll discuss this in the following steps:
# MAGIC
# MAGIC 1. Access to custom model in Databricks Marketplace.
# MAGIC
# MAGIC 1. Deploy an external model to a Databricks Model Serving endpoint
# MAGIC
# MAGIC 1. Deploy a custom application to a Databricks Model Serving endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy an External Model with Databricks Model Serving
# MAGIC
# MAGIC While we have described and used tools like the AI Playground and Foundation Model APIs for querying common LLMs, there is sometimes a need to deploy more specific models as part of our applications.
# MAGIC
# MAGIC To achieve this, we can use **Databricks Model Serving**. Databricks Model Serving is a production-ready, serverless solution that simplifies real-time (and other types of) ML model deployment.
# MAGIC
# MAGIC Next, we will demonstrate the basics of Model Serving.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Getting a Model from Databricks Marketplace
# MAGIC
# MAGIC The simplest way to deploy a model in Model Serving is by getting an existing external model from the **Databricks Marketplace**.
# MAGIC
# MAGIC Let's explore the Marketplace for the Databricks-provided **CodeLlama Models**:
# MAGIC
# MAGIC 1. Head to the **[Databricks Marketplace](/marketplace)**.
# MAGIC
# MAGIC 1. Filter to "Models" **products provided by "Databricks"**.
# MAGIC
# MAGIC 1. Click on the **"CodeLlama Models"** tile.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![genai-as-04-catalog-llama-code-](../Includes/images/genai-as-04-catalog-llama-code-.png)
# MAGIC
# MAGIC These models are designed to help with generating code – there are a series of fine-tuned versions.
# MAGIC
# MAGIC We are interested in deploying one of these models using Databricks Model Serving, so we'll need to follow the below steps:
# MAGIC
# MAGIC 1. Click on the **Get instant access** button on the models page
# MAGIC
# MAGIC 1. Specify our parameters, including that we want to use the model in Databricks and our `catalog name`.
# MAGIC
# MAGIC 1. Acknowledge the terms and conditions
# MAGIC
# MAGIC 1. Click **Get instant access**
# MAGIC
# MAGIC This will clone the models to the specified catalog. We can see them in the Catalog Explorer. Note that the catalog is created under **shared catalogs**.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![marketplace-llama](../Includes/images/marketplace-llama.png)
# MAGIC
# MAGIC **Note:** An important point here is that these models are now stored in Unity Catalog. This means that they're secure and we can govern access to them using the familiar, general Unity Catalog tooling.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Getting a Model from `system.ai` Catalog
# MAGIC
# MAGIC The Databricks **`system.ai` catalog** is part of the Databricks GenAI and Unity Catalog offerings. It is a curated list of state-of-the-art open source models managed in system.ai in Unity Catalog. These models can be easily deployed using Model Serving Foundation Model APIs or fine-tuned with Model Training.
# MAGIC
# MAGIC To view registered models;
# MAGIC - From the left panel select **Catalog**.
# MAGIC - Select **system** catalog.
# MAGIC - Select **ai** schema. This will show a list of available models that you can serve.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![genai-as-04-system-ai-catalog](../Includes/images/genai-as-04-system-ai-catalog.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying a Model using Model Serving
# MAGIC
# MAGIC Once these models are in our catalog, we can deploy them directly to Databricks Model Serving by following the below steps:
# MAGIC
# MAGIC 1. Navigate to a specific model page in the Catalog.
# MAGIC
# MAGIC 1. Click the **Serve this Model** button.
# MAGIC
# MAGIC 1. Configure the served entity.
# MAGIC     * Name: `CodeLlama_13b_Python_hf`.
# MAGIC     * For served entities, select the model.
# MAGIC
# MAGIC 1. Click the **Confirm** button.
# MAGIC
# MAGIC 1. Configure the Model Serving endpoint.
# MAGIC
# MAGIC 1. **🚨 Notice: We won't deploy the model due to associated cost. In real use-case we would click the Create button.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confirming the Deployed Model
# MAGIC
# MAGIC When the Model Serving endpoint is operational, we'll see a screen like this:
# MAGIC <br>
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![genai-as-04-serving-llama-endpoint](../Includes/images/genai-as-04-serving-llama-endpoint.png)
# MAGIC
# MAGIC
# MAGIC **Note:** Notice the "Serving Deployment Status" field on the page. This will say "Not ready" until the model is deployed.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the Deployed Model
# MAGIC
# MAGIC More realistically, we can query the deployed model directly from our serving applications.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1 -Query via the UI
# MAGIC
# MAGIC We can query the model directly in Databricks to confirm everything is working using the **Query endpoint** capability.
# MAGIC
# MAGIC Sample query:
# MAGIC `{"prompt": "from spark.sql import functions as"}`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2 - Query the Deployed Model in AI Playground
# MAGIC
# MAGIC To test the model with AI Playground, select the deployed model and use chatbox to send queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 3 - Query the Deployed Model with the SDK
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC
# MAGIC # prompt to use as base for code completion. Feel free to change it to try different prompts.
# MAGIC prompt = """df1 = df.withColumn(
# MAGIC     "life_stage",
# MAGIC     when(col("age") < 13, "child")
# MAGIC     .when(
# MAGIC """
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC response = w.serving_endpoints.query(
# MAGIC     name="CodeLlama_13b_Python_hf", #name of the model serving endpoint
# MAGIC     prompt=prompt,
# MAGIC     max_tokens=50
# MAGIC )
# MAGIC
# MAGIC print(response.as_dict()["choices"][0]["text"])
# MAGIC ```
# MAGIC
# MAGIC **💡 Tip:** `max_tokens` defines the length of suggested code completion.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC At this point, you should be able to:
# MAGIC
# MAGIC * Determine the right deployment strategy for your use case.
# MAGIC * Deploy an external model to a Databricks Model Serving endpoint.
# MAGIC * Deploy a custom application to a Databricks Model Serving endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>