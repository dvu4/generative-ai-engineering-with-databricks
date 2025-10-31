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
# MAGIC # Deconstruct and Plan a Use Case
# MAGIC
# MAGIC In this demo, we will plan a compound AI system architecture using pure python. The goal is to define the scope, functionalities and constraints of the system to be developed. 
# MAGIC
# MAGIC We will create the system architecture to outline the structure and relationship of each component of the system. At this stage, we need to address the technical challenges and constraints of language model and frameworks to be used. 
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to*:
# MAGIC
# MAGIC * Apply a class architecture to the stages identified during Decomposition
# MAGIC
# MAGIC * Explain a convention that maps stage(s) to class methods
# MAGIC
# MAGIC * Plan what method attributes to use when writing a compound application
# MAGIC
# MAGIC * Identify various components in a compound app
# MAGIC

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
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **17.3.x-cpu-ml-scala2.13**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, **run the following code cells**.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01

# COMMAND ----------

# MAGIC %md
# MAGIC ## Overview of application
# MAGIC
# MAGIC In  notebook 1.1 - Deconstruct and Plan a Use Case, we created the sketch of our application below. Now it's time to fill in some of the details about each stage. Approach is more art than science, so this activity, we'll set convention for our planning that we want to define the following method attributes for each of our stages:
# MAGIC  * **Intent**: Provided from our previous exercise. Keep this around, when you get into actual coding this will be the description part of your docstring, see [PEP-257
# MAGIC ](https://peps.python.org/pep-0257/).
# MAGIC  * **Name**: YES! Naming things is hard. You'll get a pass in the exercise because we provide the name for you keep the content organized, but consider how you would have named things. Would you have used the `run_` prefix? Also remember that [PEP-8](https://peps.python.org/pep-0008/#method-names-and-instance-variables) already provides some conventions, specifically:
# MAGIC      * lowercase with words separated by underscores as necessary to improve readability
# MAGIC      * Use one leading underscore only for non-public methods and instance variables (not applicable to our exercise here)
# MAGIC      * Avoid name clashes with subclasses
# MAGIC  * **Dependencies**: When planning you'll likely already have an idea of approach or library that you'll need in each stage. Here, you will want to capture those dependencies. After looking at those dependencies you may notice that you'll need more     
# MAGIC  * **Signature**: These are the argument names and types as well as the output type. However, when working with compound apps it's helpful to have stage methods that are directly tied to an LLM type of chat or completion to take the form:
# MAGIC      * **model_inputs**: These are inputs that will change with each request and are not a configuration setting in the application.
# MAGIC      * **params**: These are additional arguments we want exposed in our methods, but will likely not be argumented by users once the model is in model serving.
# MAGIC      * **output**: This is the output of a method and will commonly take the form of the request response of a served model if one is called within the method.
# MAGIC
# MAGIC
# MAGIC  **NOTE**: At this point in planning, you don't necessarily need to get into the decisions about what arguments should be a compound app class entity and which should be maintained as class members.
# MAGIC
# MAGIC  **NOTE**: The separation of model_inputs and params is an important one. Compound applications accumulate a lot of parameters that will need to have defaults set during class instantiation or load_context calls. By separating those arguments in the planning phase, it will be easier to identify the parameter space that is configurable in your compound application. While not exactly the same, it may be helpful to think of this collection of parameters as hyperparameters - these are configurations will spend time optimizing prior to best application selection, but not set during inference.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1. `run_search` Stage Attributes
# MAGIC
# MAGIC Our initial stage is run_search. Our intent is to get a list of candidate content that will be useful to augment our qa_model. Check out [Query a Vector Search Endpoint](https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#query-a-vector-search-endpoint) to see examples using Databricks Vector Search.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-1.png)

# COMMAND ----------

html_run_search_1

# COMMAND ----------

displayHTML(html_run_search_1)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2. `run_summary` Stage Attributes
# MAGIC
# MAGIC We'll look at the run_summary method before we look at what needs to be done to run it asynchronously with run_augment. Recall that the output of our previous stage will be a dataclass, but ultimately that dataclass will need to be transformed into it's essential parts as a list of `tuple(id, content)`. Each iteration of run_summary will therefore have content and id variables in context from a single search result.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-2.png)

# COMMAND ----------

html_run_search_2

# COMMAND ----------

displayHTML(html_run_search_2)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3. (Optional) `run_augment` Stage Attributes
# MAGIC
# MAGIC Now let's plan what's necessary to execute `run_summary` asynchronously. The actual lesson of how asyncio works is beyond the scope of this instruction. However, we can still go through the planning exercise for a stage that runs asynchronously. If you are uncomfortable that you are planning without understanding exactly how every library works, take comfort in the fact that a lot of library discovery takes place during planning.
# MAGIC
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-3.png)

# COMMAND ----------

displayHTML(html_run_search_3)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4. `run_get_context` Stage Attributes
# MAGIC
# MAGIC This stage we just need to take the results of our `run_augment` stage and convert into a context for our QA stage. That's right, a stage doesn't have to include an LLM. While this stage could have been made part of the stage before or after, it's broken out as it's own entity so this compound app could have a non-LLM stage.
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-1.png)
# MAGIC <!--  -->

# COMMAND ----------

displayHTML(html_run_search_4)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5. `run_qa` Stage Attributes
# MAGIC
# MAGIC We'll look at the `run_qa` stage. To make things interesting we are going to use a Chat LLM type instead of a Completion LLM type like we used in run_summary. That means you should take a look at [Chat Model](https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-chat-completion-model) Docs. It may look the same as the completion model signature, but look closely, it's not.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-5.png)

# COMMAND ----------

displayHTML(html_run_search_5)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 6. (Optional) `main` Putting the Stages Together
# MAGIC
# MAGIC Time to take all that work planning each individual stage and putting it all together. Take a look below how our individual state planning greatly simplifies chaining our stages together. Not only is it succinct and easy to read, the logical separation and inclusion of dataclasses will also make the following future developer tasks much more straight forward:
# MAGIC
# MAGIC * **Testing**: Test can be written by each module we've created making it easier than would have otherwise been if we wrote our code as a script.
# MAGIC
# MAGIC * **Tracing**: Beyond the scope of this instruction, but adding trace decorators each stage will allow better evaluation for future improvements.
# MAGIC
# MAGIC * **Parameterizing**: By extraction out what all parameters are in the solution, we now have an understanding of our parameter space and ability to create new model versions by simply modifying parameters with no code updates.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-6.png)

# COMMAND ----------

displayHTML(html_run_search_6)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 7. (Optional) Compound App as Pyfunc Model
# MAGIC
# MAGIC Our main method is nice, it can take a string and return a string. However, while local execution is nice, we'd really like our model to be deployable and be evaluated just as if it were a foundation model or external model. MLFlow and Databricks accomplish model deployments through built-in model types. Since we are not using a model library, we can use MLFlow's general form of a `pyfunc` model, and we can use a specific subclass of `mlflow.PythonModel`.
# MAGIC

# COMMAND ----------

displayHTML(html_run_search_7)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC %md
# MAGIC
# MAGIC ### Full Multi-Endpoint Architecture
# MAGIC
# MAGIC
# MAGIC We've gone through all the work of identifying the dependencies which include both a Data Serving Endpoint and a couple model serving endpoints. We should have a look at what our final architecture is. Even in this straight forward compound application, you can see that it has a lot of endpoint dependencies. It's worth having this perspective to see all the serving endpoints that must be maintained.
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/compound-ai-system-s-10.png)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we planned a sample compound AI system using pure code. This demo showed how different components can be defined independently and then are linked together to build the system.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>