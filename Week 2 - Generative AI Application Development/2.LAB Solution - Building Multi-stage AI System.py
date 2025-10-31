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
# MAGIC # LAB - Building Multi-stage AI System
# MAGIC
# MAGIC In this lab, you will construct a multi-stage reasoning system using Databricks' features and LangChain.
# MAGIC
# MAGIC You will start by building the first chain, which performs a search using a dataset containing product descriptions from Etsy. Following that, you will create the second chain, which creates an image for the proposed product. Finally, you will integrate these chains to form a complete multi-stage AI system.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1:** Create a Vector Store
# MAGIC
# MAGIC * **Task 2:** Build the First Chain (Vector Store Search)
# MAGIC
# MAGIC * **Task 3:** Build the Second Chain (Product Image)
# MAGIC
# MAGIC * **Task 4:**  Integrate Chains into a Multi-chain System
# MAGIC
# MAGIC **📝 Your task:** Complete the **`<FILL_IN>`** sections in the code blocks and follow the other steps as instructed.

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

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install -U -qqq databricks-sdk databricks-vectorsearch==0.60 'mlflow-skinny[databricks]==3.4.0' databricks-langchain==0.8.0 langchain==0.3.7 langchain-community==0.3.7 youtube_search==2.1.2 Wikipedia==1.4.0 
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02LAB

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dataset
# MAGIC
# MAGIC Before you start building the AI chain, you need to load and prepare the dataset and save it as a Delta table.  
# MAGIC For this demo, we will use the **[Databricks Documentation Dataset](/marketplace/consumer/listings/03bbb5c0-983d-4523-833a-57e994d76b3b?o=1120757972560637)** available from the Databricks Marketplace.
# MAGIC
# MAGIC This dataset contains documentation pages with associated `id`, `url`, and `content`.  
# MAGIC We will format the data to create a single unified `document` field combining the URL and content, which will then be used to build a Vector Store.
# MAGIC
# MAGIC The table will be created for you in the next code block.

# COMMAND ----------

## Load the docs table from Unity Catalog
vs_source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.docs"
create_docs_table(vs_source_table_fullname)
## Display a sample of the data
display(spark.sql(f"SELECT * FROM {vs_source_table_fullname}"))

# COMMAND ----------

# MAGIC %md
# MAGIC %md 
# MAGIC ## Create a Vector Store
# MAGIC
# MAGIC In this step, you will compute embeddings for the dataset containing information about the products and store them in a Vector Search index using Databricks Vector Search.
# MAGIC
# MAGIC **🚨IMPORTANT: Vector Search endpoints must be created before running the rest of the demo. These are already created for you in Databricks Lab environment.**
# MAGIC

# COMMAND ----------

## Assign Vector Search endpoint by username
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_name = vs_endpoint_prefix + str(get_fixed_integer(DA.unique_name("_")))
print(f"Assigned Vector Search endpoint name: {vs_endpoint_name}.")

# COMMAND ----------

## Index table name
vs_index_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.doc_embeddings"

## Store embeddings in vector store
## NOTE: we're using 'content' as the embedding column
create_vs_index(vs_endpoint_name, vs_index_table_fullname, vs_source_table_fullname, "document" )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Build the First Chain (Vector Store Search)
# MAGIC
# MAGIC In this task, you will create first chain that will search for product details from the Vector Store using a dataset containing product descriptions.
# MAGIC
# MAGIC **Instructions:**
# MAGIC    - Configure components for the first chain to perform a search using the Vector Store.
# MAGIC    - Utilize the loaded dataset to generate prompts for Vector Store search queries.
# MAGIC    - Set up retrieval to extract relevant product details based on the generated prompts and search results.
# MAGIC

# COMMAND ----------

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser

## Define the Databricks Chat model: llama-3
llm_llama = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens=1000)

## Define the prompt template for generating search queries
prompt_template_vs = PromptTemplate.from_template(
    """
    You are a documentation assistant. Based on the following context from a technical document, generate a concise summary or relevant content snippet for answering the user’s question.

    Write a response that is aligned with the tone and format of technical documentation and helps the user understand or resolve their query.

    Maximum 300 words.

    Use the following document snippet and context as example;

    <context>
    {context}
    </context>

    Question: {input}
    """
)

## Construct the RetrievalQA chain for Vector Store search
def get_retriever(persist_dir=None):
    vsc = VectorSearchClient(disable_notice=True)
    vs_index = vsc.get_index(vs_endpoint_name, vs_index_table_fullname)
    vectorstore = DatabricksVectorSearch(vs_index_table_fullname)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

## Construct the chain for question-answering
question_answer_chain = create_stuff_documents_chain(llm_llama, prompt_template_vs)
chain1 = create_retrieval_chain(get_retriever(), question_answer_chain)

## Invoke the chain with an example query   
response = chain1.invoke({"input": "How do I create a Delta table?"})
print(response['answer'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Build the Second Chain (Optimization)
# MAGIC
# MAGIC In this step, you will create a second chain to enhance the product details generated by the first chain. This optimization process aims to make the descriptions more compelling and SEO-friendly. In a real-world scenario, this model could be trained on your internal data or fine-tuned to align with your specific business objectives.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Define a second chain using `llama-3-70b-instruct`.  
# MAGIC
# MAGIC - Create a prompt to optimize the generated product description. For example:  
# MAGIC   *"You are a marketing expert. Revise the product title and description to be SEO-friendly and more appealing to Databricks users."*
# MAGIC
# MAGIC - Use `product_details` as the parameter to be passed into the prompt.  
# MAGIC
# MAGIC - Implement the chain and test it with a sample input.  
# MAGIC

# COMMAND ----------

## Define the Databricks Chat model using llama-3-3-70b-instruct
llm_llama3 = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens=1000)

## Define the prompt template for refining documentation output
doc_optimization_prompt = PromptTemplate.from_template(
    """
    You are a technical writer. Improve the following documentation snippet to make it clearer, concise, and aligned with the tone used in Databricks documentation.

    Documentation snippet: {doc_snippet}

    Return only the revised documentation content.
    """
)

## Define chain 2
chain2 = doc_optimization_prompt | llm_llama3 | StrOutputParser()

## Test the chain
chain2.invoke({"doc_snippet": "Query testing product with mobile app control"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Integrate Chains into a Multi-chain System
# MAGIC
# MAGIC In this task, you will link the individual chains created in Task 2 and Task 3 together to form a multi-chain system that can handle multi-stage reasoning.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Use Databricks **`Llama Chat model`** for processing text inputs, which is defined above in the first task.
# MAGIC
# MAGIC - Create a prompt template to generate an **`HTML page`** for displaying generated product details.
# MAGIC
# MAGIC - Construct the **`Multi-Chain System`**  by combining the outputs of the previous chains. **Important**: You will need to rename the output of the first chain and second chain while passing them to the next stage. This sequential chain should be as; **chain3 = chain1 > (`product_details`) > chain2 > `(optimized_product_details)` > prompt3**.  
# MAGIC
# MAGIC - Invoke the multi-chain system with the input data to generate the HTML page for the specified product.
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from IPython.display import display, HTML

## Define the prompt template for generating the HTML page
prompt_template_3 = PromptTemplate.from_template(
    """Create an HTML section for the following technical documentation snippet:
    
    Content: {optimized_doc}

    Return valid HTML (no head/body tags).
    """
)

## Construct multi-stage chain
chain3 = (
    chain1
    | RunnableMap({"doc_snippet": lambda x: x["answer"]})
    | chain2
    | RunnableMap({"optimized_doc": lambda x: x})
    | prompt_template_3
    | llm_llama
    | StrOutputParser()
)

## Sample query
query = {
    "input": "How do I create a Delta table in Databricks?"
}

output_html = chain3.invoke(query)

## Display the generated HTML output
display(HTML(output_html))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Save the Chain to Model Registry in UC
# MAGIC
# MAGIC In this task, you will save the multi-stage chain system within our Unity Catalog.
# MAGIC
# MAGIC **Instructions:**
# MAGIC
# MAGIC - Set the model registry to UC and use the model name defined.
# MAGIC
# MAGIC - Log and register the final multi-chain system.
# MAGIC
# MAGIC - To test the registered model, load the model back from model registry and query it using a sample query. 
# MAGIC
# MAGIC After registering the chain, you can view the chain and models in the **Catalog Explorer**.

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow

## Set model registry to UC
mlflow.set_registry_uri("databricks-uc")
model_name = f"{DA.catalog_name}.{DA.schema_name}.multi_stage_doc_chain"

## Log the model
with mlflow.start_run(run_name="multi_stage_doc_chain") as run:
    signature = infer_signature(query, output_html)
    model_info = mlflow.langchain.log_model(
        chain3,
        loader_fn=get_retriever, 
        name="chain",
        registered_model_name=model_name,
        input_example=query,
        signature=signature
    )

## Load and test the model
model_uri = f"models:/{model_name}/{model_info.registered_model_version}"
model = mlflow.langchain.load_model(model_uri)

output_html = model.invoke(query)
display(HTML(output_html))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you've learned how to build a multi-stage AI system using Databricks and LangChain. By integrating multiple chains, you can perform complex reasoning tasks such as searching for product details and optimizing the response based on your business needs. This approach enables the development of sophisticated AI systems capable of handling diverse tasks efficiently.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>