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
# MAGIC # Assembling and Registering a RAG Application
# MAGIC
# MAGIC In the previous demo, we created a Vector Search Index. To build a complete RAG application, it is time to connect all the components that you have learned so far and evaluate the performance of the RAG.
# MAGIC
# MAGIC After evaluating the performance of the RAG pipeline, we will create and deploy a new Model Serving Endpoint to perform RAG.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC - Describe embeddings, vector databases, and search/retrieval as key components of implementing performant RAG applications.
# MAGIC - Assemble a RAG pipeline by combining various components.
# MAGIC - Build a RAG evaluation pipeline with MLflow evaluation functions.
# MAGIC - Register a RAG pipeline to the Model Registry.
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
# MAGIC
# MAGIC
# MAGIC **🚨 Important: This demonstration relies on the resources established in the previous one. Please ensure you have completed the prior demonstration before starting this one.**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U -qqq databricks-sdk databricks-vectorsearch 'mlflow-skinny[databricks]==3.4.0' langchain==0.3.26 databricks-langchain==0.8.0 PyPDF2==3.0.0 flashrank
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-04

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
# MAGIC ## Demo Overview
# MAGIC
# MAGIC As seen in the diagram below, in this demo we will focus on the inference section (highlighted in green). The main focus of the previous demos was  Step 1 - Data preparation and vector storage. Now, it is time to put all components together to create a RAG application. 
# MAGIC
# MAGIC The flow will be the following:
# MAGIC
# MAGIC - A user asks a question
# MAGIC - The question is sent to our serverless Chatbot RAG endpoint
# MAGIC - The endpoint compute the embeddings and searches for docs similar to the question, leveraging the Vector Search Index
# MAGIC - The endpoint creates a prompt enriched with the doc
# MAGIC - The prompt is sent to the Foundation Model Serving Endpoint
# MAGIC - We display the output to our users!
# MAGIC
# MAGIC
# MAGIC <!-- <img src="https://files.training.databricks.com/images/genai/genai-as-01-llm-rag-self-managed-flow-2.png" width="100%"> -->
# MAGIC
# MAGIC <!--  -->
# MAGIC
# MAGIC ![genai-as-01-llm-rag-self-managed-flow-2](../Includes/images/genai-as-01-llm-rag-self-managed-flow-2.png)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup the RAG Components
# MAGIC
# MAGIC In this section, we will first define the components that we created before. Next, we will set up the retriever component for the application. Then, we will combine all the components together. In the final step, we will register the developed application as a model in the Model Registry with Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup the Retriever
# MAGIC
# MAGIC We will setup the Vector Search endpoint that we created in the previous demos as retriever. The retriever will return 2 relevant documents based on the query.
# MAGIC

# COMMAND ----------

# components we created before
# assign vs search endpoint by username
vs_endpoint_prefix = "vs_endpoint_"

vs_endpoint_name = vs_endpoint_prefix + str(get_fixed_integer(DA.unique_name("_")))
print(f"Assigned Vector Search endpoint name: {vs_endpoint_name}.")

vs_index_fullname = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_self_managed_vs_index"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import DatabricksEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

def get_retriever(cache_dir="/tmp"):

    def retrieve(query, k: int=10):
        if isinstance(query, dict):
            query = next(iter(query.values()))

        # get the vector search index
        vsc = VectorSearchClient(disable_notice=True)
        vs_index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)
        
        # get the query vector
        embeddings = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
        query_vector = embeddings.embed_query(query)
        
        # get similar k documents
        return query, vs_index.similarity_search(
            query_vector=query_vector,
            columns=["pdf_name", "content"],
            num_results=k)

    def rerank(query, retrieved, cache_dir, k: int=2):
        # format result to align with reranker lib format 
        passages = []
        for doc in retrieved.get("result", {}).get("data_array", []):
            new_doc = {"file": doc[0], "text": doc[1]}
            passages.append(new_doc)       
        # Load the flashrank ranker
        ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)

        # rerank the retrieved documents
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerankrequest)[:k]

        # format the results of rerank to be ready for prompt
        return [Document(page_content=r.get("text"), metadata={"source": r.get("file")}) for r in results]

    # the retriever is a runnable sequence of retrieving and reranking.
    return RunnableLambda(retrieve) | RunnableLambda(lambda x: rerank(x[0], x[1], cache_dir))

# test our retriever
question = {"input": "What is Inductive Learning?"}
vectorstore = get_retriever(cache_dir = f"{DA.paths.working_dir}/opt")
similar_documents = vectorstore.invoke(question)
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup the Foundation Model
# MAGIC
# MAGIC Our chatbot will be using `llama3.3` foundation model to provide answer. 
# MAGIC
# MAGIC While the model is available using the built-in [Foundation endpoint](/ml/endpoints), we can use Databricks Langchain Chat Model wrapper to easily build our chain.  
# MAGIC
# MAGIC Note: multiple type of endpoint or langchain models can be used.
# MAGIC
# MAGIC - Databricks Foundation models (what we'll use)
# MAGIC - Your fined-tune model
# MAGIC - An external model provider (such as Azure OpenAI)

# COMMAND ----------

from databricks_langchain import ChatDatabricks

# test Databricks Foundation LLM model
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 300)
print(f"Test chat model: {chat_model.invoke('What is Generative AI?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assembling the Complete RAG Solution
# MAGIC
# MAGIC Let's now merge the retriever and the model in a single Langchain chain.
# MAGIC
# MAGIC We will use a custom langchain template for our assistant to give proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC
# MAGIC <!-- <img src="https://files.training.databricks.com/images/genai/genai-as-01-llm-rag-self-managed-model-2.png" width="100%" /> -->
# MAGIC
# MAGIC ![genai-as-01-llm-rag-self-managed-model-2](../Includes/images/genai-as-01-llm-rag-self-managed-model-2.png)
# MAGIC
# MAGIC <!--  -->

# COMMAND ----------

# MAGIC %md
# MAGIC Some important notes about the LangChain formatting:
# MAGIC
# MAGIC * Context documents retrieved from the vector store are added by separated newline.

# COMMAND ----------

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt template
TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. 
Use the following pieces of context to answer the question at the end:

<context>
{context}
</context>

Question: {input}

Answer:
"""

prompt = ChatPromptTemplate.from_template(TEMPLATE)      

# Helper functions
def format_docs(docs):
    # what the model sees in {context}
    return "\n\n".join(d.page_content for d in docs)

def unwrap(payload):
    # return both answer and normalized context (dicts) like you wanted
    docs = payload["docs"]
    return {
        "answer": payload["answer"],
        "context": [{"metadata": getattr(d, "metadata", {}), "page_content": getattr(d, "page_content", "")}
                    for d in docs],
    }

# ---- build the chain ----
# Step 1: retrieve docs
retrieve = RunnableParallel(input=RunnablePassthrough(), docs=get_retriever())

# Step 2: pass formatted context + input to the model
rag = retrieve | {
    "input": itemgetter("input"),
    "context": RunnableLambda(lambda x: format_docs(x["docs"]))
} | prompt | chat_model | StrOutputParser()

# Keep docs for postprocessing
chain = retrieve | {
    "answer": ({"input": itemgetter("input"), "context": RunnableLambda(lambda x: format_docs(x["docs"]))}
               | prompt | chat_model | StrOutputParser()),
    "docs": itemgetter("docs"),
} | RunnableLambda(unwrap)

# COMMAND ----------

question = {"input": "What are the generative AI's economical implications?"}
response = chain.invoke(question)
print(response['answer'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the Model to Model Registry in UC
# MAGIC
# MAGIC Now that our model is ready and evaluated, we can register it within our Unity Catalog schema. 
# MAGIC
# MAGIC After registering the model, you can view the model and models in the **Catalog Explorer**.

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

# set model registry to UC
mlflow.set_registry_uri("databricks-uc")
model_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_demo4"

with mlflow.start_run(run_name="rag_app_demo4") as run:
    signature = infer_signature(question, response)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever, 
        name="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        signature=signature,
        input_example=question
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Resources
# MAGIC
# MAGIC This is the final demo. You can delete all resources created in this course.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we illustrated the process of constructing a comprehensive RAG application utilizing a variety of Databricks products. Initially, we established the RAG components that were previously created in the earlier demos, namely the Vector Search endpoint and Vector Search index. Subsequently, we constructed the retriever component and set up the foundational model for use. Following this, we put together the entire RAG application and evaluated the performance of the pipeline using MLflow's LLM evaluation functions. As a final step, we registered the newly created RAG application as a model within the Model Registry with Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>