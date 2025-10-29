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
# MAGIC # LAB - Assembling a RAG Application
# MAGIC
# MAGIC In this lab, we will assemble a Retrieval-augmented Generation (RAG) application using the components we previously created. The primary goal is to create a seamless pipeline where users can ask questions, and our system retrieves relevant documents from a Vector Search index to generate informative responses.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Setup the Retriever Component
# MAGIC
# MAGIC * **Task 2 :** Setup the Foundation Model
# MAGIC
# MAGIC * **Task 3 :** Assemble the Complete RAG Solution
# MAGIC
# MAGIC * **Task 4 :** Save the Model to Model Registry in Unity Catalog
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
# MAGIC
# MAGIC **🚨 Important:** This lab relies on the resources established in the previous one. Please ensure you have completed the prior lab before starting this one.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install -U -qqq databricks-sdk databricks-vectorsearch 'mlflow-skinny[databricks]==3.4.0' langchain==0.3.26 databricks-langchain==0.8.0 PyPDF2==3.0.0 flashrank
# MAGIC %restart_python

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
# MAGIC ## Task 1: Setup the Retriever Component
# MAGIC **Steps:**
# MAGIC 1. Define the embedding model.
# MAGIC 1. Get the vector search index that was created in the previous lab.
# MAGIC 1. Generate a **retriever** from the vector store. The retriever should return **three results.**
# MAGIC 1. Write a test prompt and show the returned search results.
# MAGIC

# COMMAND ----------

## Components we created before
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_name = vs_endpoint_prefix+str(get_fixed_integer(DA.unique_name("_")))
print(f"Assigned Vector Search endpoint name: {vs_endpoint_name}.")

vs_index_fullname = f"{DA.catalog_name}.{DA.schema_name}.lab_pdf_text_managed_vs_index"

# COMMAND ----------


from databricks.vector_search.client import VectorSearchClient
from databricks_langchain import DatabricksEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.docstore.document import Document
from flashrank import Ranker, RerankRequest


def get_retriever(cache_dir=f"{DA.paths.working_dir}/opt"):

    def retrieve(query, k: int=10):
        if isinstance(query, dict):
            query = next(iter(query.values()))

        ## get the vector search index
        vsc = VectorSearchClient(disable_notice=True)
        vs_index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)

        # get the query vector
        embeddings = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
        query_vector = embeddings.embed_query(query)
        
        ## get similar k documents
        return query, vs_index.similarity_search_with_score(
            query_vector=query_vector,
            columns=["pdf_name", "content"],
            num_results=k)


    def rerank(query, retrieved, cache_dir, k: int=2):
        ## format result to align with reranker lib format 
        passages = []
        for doc in retrieved.get("result", {}).get("data_array", []):
            new_doc = {"file": doc[0], "text": doc[1]}
            passages.append(new_doc)       
        ## Load the flashrank ranker
        ranker = Ranker(model_name="rank-T5-flan", cache_dir=cache_dir)

        ## rerank the retrieved documents
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerankrequest)[:k]

        ## format the results of rerank to be ready for prompt
        return [Document(page_content=r.get("text"), metadata={"source": r.get("file")}) for r in results]

    ## the retriever is a runnable sequence of retrieving and reranking.
    return RunnableLambda(retrieve) | RunnableLambda(lambda x: rerank(x[0], x[1], cache_dir))


## test your retriever
question = {"input": "Will Generative AI replace humans especially in workfoce and how can human survive ?"}
vectorstore = get_retriever()
similar_documents = vectorstore.invoke(question)
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Setup the Foundation Model
# MAGIC **Steps:**
# MAGIC 1. Define the foundation model for generating responses. Use `llama-3.3` as foundation model. 
# MAGIC 2. Test the foundation model to ensure it provides accurate responses.

# COMMAND ----------

## import necessary libraries
from databricks_langchain import ChatDatabricks

## define foundation model for generating responses
chat_model = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 500)

## test foundation model
print(f"Test chat model: {chat_model.invoke('Will Generative AI replace humans especially in workfoce?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3: Assemble the Complete RAG Solution
# MAGIC **Steps:**
# MAGIC 1. Merge the retriever and foundation model into a single Langchain chain.
# MAGIC 2. Configure the Langchain chain with proper templates and context for generating responses.
# MAGIC 3. Test the complete RAG solution with sample queries.

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

# Test the complete RAG solution with sample query
question = {"input": "Will Generative AI replace humans especially in workfoce?"}
response = chain.invoke(question)
print(response['answer'])

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

# Test the complete RAG solution with sample query
question = {"input": "WWill Generative AI replace humans especially in workfoce?"}
response = chain.invoke(question)
print(response['answer'])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Save the Model to Model Registry in Unity Catalog
# MAGIC **Steps:**
# MAGIC 1. Register the assembled RAG model in the Model Registry with Unity Catalog.
# MAGIC 2. Ensure that all necessary dependencies and requirements are included.
# MAGIC 3. Provide an input example and infer the signature for the model.

# COMMAND ----------

## import necessary libraries
from mlflow.models import infer_signature
import mlflow
import langchain

## set Model Registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_demo4"

## register the assembled RAG model in Model Registry with Unity Catalog
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
# MAGIC This was the final lab. You can delete all resources created in this course.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned how to assemble a Retrieval-augmented Generation (RAG) application using Databricks components. By integrating Vector Search for document retrieval and a foundational model for response generation, you created a powerful tool for answering user queries. This lab provided hands-on experience in building end-to-end AI applications and demonstrated the capabilities of Databricks for natural language processing tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>