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
# MAGIC # Agent Design in Databricks
# MAGIC
# MAGIC In the previous demo, we build a multi-stage AI system by manually stitching them together. With Agents, we can build the same system in an autonomous way. An agent, typically, has a brain which make the decisions, a planning outline and tools to use. 
# MAGIC
# MAGIC In this demo, we will create two types of agents. The first agent will use **a search engine, Wikipedia, and Youtube** to recommend a movie, collect data about the movie and show the trailer video. 
# MAGIC
# MAGIC The second agent is a very specific type agent; it will allow us to "talk with data" using natural language queries. 
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Build semi-automated systems with LLM agents to perform internet searches and dataset analysis using LangChain.
# MAGIC
# MAGIC * Use appropriate tool for the agent task to be achieved.
# MAGIC
# MAGIC * Explore LangChain’s built-in agents for specific, advanced workflows.
# MAGIC
# MAGIC * Create a Pandas DataFrame Agent to interact with a Pandas DataFrame as needed.
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
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install -U -qqq 'mlflow-skinny[databricks]==3.4.0' databricks-langchain==0.8.0 langchain==0.3.7 langchain-community==0.3.7 langchain-experimental==0.3.3 youtube_search==2.1.2 Wikipedia==1.4.0 huggingface-hub==0.36.0 tabulate==0.9.0
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-03

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
# MAGIC ## Enable MLflow Auto-Log
# MAGIC
# MAGIC MLflow has support for auto-logging LangChain models. We will enable this below.
# MAGIC

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an Autonomous Agent (Brixo 🤖)
# MAGIC
# MAGIC In the previous demo, we created chains using various prompts and tools combinations to solve a problem defined by the prompt. In chains, we need to define the input parameters and prompts. 
# MAGIC
# MAGIC In this demo, we will create an agent that can **autonomously reason** about the steps to take and select **the tools** to use for each task.
# MAGIC
# MAGIC **🤖 Agent name: Brixo :)**
# MAGIC
# MAGIC **✅ Agent Abilities: This agent can help you by suggesting fun activities, pick videos and even write code.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Brain of the Agent
# MAGIC
# MAGIC LLM is the brain of the agent. We will use **Llama-3 model** as the brain of our agent.

# COMMAND ----------

from databricks_langchain import ChatDatabricks

# play with max_tokens to define the length of the response
llm_llama = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 2500)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Tools that the Agent Can Use
# MAGIC
# MAGIC Agent can use various tools for completing a task. Here we will define the tools that can be used by **Brixo 🤖**.

# COMMAND ----------

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools import YouTubeSearchTool

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

# Wiki tool for info retrieval
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# tool to search youtube videos
tool_youtube = YouTubeSearchTool(handle_tool_error=True)

# tool to write python code
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# toolset
tools = [tool_wiki, tool_youtube, repl_tool]
tools

# COMMAND ----------

type(tool_wiki), type(tool_youtube), type(repl_tool)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Planning Logic
# MAGIC
# MAGIC While working on tasks, our agent will need to done some reasoning and planning. We can define the format of this plan by passing a prompt.

# COMMAND ----------

from langchain.prompts import PromptTemplate

template = """Answer the following questions as best you can. You have access to the following tools:

'{{tools}}'

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of ['{{tool_names}}']
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}'
Thought:{agent_scratchpad}"""

prompt= PromptTemplate.from_template(template)
prompt

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Agent
# MAGIC
# MAGIC The final step is to put all these together and build an agent.

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent

agent = create_tool_calling_agent(llm_llama, tools, prompt)
brixo  = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
agent_response = brixo.invoke({"input": 
    """What would be a nice movie to watch in rainy weather. Follow these steps.

    First, decide which movie you would recommend.

    Second, show me the trailer video of the movie that you suggest. Search YouTube only once.

    Add a newline at the end of each step.

    Next, collect data about the movie using search tool and  draw a bar chart using Python libraries. If you can't find latest data use some dummy data as we to show your abilities to the learners. Remove any enclosing tags and keywords for the code as it will be passed to a Python REPL. Make sure it is valid python code and ready to be executed. 

    Finally, tell a funny joke about agents.
    """})
print(agent_response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an Autonomous Agent 2 (DataQio 🤖)
# MAGIC
# MAGIC In this section we will create a quite different agent; this agent will allow us to communicate with our **Pandas dataframe** using natural language.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Dataset
# MAGIC
# MAGIC First, let's download a dataset from Hugging Face🤗 and convert it to Pandas dataframe.

# COMMAND ----------

import datasets
from datasets import load_dataset

datasets.utils.logging.disable_progress_bar()
dataset = load_dataset(
    "maharshipandya/spotify-tracks-dataset",
    cache_dir=DA.paths.working_dir + "/hf_cache")

df = dataset["train"].to_pandas()
display(df.sort_values("popularity", ascending=False).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Brain and Tools
# MAGIC
# MAGIC Next we will define the model(brain) of our agent and define the toolset to use.

# COMMAND ----------

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from databricks_langchain import ChatDatabricks

llm_llama = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens = 5000)

prefix = """ Input should be sanitized by removing any leading or trailing backticks. if the input starts with ”python”, remove that word as well. Use the dataset provided. The output must start with a new line. Make sure the response is a valid Python code and does not exceed 5000 tokens."""

dataqio = create_pandas_dataframe_agent(
    llm_llama,
    df,
    verbose=True,
    max_iterations=3,
    prefix=prefix,
    allow_dangerous_code=True,
    agent_executor_kwargs={
        "handle_parsing_errors": True
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Talk with DataQio 🤖
# MAGIC
# MAGIC We are ready to talk with our agent to ask questions about the data.

# COMMAND ----------

query = "Top 5 fastest songs?" 
response = dataqio.invoke(query)

# COMMAND ----------

query = "Python and top movies ?" 
response = dataqio.invoke(query)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the Agent to Model Registry in UC
# MAGIC
# MAGIC Now that our agent is ready and evaluated, we can register it within our Unity Catalog schema. 
# MAGIC
# MAGIC After registering the agent, you can view the agent and models in the **Catalog Explorer**.

# COMMAND ----------

import pandas as pd

local_artifact_path = "spotify.parquet"
df.to_parquet(local_artifact_path, index=False)   # reuse the existing `df` from above
print(f"Wrote dataset artifact → {local_artifact_path}, rows={len(df)}")

# Write the model to a standalone Python file
model_file = "dataqio_model.py"

model_src = """
import mlflow
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel

import pandas as pd
import numpy as np

from databricks_langchain import ChatDatabricks
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

class DataQioModel(PythonModel):
    def load_context(self, context):
        # Load the dataset artifact and initialize LLM once per process
        self.df = pd.read_parquet(context.artifacts["spotify_parquet"])
        self.llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", max_tokens=3000)
        self.prefix = (
            " Input should be sanitized by removing any leading or trailing backticks. "
            'If the input starts with "python", remove that word as well. Use the dataset provided. '
            "The output must start with a new line. Make sure the response is valid Python code and <= 5000 tokens."
        )

    def _make_agent(self):
        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=False,
            max_iterations=8,
            prefix=self.prefix,
            allow_dangerous_code=True,
            agent_executor_kwargs={\"handle_parsing_errors\": True},
        )

    def _run_query(self, q: str) -> str:
        agent = self._make_agent()
        result = agent.invoke(q)
        # LangChain agents commonly return {'input': ..., 'output': ...}
        if isinstance(result, dict) and "output" in result:
            return str(result["output"])
        return str(result)

    def _coerce_to_queries(self, model_input):
        # Accept common MLflow input shapes
        if isinstance(model_input, str):
            return [model_input]
        if isinstance(model_input, pd.Series):
            return [str(x) for x in model_input.tolist()]
        if isinstance(model_input, (list, tuple, np.ndarray)):
            return [str(x) for x in list(model_input)]
        if isinstance(model_input, pd.DataFrame):
            if "inputs" in model_input.columns:
                return [str(x) for x in model_input["inputs"].tolist()]
            if model_input.shape[1] == 1:
                return [str(x) for x in model_input.iloc[:, 0].tolist()]
            # last resort: join columns row-wise
            return [\" \".join(map(str, row)) for _, row in model_input.astype(str).iterrows()]
        return [str(model_input)]

    def predict(self, context, model_input, params=None):
        queries = self._coerce_to_queries(model_input)
        return [self._run_query(q) for q in queries]

set_model(DataQioModel())
"""

with open(model_file, "w") as f:
    f.write(model_src)

print(f"Wrote model file → {model_file}")

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

model_name = f"{DA.catalog_name}.{DA.schema_name}.multi_stage_demo"

# Use a DF input example to match serving/batch usage
input_example_df = pd.DataFrame({"inputs": ["What is the name of song with highest tempo?"]})
output_example = ["<agent_response_here>"]  # shape only; not executed
signature = infer_signature(input_example_df, output_example)


with mlflow.start_run(run_name="multi_stage_demo"):
    model_info = mlflow.pyfunc.log_model(
        python_model=model_file,                               # path to the .py file
        artifacts={"spotify_parquet": local_artifact_path},    # artifact key used in load_context
        registered_model_name=model_name,
        input_example=input_example_df,
        signature=signature
    )

print("Registered model:")
print("  name:", model_name)
print("  uri :", model_info.model_uri)

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(model_info.model_uri)

# Single query (as DF)
single_out = loaded.predict(pd.DataFrame({"inputs": ["Top 5 fastest songs?"]}))
print(single_out[0])

# Batch queries
batch_prompts = pd.DataFrame({"inputs": [
    "Top 5 fastest songs?",
    "What is the name of song with highest tempo?"
]})
batch_out = loaded.predict(batch_prompts)
for i, ans in enumerate(batch_out, 1):
    print(f"[{i}] {ans[:200]}{'...' if len(ans)>200 else ''}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we explored agent design in Databricks, moving beyond manual system stitching to autonomous agent-based systems. Agents, equipped with decision-making branches, planning outlines, and tools, streamline the process. We created two types of agents: one utilizing a search engine, Wikipedia, and YouTube to recommend movies and another enabling natural language data queries. By leveraging LangChain's capabilities, participants learned to build semi-automated systems, choose appropriate tools, and utilize built-in agents for advanced workflows, including interacting with Pandas DataFrames.

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>