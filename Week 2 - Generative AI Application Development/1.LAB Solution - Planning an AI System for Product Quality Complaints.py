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
# MAGIC # LAB Solution - Planning a Compound AI System for Product Quality Complaints
# MAGIC
# MAGIC In this lab, you will deconstruct a use case and define possible components of the AI system. In this scenario, let's consider **a compound AI system designed to handle a customer complaint about a product's quality**. The customer contacts the service center via a chat interface, expressing dissatisfaction with a recently purchased item. The AI system will utilize various components to address and resolve the complaint effectively.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Define system components
# MAGIC
# MAGIC * **Task 2 :** Draw architectural diagram
# MAGIC
# MAGIC * **Task 3 :** Define possible input and output parameters for each component
# MAGIC
# MAGIC * **Task 4:** Define libraries or frameworks for each component
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## AI System Details
# MAGIC
# MAGIC **Product Quality Complaints:** Let's consider **a compound AI system designed to handle a customer complaint about a product's quality**. The customer contacts the service center via a chat interface, expressing dissatisfaction with a recently purchased item. 
# MAGIC
# MAGIC **Example Agent-User Interaction:** 
# MAGIC
# MAGIC **👱‍♀️ Customer:** "I'm unhappy with the quality of the product I received. It seems defective."
# MAGIC
# MAGIC **🤖 AI Agent:** "I'm sorry to hear that you're not satisfied with your purchase. Let me look into this for you."
# MAGIC
# MAGIC **🤖 AI Agent:** (after processing): "We've reviewed similar feedback and checked the shipping details. It appears there have been a few similar complaints. We can offer a replacement or a full refund. Which would you prefer?"
# MAGIC
# MAGIC **👱‍♀️ Customer:** "I would like a replacement, please."
# MAGIC
# MAGIC **🤖 AI Agent:**  "I've arranged for a replacement to be sent to you immediately. We apologize for the inconvenience and thank you for your understanding."
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Define Components
# MAGIC
# MAGIC Based on the scenario, identify all necessary components that will interact within the system. Also, provide a brief description of the role and functionality of each component. 
# MAGIC
# MAGIC An example component and description could be;
# MAGIC
# MAGIC * **Data Retrieval Component:** Retrieves customer and product data from the company’s database.
# MAGIC
# MAGIC * **Search Past Customer Reviews Component:** Analyzes customer reviews to find similar complaints using natural language processing.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Solution 1: Components
# MAGIC
# MAGIC | Component Name                  | Description                                                                 |
# MAGIC |--------------------------------|-----------------------------------------------------------------------------|
# MAGIC | User Interface (Chat UI)       | Frontend interface through which customers submit complaints (e.g., web or mobile chat). |
# MAGIC | NLP/NLU (Natural Language Understanding) | Interprets user input to identify intent (e.g., complaint, refund request) and extract entities (e.g., product name, issue). |
# MAGIC | Customer Profile Retrieval     | Fetches customer’s purchase history, product info, and past interactions from databases. |
# MAGIC | Product Quality Insights       | Analyzes historical data, reviews, and known product issues using NLP.      |
# MAGIC | Issue Classifier               | Classifies the nature of the complaint (e.g., defective, damaged, counterfeit). |
# MAGIC | Decision Engine                | Determines next steps—refund, replacement, or escalation—based on company policies and prior cases. |
# MAGIC | LLM Response Generator         | Generates empathetic, context-aware responses to the user using a language model. |
# MAGIC | Action Executor                | Triggers workflows like refund initiation, order replacement, or human escalation. |
# MAGIC | Audit & Logging Module         | Logs all actions taken, for traceability and compliance.                    |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Draw Architectural Diagram
# MAGIC
# MAGIC Begin by selecting a **diagramming tool** such as [draw.io](https://draw.io), or any other tool that you feel comfortable with. Next, **arrange the components identified in Task 1** on the diagram canvas in a logical sequence based on their data interactions. Connect these components with directional arrows to depict the flow of data and interactions clearly. Each component and connection should be clearly labeled, possibly with brief descriptions if necessary to enhance clarity. 
# MAGIC
# MAGIC Finally, review the diagram to ensure it is easy to understand and accurately represents all components and their interactions within the system.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Solution 2: Architectural Diagram
# MAGIC
# MAGIC <!-- ![new task 2_ Architectural Diagram.png](./new task 2_ Architectural Diagram.png "new task 2_ Architectural Diagram.png") -->
# MAGIC
# MAGIC ![new_schedule](../Includes/images/task 2-Architectural-Diagram.png)
# MAGIC ---
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Define Possible Input and Output Parameters for Each Component
# MAGIC
# MAGIC For each component, specify what data it receives (input) and what it sends out (output).
# MAGIC
# MAGIC Example for the Data Retrieval Component:
# MAGIC * Input: Customer ID, Product ID
# MAGIC * Output: Customer purchase history, Product details, Previous complaints

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Solution 3: Possible Input and Output Parameters for Each Component
# MAGIC
# MAGIC | Component                    | Input Parameters                                      | Output Parameters                                     |
# MAGIC |-----------------------------|-------------------------------------------------------|--------------------------------------------------------|
# MAGIC | Customer Chat UI            | Customer complaint text (user message)               | Raw input text                                         |
# MAGIC | NLU (Natural Language Understanding) | Raw text message                                      | Extracted intent, product name, issue type             |
# MAGIC | Issue Classifier            | Extracted intent, issue context                      | Issue category (e.g., defective, damaged, wrong item)  |
# MAGIC | Customer Profile Retrieval  | Customer ID, Product ID                              | Query to fetch customer history and past orders        |
# MAGIC | Customer Database           | Query from profile retrieval component               | Customer profile, purchase history, loyalty info       |
# MAGIC | Product Quality Insights    | Complaint text, product ID, issue type               | Similar complaints, defect trends, known issues        |
# MAGIC | Decision Engine             | Issue category, product metadata, customer history, trends | Recommended action (refund, replacement, escalate)    |
# MAGIC | LLM Response Generator      | Intent, context, and decision outcome                | Empathetic user-facing response                        |
# MAGIC | Action Executor             | Action type, customer ID, product ID                 | Triggered backend process (e.g., refund issued)        |
# MAGIC | Order System / CRM          | Execution trigger (from Action Executor)             | Updated order status, CRM logs                         |
# MAGIC | Audit & Logging Module      | System events, decision logs, action confirmations   | Structured logs for monitoring and compliance          |
# MAGIC | (User Response)             | Final generated message                              | N/A (ends user-facing flow)                            |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Define Libraries or Frameworks for Each Component
# MAGIC
# MAGIC For this task, you will need to select appropriate libraries or frameworks that will be utilized to build each component of the system. For retrieval and generation tasks, identify the type of the language model that need to be used.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Solution 4: Libraries or Frameworks for Each Component
# MAGIC
# MAGIC | Component                   | Suggested Tools / Libraries                          |
# MAGIC |----------------------------|------------------------------------------------------|
# MAGIC | Customer Chat UI           | React, Streamlit, Twilio                             |
# MAGIC | NLU                        | Hugging Face Transformers, spaCy, OpenAI             |
# MAGIC | Issue Classifier           | Scikit-learn, MLflow, FastAPI                        |
# MAGIC | Customer Profile Retrieval | SQLAlchemy, Databricks SQL                           |
# MAGIC | Customer Database          | Delta Lake, PostgreSQL                               |
# MAGIC | Product Quality Insights   | LangChain, PySpark, Sentence Transformers            |
# MAGIC | Decision Engine            | LangChain Agents, rule-based engine, MLflow          |
# MAGIC | LLM Response Generator     | OpenAI GPT-4, Claude, Mosaic AI                      |
# MAGIC | Action Executor            | Airflow, REST APIs, Azure Functions                  |
# MAGIC | Order System / CRM         | Salesforce APIs, Webhooks                            |
# MAGIC | Audit & Logging Module     | MLflow Logging, Datadog, Delta Logs                  |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="_blank">Apache Software Foundation</a>.<br/><br/><a href="https://databricks.com/privacy-policy" target="_blank">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use" target="_blank">Terms of Use</a> | <a href="https://help.databricks.com/" target="_blank">Support</a>