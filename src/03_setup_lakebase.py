# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Lakebase for Session History
# MAGIC Provisions a Lakebase PostgreSQL instance and creates the sessions/messages tables.

# COMMAND ----------

import yaml
_cfg_path = "/Workspace" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/src/", 1)[0] + "/config.yaml"
try:
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f)
except Exception:
    _cfg = {}

CATALOG = _cfg.get("catalog", "YOUR_CATALOG")
SCHEMA = _cfg.get("schema", "vi_demo")
LAKEBASE_INSTANCE = "vi_agent_memory"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Lakebase instance (if not exists)

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Check if Lakebase API is available
try:
    # List existing instances
    existing = list(w.lakebase_instances.list())
    instance_names = [i.name for i in existing]
    print(f"Existing Lakebase instances: {instance_names}")

    if LAKEBASE_INSTANCE not in instance_names:
        print(f"Creating Lakebase instance: {LAKEBASE_INSTANCE}")
        w.lakebase_instances.create(
            name=LAKEBASE_INSTANCE,
            catalog_name=CATALOG,
            schema_name=SCHEMA,
        )
        print("Lakebase instance created. Waiting for it to be ready...")
    else:
        print(f"Lakebase instance '{LAKEBASE_INSTANCE}' already exists.")
except AttributeError:
    print("WARNING: Lakebase API not available in this SDK version.")
    print("Create the Lakebase instance manually via the UI.")
except Exception as e:
    print(f"WARNING: Could not provision Lakebase: {e}")
    print("Create the Lakebase instance manually via the UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create sessions table
# MAGIC
# MAGIC Once the Lakebase instance is provisioned, connect and create the schema.
# MAGIC
# MAGIC **Note:** You may need to run this step manually after the instance is ready.
# MAGIC The connection details (host, port, credentials) will be available in the
# MAGIC Lakebase instance details in the Databricks UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expected table schema (run against Lakebase PostgreSQL):
# MAGIC
# MAGIC ```sql
# MAGIC CREATE TABLE IF NOT EXISTS sessions (
# MAGIC     session_id TEXT PRIMARY KEY,
# MAGIC     created_at TIMESTAMP DEFAULT NOW()
# MAGIC );
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS messages (
# MAGIC     id SERIAL PRIMARY KEY,
# MAGIC     session_id TEXT REFERENCES sessions(session_id),
# MAGIC     role TEXT NOT NULL,
# MAGIC     content TEXT NOT NULL,
# MAGIC     tool_calls JSONB,
# MAGIC     trace_steps JSONB,
# MAGIC     created_at TIMESTAMP DEFAULT NOW()
# MAGIC );
# MAGIC
# MAGIC CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
# MAGIC ```
