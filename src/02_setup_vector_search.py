# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Vector Search
# MAGIC Verifies VS endpoint is ONLINE, creates Delta Sync index if needed, and tests a sample query.

# COMMAND ----------

import json
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

import yaml
_cfg_path = "/Workspace" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/src/", 1)[0] + "/config.yaml"
try:
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f)
except Exception:
    _cfg = {}

CATALOG = _cfg.get("catalog", "YOUR_CATALOG")
SCHEMA = _cfg.get("schema", "vi_demo")
VS_ENDPOINT = _cfg.get("vs_endpoint", "vi-vs-endpoint")
VS_INDEX = f"{CATALOG}.{SCHEMA}.research_documents_index"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.research_documents"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify VS endpoint is ONLINE

# COMMAND ----------

endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
print(f"Endpoint: {endpoint.name}")
print(f"Status: {endpoint.endpoint_status.state}")
assert "ONLINE" in str(endpoint.endpoint_status.state), f"VS endpoint not ONLINE: {endpoint.endpoint_status.state}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create or verify VS index

# COMMAND ----------

try:
    index = w.vector_search_indexes.get_index(VS_INDEX)
    print(f"Index already exists: {index.name}")
    print(f"Status: {index.status.ready}")
except Exception as e:
    print(f"Index not found, creating via REST API: {e}")
    host = w.config.host.rstrip("/")
    header_factory = w.config.authenticate
    headers = header_factory()
    headers["Content-Type"] = "application/json"

    payload = {
        "name": VS_INDEX,
        "endpoint_name": VS_ENDPOINT,
        "primary_key": "chunk_id",
        "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": SOURCE_TABLE,
            "pipeline_type": "TRIGGERED",
            "embedding_source_columns": [
                {
                    "name": "content",
                    "embedding_model_endpoint_name": "databricks-gte-large-en",
                }
            ],
        },
    }

    resp = requests.post(
        f"{host}/api/2.0/vector-search/indexes",
        headers=headers,
        json=payload,
    )
    print(f"Response: {resp.status_code} {resp.text}")
    resp.raise_for_status()
    print(f"Created Delta Sync index: {VS_INDEX}")
    print("Index is syncing. It may take a few minutes to become ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test a sample query (may fail if index is still syncing)

# COMMAND ----------

import time

# Wait a bit for index to become queryable if just created
for attempt in range(3):
    try:
        index = w.vector_search_indexes.get_index(VS_INDEX)
        if index.status.ready:
            results = w.vector_search_indexes.query_index(
                index_name=VS_INDEX,
                columns=["chunk_id", "company", "content"],
                query_text="Indonesian concession risk",
                num_results=3,
            )
            for r in results.result.data_array:
                print(f"  chunk={r[0]}, company={r[1]}, content={r[2][:100]}...")
            break
        else:
            print(f"Index not ready yet (attempt {attempt+1}/3). Waiting 30s...")
            time.sleep(30)
    except Exception as e:
        print(f"Query failed (attempt {attempt+1}/3): {e}")
        if attempt < 2:
            time.sleep(30)
        else:
            print("Index may still be syncing. Query will work once sync completes.")
