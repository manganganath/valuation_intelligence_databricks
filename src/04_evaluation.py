# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Evaluate — LLM-as-a-Judge
# MAGIC Evaluates the Valuation Intelligence Agent using Q&A pairs.
# MAGIC Uses `mlflow.genai.evaluate()` with the same scorers used for production monitoring.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.1" "databricks-agents" requests
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import time
import requests
import yaml
import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# Load central config
_cfg_path = "/Workspace" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/src/", 1)[0] + "/config.yaml"
try:
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f)
except Exception:
    print(f"config.yaml not found at {_cfg_path}, using defaults")
    _cfg = {}

CATALOG = _cfg.get("catalog", "YOUR_CATALOG")
SCHEMA = _cfg.get("schema", "vi_demo")
MODEL = _cfg.get("model", "databricks-claude-sonnet-4-6")
EXPERIMENT_PATH = _cfg.get("mlflow_experiment", "/Shared/vi_demo/vi_demo")
GENIE_SPACE_ID = _cfg.get("genie_space_id", "YOUR_GENIE_SPACE_ID")
KNOWLEDGE_ASSISTANT_ID = _cfg.get("knowledge_assistant_id", "YOUR_KA_ENDPOINT")
SQL_WAREHOUSE_ID = _cfg.get("sql_warehouse_id", "YOUR_SQL_WAREHOUSE_ID")

mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load evaluation dataset

# COMMAND ----------

eval_df = spark.table(f"{CATALOG}.{SCHEMA}.eval_dataset").toPandas()
print(f"Loaded {len(eval_df)} evaluation pairs")
display(eval_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define agent as predict_fn

# COMMAND ----------

SYSTEM_PROMPT = (
    "You are a valuation intelligence agent for institutional fund managers. "
    "You have three tools: query_financials, search_research, get_risk_assessment. "
    "Only call the tools you need. Be quantitative and precise. Use markdown."
)

TOOLS = [
    {"type": "function", "function": {"name": "query_financials",
        "description": "Queries company financial data using natural language.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "search_research",
        "description": "Searches research documents and analyst memos via vector similarity.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_risk_assessment",
        "description": "Assesses valuation risk by comparing PE ratio to sector median.",
        "parameters": {"type": "object", "properties": {"p_ticker": {"type": "string"}}, "required": ["p_ticker"]}}},
]

w = WorkspaceClient()
host = w.config.host.rstrip("/")
_auth = w.config.authenticate


def _headers():
    h = _auth()
    h["Content-Type"] = "application/json"
    return h


def _genie(question):
    h = _headers()
    r = requests.post(f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation",
                      headers=h, json={"content": question}, timeout=60).json()
    cid, mid = r["conversation_id"], r["message_id"]
    for _ in range(30):
        time.sleep(1)
        msg = requests.get(f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{cid}/messages/{mid}",
                           headers=h, timeout=30).json()
        if msg.get("status") in ("COMPLETED", "FAILED"):
            break
    for att in msg.get("attachments", []):
        if "query" in att and att["query"].get("statement_id"):
            d = requests.get(f"{host}/api/2.0/sql/statements/{att['query']['statement_id']}",
                             headers=h, timeout=30).json()
            cols = [c["name"] for c in d.get("manifest", {}).get("schema", {}).get("columns", [])]
            rows = d.get("result", {}).get("data_array", [])
            return json.dumps([dict(zip(cols, row)) for row in rows], default=str)
    return json.dumps({"error": "No results"})


def _ka(query):
    r = requests.post(f"{host}/serving-endpoints/{KNOWLEDGE_ASSISTANT_ID}/invocations",
                      headers=_headers(), json={"input": [{"role": "user", "content": query}]}, timeout=60).json()
    texts = []
    for item in r.get("output", []):
        if item.get("type") == "message":
            for b in item.get("content", []):
                if b.get("type") == "output_text" and b.get("text", "").strip():
                    texts.append(b["text"].strip())
    return "\n".join(texts) or "No results."


def _risk(ticker):
    d = requests.post(f"{host}/api/2.0/sql/statements", headers=_headers(),
                      json={"warehouse_id": SQL_WAREHOUSE_ID,
                            "statement": f"SELECT * FROM {CATALOG}.{SCHEMA}.get_risk_assessment('{ticker}')",
                            "wait_timeout": "30s"}, timeout=60).json()
    if d.get("status", {}).get("state") != "SUCCEEDED":
        return json.dumps({"error": "SQL failed"})
    cols = [c["name"] for c in d.get("manifest", {}).get("schema", {}).get("columns", [])]
    return json.dumps([dict(zip(cols, row)) for row in d.get("result", {}).get("data_array", [])], default=str)


def _tool(name, args):
    if name == "query_financials": return _genie(args.get("question", ""))
    if name == "search_research": return _ka(args.get("query", ""))
    if name == "get_risk_assessment": return _risk(args.get("p_ticker", ""))
    return json.dumps({"error": f"Unknown: {name}"})


@mlflow.trace
def predict_fn(question: str) -> dict:
    """Agent predict function for mlflow.genai.evaluate()."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    url = f"{host}/serving-endpoints/{MODEL}/invocations"
    for _ in range(5):
        data = requests.post(url, headers=_headers(),
                             json={"messages": messages, "max_tokens": 4096, "tools": TOOLS, "tool_choice": "auto"},
                             timeout=180).json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        tcs = msg.get("tool_calls", [])
        if not tcs or choice.get("finish_reason") == "stop":
            return {"response": msg.get("content", "") or ""}
        messages.append(msg)
        for tc in tcs:
            fn = tc.get("function", {})
            try:
                result = _tool(fn.get("name", ""), json.loads(fn.get("arguments", "{}")))
            except Exception as e:
                result = json.dumps({"error": str(e)})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    return {"response": "Max rounds reached."}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Define scorers (reusable for monitoring)

# COMMAND ----------

scorers = [
    RelevanceToQuery(),
    Safety(),
]

print(f"Scorers: {[type(s).__name__ for s in scorers]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run evaluation

# COMMAND ----------

eval_data = [{"inputs": {"question": row["question"]}} for _, row in eval_df.iterrows()]

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=scorers,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. View results

# COMMAND ----------

if hasattr(results, 'tables') and results.tables:
    for key in results.tables:
        print(f"Table: {key}")
        display(results.tables[key])
elif hasattr(results, 'metrics'):
    print("Metrics:", results.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Attach scorers for production monitoring
# MAGIC Register and start the same scorers on the experiment so they run on incoming traces.

# COMMAND ----------

from mlflow.genai.scorers import ScorerSamplingConfig, list_scorers, delete_scorer

# Clean up any existing broken scorers
try:
    existing = list_scorers()
    for s in existing:
        print(f"Removing existing scorer: {s.name}")
        try:
            s.stop()
        except Exception:
            pass
        delete_scorer(name=s.name)
except Exception as e:
    print(f"Cleanup skipped: {e}")

# Register and start fresh
for scorer in scorers:
    try:
        name = f"vi_agent_{type(scorer).__name__.lower()}"
        registered = scorer.register(name=name)
        registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
        print(f"Attached {type(scorer).__name__} as '{name}' (50% sampling)")
    except Exception as e:
        print(f"Scorer {type(scorer).__name__} attachment failed: {e}")
