# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Monitoring Dashboard
# MAGIC Reads traces from MLflow experiment, runs the same scorers used in evaluation,
# MAGIC and computes latency, volume, and quality metrics.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.1" "databricks-agents"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml
import mlflow
import pandas as pd
from mlflow.genai.scorers import RelevanceToQuery, Safety

_cfg_path = "/Workspace" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().rsplit("/src/", 1)[0] + "/config.yaml"
try:
    with open(_cfg_path) as f:
        _cfg = yaml.safe_load(f)
except Exception:
    _cfg = {}

EXPERIMENT_PATH = _cfg.get("mlflow_experiment", "/Shared/vi_demo/vi_demo")
mlflow.set_experiment(EXPERIMENT_PATH)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

print(f"Experiment: {experiment.name}")
print(f"Experiment ID: {experiment.experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load traces

# COMMAND ----------

traces = mlflow.search_traces(experiment_ids=[experiment.experiment_id])
print(f"Total traces: {len(traces)}")
if len(traces) > 0:
    display(traces.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Score existing traces with the same eval scorers
# MAGIC Runs the same judges used in evaluation on all existing traces.

# COMMAND ----------

if len(traces) > 0:
    scorers = [RelevanceToQuery(), Safety()]
    try:
        scored = mlflow.genai.evaluate(
            data=traces,
            scorers=scorers,
        )
        print("Scoring complete.")
        # Display results - try different accessors for compatibility
        if hasattr(scored, 'tables') and scored.tables:
            for key in scored.tables:
                print(f"Table: {key}")
                display(scored.tables[key])
        elif hasattr(scored, 'metrics'):
            print("Metrics:", scored.metrics)
    except Exception as e:
        print(f"Scoring failed: {e}")
        print("Showing raw traces instead:")
        display(traces.head(10))
else:
    print("No traces to score.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Request volume over time

# COMMAND ----------

if len(traces) > 0 and "timestamp_ms" in traces.columns:
    traces["timestamp"] = pd.to_datetime(traces["timestamp_ms"], unit="ms")
    volume = traces.set_index("timestamp").resample("1h").size().reset_index(name="request_count")
    display(volume)
elif len(traces) > 0:
    print(f"Columns available: {list(traces.columns)}")
    display(traces.head(20))
else:
    print("No traces yet.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Latency metrics

# COMMAND ----------

if len(traces) > 0 and "execution_time_ms" in traces.columns:
    latency = traces["execution_time_ms"].describe()
    print("=== Latency (ms) ===")
    print(f"  Count:  {latency['count']:.0f}")
    print(f"  Mean:   {latency['mean']:.0f}")
    print(f"  Median: {latency['50%']:.0f}")
    print(f"  p95:    {traces['execution_time_ms'].quantile(0.95):.0f}")
    print(f"  Max:    {latency['max']:.0f}")
else:
    print("No latency data available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Status breakdown

# COMMAND ----------

if len(traces) > 0 and "status" in traces.columns:
    status_counts = traces["status"].value_counts()
    print("=== Status Breakdown ===")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    error_rate = (status_counts.get("ERROR", 0) / len(traces)) * 100
    print(f"\n  Error rate: {error_rate:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Summary

# COMMAND ----------

print("=== Monitoring Summary ===")
print(f"Experiment: {EXPERIMENT_PATH}")
print(f"Total traces: {len(traces)}")
if len(traces) > 0:
    if "execution_time_ms" in traces.columns:
        print(f"Avg latency: {traces['execution_time_ms'].mean():.0f} ms")
        print(f"p95 latency: {traces['execution_time_ms'].quantile(0.95):.0f} ms")
    if "status" in traces.columns:
        errors = (traces["status"] == "ERROR").sum()
        print(f"Errors: {errors}/{len(traces)} ({errors/len(traces)*100:.1f}%)")
