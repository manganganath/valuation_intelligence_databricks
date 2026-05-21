# Databricks notebook source
# MAGIC %md
# MAGIC # Setup UC Functions
# MAGIC Creates 3 Unity Catalog functions for the Valuation Intelligence Agent:
# MAGIC - `get_company_financials` — retrieve financial data by ticker
# MAGIC - `get_risk_assessment` — PE divergence vs sector median
# MAGIC - `search_research_docs` — keyword search fallback for research docs

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
APP_SP_CLIENT_ID = _cfg.get("app_sp_client_id", "YOUR_APP_SP_CLIENT_ID")

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. get_company_financials

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_company_financials(p_ticker STRING)
RETURNS TABLE(
    ticker STRING,
    company STRING,
    sector STRING,
    geography STRING,
    revenue_usd DOUBLE,
    pe_ratio DOUBLE,
    ps_ratio DOUBLE,
    ebitda_margin_pct DOUBLE,
    analyst_rating STRING,
    target_price_usd DOUBLE
)
COMMENT 'Retrieves financial data for a company by ticker symbol. Returns key metrics including revenue, PE ratio, PS ratio, EBITDA margin, analyst rating, and target price.'
RETURN
    SELECT ticker, company, sector, geography, revenue_usd, pe_ratio, ps_ratio,
           ebitda_margin_pct, analyst_rating, target_price_usd
    FROM {CATALOG}.{SCHEMA}.company_financials
    WHERE UPPER(ticker) = UPPER(p_ticker)
""")

print("Created get_company_financials")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. get_risk_assessment

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_risk_assessment(p_ticker STRING)
RETURNS TABLE(
    ticker STRING,
    company STRING,
    sector STRING,
    pe_ratio DOUBLE,
    sector_median_pe DOUBLE,
    pe_divergence_pct DOUBLE,
    risk_level STRING
)
COMMENT 'Assesses valuation risk for a company by comparing its PE ratio to the sector median. Returns the PE divergence percentage and a risk level classification (LOW/MEDIUM/HIGH/CRITICAL).'
RETURN
    WITH sector_stats AS (
        SELECT sector, PERCENTILE_APPROX(pe_ratio, 0.5) AS sector_median_pe
        FROM {CATALOG}.{SCHEMA}.company_financials
        GROUP BY sector
    ),
    company_data AS (
        SELECT c.ticker, c.company, c.sector, c.pe_ratio, s.sector_median_pe,
               ROUND(((c.pe_ratio - s.sector_median_pe) / s.sector_median_pe) * 100, 1) AS pe_divergence_pct
        FROM {CATALOG}.{SCHEMA}.company_financials c
        JOIN sector_stats s ON c.sector = s.sector
        WHERE UPPER(c.ticker) = UPPER(p_ticker)
    )
    SELECT ticker, company, sector, pe_ratio, sector_median_pe, pe_divergence_pct,
           CASE
               WHEN ABS(pe_divergence_pct) > 40 THEN 'CRITICAL'
               WHEN ABS(pe_divergence_pct) > 25 THEN 'HIGH'
               WHEN ABS(pe_divergence_pct) > 10 THEN 'MEDIUM'
               ELSE 'LOW'
           END AS risk_level
    FROM company_data
""")

print("Created get_risk_assessment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. search_research_docs

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.search_research_docs(search_query STRING)
RETURNS TABLE(
    doc_id STRING,
    chunk_id STRING,
    company STRING,
    ticker STRING,
    doc_type STRING,
    content STRING
)
COMMENT 'Searches research documents by keyword. Pass a SINGLE short keyword or company name (e.g. NovaTech, concession, infrastructure). One word works best.'
RETURN
    SELECT doc_id, chunk_id, company, ticker, doc_type, content
    FROM {CATALOG}.{SCHEMA}.research_documents
    WHERE LOWER(content) LIKE CONCAT('%', LOWER(search_query), '%')
       OR LOWER(company) LIKE CONCAT('%', LOWER(search_query), '%')
       OR LOWER(ticker) LIKE CONCAT('%', LOWER(search_query), '%')
    LIMIT 10
""")

print("Created search_research_docs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Grant EXECUTE to App Service Principal

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# The app SP already has ALL PRIVILEGES on the schema (granted during workspace setup).
# ALL PRIVILEGES includes EXECUTE, so explicit per-function grants are not needed.
# We attempt them anyway for explicitness, but catch errors gracefully.
sp = None
for s in w.service_principals.list():
    if s.application_id == APP_SP_CLIENT_ID:
        sp = s
        break

if sp:
    for func_name in ["get_company_financials", "get_risk_assessment", "search_research_docs"]:
        try:
            spark.sql(f"GRANT EXECUTE ON FUNCTION {CATALOG}.{SCHEMA}.{func_name} TO `{sp.application_id}`")
            print(f"Granted EXECUTE on {func_name} to SP {sp.application_id}")
        except Exception as e:
            print(f"EXECUTE grant on {func_name} skipped (SP has ALL PRIVILEGES on schema): {e}")
else:
    print(f"SP {APP_SP_CLIENT_ID} not found, but ALL PRIVILEGES on schema covers EXECUTE.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify functions work

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.get_company_financials('PTIK')"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.get_risk_assessment('PTIK')"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.search_research_docs('concession risk')"))
