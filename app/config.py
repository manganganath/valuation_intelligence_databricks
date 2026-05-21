"""Centralized configuration — reads from config.yaml at project root."""

import os
import yaml

# Load config.yaml (look in parent dir since app/ is the working dir in Databricks Apps)
_config_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    "config.yaml",
]

_cfg = {}
for path in _config_paths:
    if os.path.exists(path):
        with open(path) as f:
            _cfg = yaml.safe_load(f)
        break

if not _cfg:
    raise FileNotFoundError(f"config.yaml not found in: {_config_paths}")

# Unity Catalog
CATALOG = _cfg["catalog"]
SCHEMA = _cfg["schema"]

# LLM
MODEL = _cfg["model"]

# Tool resource IDs
KNOWLEDGE_ASSISTANT_ID = _cfg["knowledge_assistant_id"]
GENIE_SPACE_ID = _cfg["genie_space_id"]
SQL_WAREHOUSE_ID = _cfg["sql_warehouse_id"]

# Lakebase
LAKEBASE_HOST = _cfg["lakebase_host"]
LAKEBASE_DB = _cfg["lakebase_db"]

# System prompt (not in YAML — it's code, not config)
AGENT_SYSTEM_PROMPT = (
    "You are a valuation intelligence agent for institutional fund managers. "
    "You have three tools available:\n"
    "- query_financials: for financial data (revenue, PE, margins, ratings)\n"
    "- search_research: for qualitative research, analyst memos, risk reports\n"
    "- get_risk_assessment: for PE divergence vs sector median\n\n"
    "TOOL SELECTION RULES:\n"
    "- Only call the tools you actually need. Do NOT call all tools for every query.\n"
    "- If the user asks about financial metrics only, just call query_financials.\n"
    "- If the user asks about risks or research, just call search_research.\n"
    "- For a full valuation analysis, call query_financials and search_research. "
    "Only add get_risk_assessment if PE comparison is specifically relevant.\n"
    "- For follow-up questions, use conversation context first. Only call tools if NEW data is needed.\n"
    "- Never explain to the user whether or not you are calling tools. Just answer directly.\n\n"
    "FORMAT: Use markdown. Be quantitative and precise. This audience is fund managers and quants. "
    "Use plain hyphens (-), straight quotes, and standard ASCII punctuation only."
)

# Preset companies for the UI
PRESET_COMPANIES = [
    {"name": "PT Infratek Nusantara", "query": "What is the valuation outlook for PT Infratek Nusantara (PTIFK-JK)? Include key financial metrics, research insights, and risk factors."},
    {"name": "NovaTech Semiconductors", "query": "Provide a valuation analysis for NovaTech Semiconductors (NVTC-TW). What are the key drivers and risks?"},
    {"name": "SingCore Data Centres", "query": "Analyze SingCore Data Centres (SCDC-SG) valuation. How does it compare to sector peers?"},
    {"name": "Meridian Grid Holdings", "query": "What is the investment case for Meridian Grid Holdings (MGRH-AU)? Assess valuation and risk factors."},
]
