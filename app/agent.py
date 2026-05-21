"""Agent with client-side tool loop using Genie Space, Knowledge Assistant, and UC functions.

LLM generates tool_calls -> we execute via Genie/KA/SQL APIs -> feed results back -> stream final response.
Tools run in parallel for speed. MLflow tracing on each tool.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType

from config import (
    AGENT_SYSTEM_PROMPT,
    CATALOG,
    GENIE_SPACE_ID,
    KNOWLEDGE_ASSISTANT_ID,
    MODEL,
    SCHEMA,
    SQL_WAREHOUSE_ID,
)

logger = logging.getLogger(__name__)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_financials",
            "description": "Queries company financial data using natural language. Can answer questions about revenue, PE ratio, PS ratio, EBITDA margin, analyst ratings, target prices, and comparisons across companies and sectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Natural language question about company financials (e.g. 'What are the financial metrics for PT Infratek Nusantara?', 'Compare PE ratios across infrastructure sector')"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_research",
            "description": "Searches research documents, analyst memos, and qualitative insights using vector similarity. Use for finding risk analysis, growth drivers, concession details, competitive positioning, and company-specific research.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query about a company or topic (e.g. 'PT Infratek concession risk', 'NovaTech growth drivers')"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_assessment",
            "description": "Assesses valuation risk by comparing a company's PE ratio to its sector median. Returns PE divergence percentage and risk level (LOW/MEDIUM/HIGH/CRITICAL). Requires the exact ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "p_ticker": {"type": "string", "description": "Exact ticker symbol (e.g. PTIFK-JK, NVTC-TW, SCDC-SG, MGRH-AU)"}
                },
                "required": ["p_ticker"],
            },
        },
    },
]

MAX_TOOL_ROUNDS = 5

# Reusable HTTP session for connection pooling
_http_session = requests.Session()
_http_session.headers.update({"Content-Type": "application/json"})

# Cached auth
_cached_auth = {"headers": None, "host": None, "expires": 0}


def _get_auth_headers() -> dict:
    """Get auth headers with caching (refresh every 30 min)."""
    now = time.time()
    if _cached_auth["headers"] and now < _cached_auth["expires"]:
        return dict(_cached_auth["headers"])

    w = WorkspaceClient()
    header_factory = w.config.authenticate
    headers = header_factory()
    headers["Content-Type"] = "application/json"
    _cached_auth["headers"] = headers
    _cached_auth["host"] = w.config.host.rstrip("/")
    _cached_auth["expires"] = now + 1800  # 30 min
    return dict(headers)


def _get_host() -> str:
    if _cached_auth["host"]:
        return _cached_auth["host"]
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if not host:
        w = WorkspaceClient()
        host = w.config.host.rstrip("/")
    return host


def _call_llm(messages: list[dict], headers: dict, stream: bool = False):
    url = f"{_get_host()}/serving-endpoints/{MODEL}/invocations"
    payload = {"messages": messages, "max_tokens": 4096, "tools": TOOLS, "tool_choice": "auto", "stream": stream}
    resp = _http_session.post(url, headers=headers, json=payload, timeout=180, stream=stream)
    resp.raise_for_status()
    if stream:
        return resp
    return resp.json()


# --- Tool Implementations ---

@mlflow.trace(name="genie_space", span_type=SpanType.TOOL)
def _query_genie_space(question: str, headers: dict, host: str) -> str:
    """Query Genie Space with fast polling."""
    try:
        resp = _http_session.post(
            f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation",
            headers=headers, json={"content": question}, timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        conv_id = data["conversation_id"]
        msg_id = data["message_id"]

        # Fast polling: 1s for first 10s, then 2s
        for i in range(30):
            time.sleep(1 if i < 10 else 2)
            resp = _http_session.get(
                f"{host}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conv_id}/messages/{msg_id}",
                headers=headers, timeout=30,
            )
            resp.raise_for_status()
            msg = resp.json()
            status = msg.get("status", "")
            if status in ("COMPLETED", "FAILED"):
                break

        if msg.get("status") != "COMPLETED":
            return json.dumps({"error": f"Genie query did not complete: {msg.get('status')}"})

        for att in msg.get("attachments", []):
            if "query" in att:
                query_info = att["query"]
                statement_id = query_info.get("statement_id", "")
                if statement_id:
                    result_resp = _http_session.get(
                        f"{host}/api/2.0/sql/statements/{statement_id}",
                        headers=headers, timeout=30,
                    )
                    result_resp.raise_for_status()
                    result_data = result_resp.json()
                    columns = [c["name"] for c in result_data.get("manifest", {}).get("schema", {}).get("columns", [])]
                    rows = result_data.get("result", {}).get("data_array", [])
                    results = [dict(zip(columns, row)) for row in rows]
                    return json.dumps({
                        "description": query_info.get("description", ""),
                        "sql": query_info.get("query", ""),
                        "results": results,
                    }, default=str)

        return json.dumps({"error": "No query results in Genie response"})
    except Exception as e:
        logger.error(f"Genie Space error: {e}")
        return json.dumps({"error": str(e)})


@mlflow.trace(name="knowledge_assistant", span_type=SpanType.RETRIEVER)
def _query_knowledge_assistant(query: str, headers: dict, host: str) -> str:
    """Query Knowledge Assistant."""
    try:
        resp = _http_session.post(
            f"{host}/serving-endpoints/{KNOWLEDGE_ASSISTANT_ID}/invocations",
            headers=headers,
            json={"input": [{"role": "user", "content": query}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        texts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text" and block.get("text", "").strip():
                        texts.append(block["text"].strip())
        return "\n".join(texts) if texts else "No results found."
    except Exception as e:
        logger.warning(f"Knowledge Assistant error: {e}")
        return json.dumps({"error": str(e)})


@mlflow.trace(name="risk_assessment", span_type=SpanType.TOOL)
def _execute_risk_assessment(ticker: str, headers: dict, host: str) -> str:
    """Execute risk assessment UC function via SQL."""
    resp = _http_session.post(
        f"{host}/api/2.0/sql/statements",
        headers=headers,
        json={"warehouse_id": SQL_WAREHOUSE_ID, "statement": f"SELECT * FROM {CATALOG}.{SCHEMA}.get_risk_assessment('{ticker}')", "wait_timeout": "30s"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status", {}).get("state") != "SUCCEEDED":
        return json.dumps({"error": data.get("status", {}).get("error", {}).get("message", "SQL failed")})
    columns = [c["name"] for c in data.get("manifest", {}).get("schema", {}).get("columns", [])]
    rows = data.get("result", {}).get("data_array", [])
    return json.dumps([dict(zip(columns, row)) for row in rows], default=str)


@mlflow.trace(name="execute_tool", span_type=SpanType.TOOL)
def _execute_tool(name: str, arguments: dict, headers: dict, host: str) -> str:
    if name == "query_financials":
        return _query_genie_space(arguments["question"], headers, host)
    elif name == "search_research":
        return _query_knowledge_assistant(arguments["query"], headers, host)
    elif name == "get_risk_assessment":
        return _execute_risk_assessment(arguments["p_ticker"], headers, host)
    return json.dumps({"error": f"Unknown tool: {name}"})


# --- SSE + Memory Helpers ---

def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_system_prompt(memories: list[dict]) -> str:
    prompt = AGENT_SYSTEM_PROMPT
    if memories:
        memory_text = "\n".join(f"- {m['topic']}: {m['content']}" for m in memories)
        prompt += (
            "\n\nYou have the following long-term memories from previous sessions. "
            "Use these to personalize your response and avoid re-asking for information "
            "the user has already discussed:\n" + memory_text
        )
    return prompt


def _extract_memories(user_message: str, assistant_response: str, headers: dict) -> list[dict]:
    if not assistant_response or len(assistant_response) < 100:
        return []
    url = f"{_get_host()}/serving-endpoints/{MODEL}/invocations"
    payload = {
        "messages": [
            {"role": "system", "content": (
                "Extract 1-3 key factual insights from this conversation that would be useful "
                "to remember for future sessions. Return a JSON array of objects with 'topic' "
                "and 'content' keys. Topics should be company tickers or themes. "
                "Keep content concise (1-2 sentences). Only extract concrete facts, not opinions. "
                "If there are no memorable insights, return an empty array []."
            )},
            {"role": "user", "content": f"User asked: {user_message}\n\nAgent responded: {assistant_response[:2000]}"},
        ],
        "max_tokens": 500,
    }
    try:
        resp = _http_session.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(content)
    except Exception as e:
        logger.debug(f"Memory extraction failed: {e}")
        return []


def _stream_final_response(messages: list[dict], headers: dict, response_id: str) -> Generator[str, None, None]:
    """Stream a final LLM response token-by-token."""
    stream_resp = _call_llm(messages, headers, stream=True)
    full_content = ""
    for line in stream_resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        response_id = chunk.get("id", response_id)
        content_delta = delta.get("content", "")
        if content_delta:
            full_content += content_delta
            yield _sse_event("token", {"text": content_delta})
    stream_resp.close()
    yield _sse_event("done", {"content": full_content, "response_id": response_id})


def _execute_tools_parallel(tool_calls: list[dict], headers: dict, host: str) -> list[tuple[str, str, dict, str]]:
    """Execute multiple tool calls in parallel. Returns [(tool_call_id, tool_name, tool_args, result), ...]."""
    results = {}

    def run_tool(tc):
        func = tc.get("function", {})
        tool_name = func.get("name", "")
        tool_args = json.loads(func.get("arguments", "{}"))
        tool_call_id = tc.get("id", "")
        try:
            result = _execute_tool(tool_name, tool_args, headers, host)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            result = json.dumps({"error": f"Tool unavailable: {str(e)[:200]}"})
        return tool_call_id, tool_name, tool_args, result

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_tool, tc): tc for tc in tool_calls}
        for future in as_completed(futures):
            tool_call_id, tool_name, tool_args, result = future.result()
            results[tool_call_id] = (tool_call_id, tool_name, tool_args, result)

    # Return in original order
    return [results[tc.get("id", "")] for tc in tool_calls if tc.get("id", "") in results]


def call_agent_stream(user_message: str, conversation_history: list[dict] | None = None, memories: list[dict] | None = None) -> Generator[str, None, None]:
    """Run agent loop yielding SSE events. Tools execute in parallel."""
    system_prompt = _build_system_prompt(memories or [])
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    headers = _get_auth_headers()
    host = _get_host()
    response_id = ""

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            data = _call_llm(messages, headers, stream=False)
            response_id = data.get("id", response_id)
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls or choice.get("finish_reason") == "stop":
                yield from _stream_final_response(messages, headers, response_id)
                return

            messages.append(message)

            # Emit all tool call traces first
            for tc in tool_calls:
                func = tc.get("function", {})
                yield _sse_event("trace", {"type": "tool_call", "label": func.get("name", ""), "data": func.get("arguments", "{}")})

            # Execute all tools in parallel
            results = _execute_tools_parallel(tool_calls, headers, host)

            # Emit results and add to messages
            for tool_call_id, tool_name, tool_args, result in results:
                yield _sse_event("trace", {"type": "tool_result", "label": tool_name, "data": result[:500]})
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

            # Stream the next LLM response
            stream_resp = _call_llm(messages, headers, stream=True)
            full_content = ""
            has_tool_calls = False
            tc_buffer = {}

            for line in stream_resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                response_id = chunk.get("id", response_id)

                if delta.get("tool_calls"):
                    has_tool_calls = True
                    for tcd in delta["tool_calls"]:
                        idx = tcd.get("index", 0)
                        if idx not in tc_buffer:
                            tc_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                        if tcd.get("id"):
                            tc_buffer[idx]["id"] = tcd["id"]
                        if tcd.get("function", {}).get("name"):
                            tc_buffer[idx]["name"] = tcd["function"]["name"]
                        if tcd.get("function", {}).get("arguments"):
                            tc_buffer[idx]["arguments"] += tcd["function"]["arguments"]

                content_delta = delta.get("content", "")
                if content_delta:
                    full_content += content_delta
                    yield _sse_event("token", {"text": content_delta})

            stream_resp.close()

            if has_tool_calls and tc_buffer:
                assistant_msg = {"role": "assistant", "content": full_content or None, "tool_calls": [
                    {"id": tc_buffer[i]["id"], "type": "function", "function": {"name": tc_buffer[i]["name"], "arguments": tc_buffer[i]["arguments"]}}
                    for i in sorted(tc_buffer.keys())
                ]}
                messages.append(assistant_msg)

                # Parallel execution for streamed tool calls too
                for tc in assistant_msg["tool_calls"]:
                    func = tc["function"]
                    yield _sse_event("trace", {"type": "tool_call", "label": func["name"], "data": func["arguments"]})

                results = _execute_tools_parallel(assistant_msg["tool_calls"], headers, host)
                for tool_call_id, tool_name, tool_args, result in results:
                    yield _sse_event("trace", {"type": "tool_result", "label": tool_name, "data": result[:500]})
                    messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
                continue

            yield _sse_event("done", {"content": full_content, "response_id": response_id})
            return

        yield _sse_event("done", {"content": "Agent reached maximum tool rounds.", "response_id": response_id})

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        yield _sse_event("error", {"message": str(e)})
