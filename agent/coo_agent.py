"""
COO Report Agent — autonomous tool-calling loop.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Import tool registry
from tools import tool_functions


COO_SYSTEM_PROMPT = """You are a Senior Innovation Investment Analyst.
Write a risk report for the COO on the corporate innovation portfolio.

Use the tools to gather evidence. You decide which tools to call, in what order, and how many times.

WORKFLOW:
1. get_portfolio_overview
2. get_trend_over_time for 'rd_intensity' and 'tobinq'
3. get_segment_breakdown('industry')
4. run_rd_tobinq_regression
5. find_high_risk_companies
6. get_industry_benchmark

Your final output must be a markdown report with exactly these sections:
# Innovation Investment Risk Report — COO Briefing
## 1. Executive Summary
## 2. Portfolio Overview
## 3. Trends Over Time
## 4. Industry Comparison
## 5. Statistical Evidence (R&D vs Tobin's Q)
## 6. Risk Flags & Alerts
## 7. Recommendations

Rules:
- Base every claim on a tool result. Never invent numbers.
- State data limitations explicitly.
- No buy/sell advice, no forecasts.
"""


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_overview",
            "description": "Return high-level stats for the innovation portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {"type": "string", "description": "Optional industry filter."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trend_over_time",
            "description": "Return year-by-year median values for a metric.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["rd_intensity", "roa", "tobinq"]},
                    "industry": {"type": "string"},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_segment_breakdown",
            "description": "Return per-segment medians for key metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": ["industry", "year", "name"]},
                },
                "required": ["dimension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rd_tobinq_regression",
            "description": "Run OLS regression of Tobin's Q on R&D intensity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_high_risk_companies",
            "description": "Find companies with high R&D intensity, low/negative ROA, and low Tobin's Q.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_rd_intensity": {"type": "number"},
                    "max_roa": {"type": "number"},
                    "max_tobinq": {"type": "number"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_industry_benchmark",
            "description": "Return the top-N industries ranked by median R&D intensity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {"type": "integer"},
                },
            },
        },
    },
]


def execute_tool_call(tool_name: str, tool_arguments: dict) -> dict | list:
    fn = tool_functions.get(tool_name)
    if fn is None:
        logger.error(f"Unknown tool: {tool_name}")
        return {"error": f"Tool '{tool_name}' is not available."}
    try:
        return fn(**(tool_arguments or {}))
    except TypeError as exc:
        logger.exception(f"Invalid arguments for {tool_name}: {exc}")
        return {"error": f"Invalid arguments for {tool_name}: {exc}"}
    except Exception as exc:
        logger.exception(f"Tool {tool_name} crashed: {exc}")
        return {"error": f"Tool {tool_name} failed: {exc}"}


def run_coo_report_agent(model: str | None = None, max_iterations: int = 15) -> str:
    model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        return "Error: OPENROUTER_API_KEY is not set. Check the .env file in the project root."

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages = [
        {"role": "system", "content": COO_SYSTEM_PROMPT},
        {"role": "user", "content": "Write the COO risk report for the innovation portfolio."},
    ]

    iteration = 0
    total_tool_calls = 0

    while iteration < max_iterations:
        iteration += 1
        logger.info(f"Agent iteration {iteration} | messages={len(messages)}")

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                max_tokens=4000,
                temperature=0.2,
            )
        except Exception as exc:
            logger.error(f"API call failed: {type(exc).__name__}: {exc}")
            return f"The AI service call failed: {exc}"

        if not response.choices:
            return "The AI service returned no response."

        assistant_message = response.choices[0].message

        if not assistant_message.tool_calls:
            logger.info(f"Agent finished | {total_tool_calls} tool calls | {iteration} iterations")
            return assistant_message.content or "(empty report)"

        msg_dump = assistant_message.model_dump()
        msg_dump = {k: v for k, v in msg_dump.items() if v is not None}
        if not msg_dump.get("content"):
            msg_dump["content"] = ""
        messages.append(msg_dump)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool_call(tool_name, args)
            total_tool_calls += 1
            logger.info(f"Tool call #{total_tool_calls}: {tool_name}({args})")
            preview = json.dumps(result, default=str)[:200]
            logger.info(f"Tool result #{total_tool_calls}: {preview}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(result, default=str),
            })

    return "Agent reached maximum iterations without producing a final report."