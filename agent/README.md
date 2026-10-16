# Innovation Investment COO Report Agent

## What it does
An autonomous AI agent that generates a structured risk report on the corporate innovation portfolio.

## How to run
1. Add `OPENROUTER_API_KEY` to `.env` in the project root
2. `uv sync`
3. `uv run streamlit run agent/streamlit_app.py`
4. Click "Generate COO Report"

## Tools available (6)
- `get_portfolio_overview()` — High-level stats
- `get_trend_over_time(metric)` — Year-by-year medians
- `get_segment_breakdown(dimension)` — Group-by analysis
- `run_rd_tobinq_regression()` — OLS regression
- `find_high_risk_companies()` — Risk discovery
- `get_industry_benchmark()` — Industry league table

## How it works
1. LLM reads the system prompt
2. Decides which tools to call (autonomous)
3. Each tool returns JSON
4. LLM synthesizes findings
5. Outputs a 7-section markdown report

## Logging
All tool calls are logged to `logs/week11_agent.log`.

## Limitations
- Descriptive only, no causal claims
- AI output should be verified against data
- No buy/sell advice
