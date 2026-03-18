# -*- coding: utf-8 -*-
"""Pedagogical Streamlit Deep Agent app for stock research.

This version is intentionally designed for teaching:
- a visible main chat UI for the user request and final answer
- parallel visible agent panels for Fundamental, Technical, and Risk analysis
- each agent panel shows:
  1. the request it received
  2. the tool call it made
  3. the tool response it received
  4. the agent result it produced
- a final synthesis step combines the specialist outputs into one structured report

Why this version does manual orchestration instead of hidden internal subagents:
Deep Agent subagents are useful, but their internal reasoning is not automatically visible
in a Streamlit interface. For pedagogy, we explicitly run specialized agents in parallel and
log their tool interactions so students can see what happened.
"""

import asyncio
import logging
import os
import threading
import time
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

import streamlit as st
import yfinance as yf
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Environment configuration
# -----------------------------------------------------------------------------
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FinanceAgent"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("langsmith").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Module-level log buffer
# Streamlit session_state is not safe to touch from worker threads used by tools.
# Tools therefore write here, and the UI reads from here.
# -----------------------------------------------------------------------------
AGENT_NAMES = (
    "Fundamental Agent",
    "Technical Agent",
    "Risk Agent",
    "Synthesis Agent",
)

AGENT_LOG_BUFFER: dict[str, list[dict[str, Any]]] = {
    name: [] for name in AGENT_NAMES
}

AGENT_LOG_LOCK = threading.Lock()

# -----------------------------------------------------------------------------
# Streamlit session helpers
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize Streamlit state used by the UI."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_logs" not in st.session_state:
        st.session_state.agent_logs = {name: [] for name in AGENT_NAMES}
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_elapsed_seconds" not in st.session_state:
        st.session_state.last_elapsed_seconds = None


def reset_agent_logs() -> None:
    """Clear per-agent visible logs before a new run."""
    global AGENT_LOG_BUFFER
    with AGENT_LOG_LOCK:
        AGENT_LOG_BUFFER = {name: [] for name in AGENT_NAMES}
    try:
        st.session_state.agent_logs = {name: [] for name in AGENT_NAMES}
    except Exception:
        pass


def add_agent_log(agent_name: str, kind: str, payload: Any) -> None:
    """Append a visible event to the chosen agent panel.

    This function must be safe to call from worker threads, so it must not rely
    on Streamlit APIs for correctness.
    """
    event = {
        "kind": kind,
        "payload": payload,
    }
    with AGENT_LOG_LOCK:
        AGENT_LOG_BUFFER.setdefault(agent_name, [])
        AGENT_LOG_BUFFER[agent_name].append(event)

def get_agent_log_snapshot() -> dict[str, list[dict[str, Any]]]:
    """Return a thread-safe snapshot of the current agent logs."""
    with AGENT_LOG_LOCK:
        return {
            name: [dict(event) for event in events]
            for name, events in AGENT_LOG_BUFFER.items()
        }


def run_workflow_in_background(user_query: str, result_holder: dict[str, Any]) -> None:
    """Run the async workflow in a background thread and store the result."""
    try:
        result_holder["report"] = asyncio.run(run_full_workflow(user_query))
    except Exception as exc:
        result_holder["error"] = exc
    finally:
        result_holder["done"] = True
        
# -----------------------------------------------------------------------------
# Helpers for clean, consistent tool payloads
# -----------------------------------------------------------------------------
def _to_float(value: Any) -> float | None:
    """Convert a value to float when possible, otherwise return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    """Convert a value to int when possible, otherwise return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# Tool-call visibility plumbing
# Each tool reads the current agent name from a context variable so it can log
# visible tool activity into the correct Streamlit panel.
# -----------------------------------------------------------------------------
CURRENT_AGENT_NAME: ContextVar[str] = ContextVar(
    "CURRENT_AGENT_NAME",
    default="Unknown Agent",
)


def _log_tool_call(tool_name: str, tool_args: dict[str, Any]) -> None:
    agent_name = CURRENT_AGENT_NAME.get()
    add_agent_log(
        agent_name,
        "tool_call",
        {
            "tool_name": tool_name,
            "arguments": tool_args,
        },
    )


def _log_tool_response(tool_name: str, response: dict[str, Any]) -> None:
    agent_name = CURRENT_AGENT_NAME.get()
    add_agent_log(
        agent_name,
        "tool_response",
        {
            "tool_name": tool_name,
            "response": response,
        },
    )


# -----------------------------------------------------------------------------
# Structured output schemas
# -----------------------------------------------------------------------------
class SpecialistReport(BaseModel):
    """Structured output from a specialist agent."""

    agent_name: str = Field(description="Name of the specialist agent")
    ticker: str = Field(description="Primary ticker analyzed")
    summary: str = Field(description="Main summary from this specialist")
    key_points: list[str] = Field(
        default_factory=list,
        description="Important bullet-point findings",
    )
    risks_or_caveats: list[str] = Field(
        default_factory=list,
        description="Important limitations, risks, or caveats",
    )
    methodology_note: str = Field(
        description="What tools were used and what the agent could or could not conclude",
    )


class StockResearchReport(BaseModel):
    """Structured final output for the full stock research workflow."""

    ticker: str = Field(description="Primary ticker analyzed, for example AAPL")
    company_name: str | None = Field(
        default=None,
        description="Company name if available",
    )
    recommendation: str = Field(
        description="Final action recommendation such as Buy, Hold, Sell, or Watch",
    )
    target_price: float | None = Field(
        default=None,
        description="Base-case target price if one is given",
    )
    time_horizon: str = Field(
        description="Time horizon for the recommendation, for example 12 months",
    )
    fundamental_summary: str = Field(
        description="Concise summary of key fundamental findings",
    )
    technical_summary: str = Field(
        description="Concise summary of key technical findings",
    )
    risk_summary: str = Field(
        description="Concise summary of key risks and caveats",
    )
    bull_case: str = Field(description="Main upside argument")
    bear_case: str = Field(description="Main downside argument")
    risks: list[str] = Field(
        default_factory=list,
        description="Key investment risks",
    )
    peer_comparison: str = Field(
        description="Short comparison with peers based on available tool data",
    )
    news_summary: str = Field(
        description="Short summary of recent news headlines gathered by the tools",
    )
    methodology_note: str = Field(
        description="Brief explanation of what data/tools were used and any limitations",
    )


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
openai_model = init_chat_model(
    "openai:gpt-5-mini",
    max_retries=8,
    timeout=120,
    temperature=0,
)


# -----------------------------------------------------------------------------
# Tools
# These are read-only tools, so human approval is not needed.
# If you later add side-effecting tools (trade execution, email, file writes),
# use interrupt_on plus a checkpointer.
# -----------------------------------------------------------------------------
@tool

def get_stock_price(symbol: str) -> dict[str, Any]:
    """Get current stock price and basic valuation snapshot for one ticker."""
    _log_tool_call("get_stock_price", {"symbol": symbol})
    logger.info("[TOOL] get_stock_price(%s)", symbol)
    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
        hist = stock.history(period="5d")
        if hist.empty:
            response = {
                "ok": False,
                "symbol": symbol,
                "error": "No recent historical price data was returned.",
            }
            _log_tool_response("get_stock_price", response)
            return response

        current_price = _to_float(hist["Close"].iloc[-1])
        response = {
            "ok": True,
            "symbol": symbol,
            "company_name": info.get("longName") or info.get("shortName"),
            "currency": info.get("currency"),
            "current_price": current_price,
            "market_cap": _to_int(info.get("marketCap")),
            "trailing_pe": _to_float(info.get("trailingPE")),
            "forward_pe": _to_float(info.get("forwardPE")),
            "fifty_two_week_high": _to_float(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": _to_float(info.get("fiftyTwoWeekLow")),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
        _log_tool_response("get_stock_price", response)
        return response
    except Exception as exc:
        logger.exception("get_stock_price failed")
        response = {"ok": False, "symbol": symbol, "error": str(exc)}
        _log_tool_response("get_stock_price", response)
        return response


@tool

def get_financial_statements(symbol: str) -> dict[str, Any]:
    """Retrieve a concise snapshot from the latest available financial statements."""
    _log_tool_call("get_financial_statements", {"symbol": symbol})
    logger.info("[TOOL] get_financial_statements(%s)", symbol)
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet

        if financials is None or financials.empty:
            response = {
                "ok": False,
                "symbol": symbol,
                "error": "Financial statements were not available.",
            }
            _log_tool_response("get_financial_statements", response)
            return response

        latest_period = financials.columns[0]
        total_revenue = _to_float(
            financials.loc["Total Revenue", latest_period]
        ) if "Total Revenue" in financials.index else None
        net_income = _to_float(
            financials.loc["Net Income", latest_period]
        ) if "Net Income" in financials.index else None
        total_assets = _to_float(
            balance_sheet.loc["Total Assets", latest_period]
        ) if "Total Assets" in balance_sheet.index else None
        total_debt = _to_float(
            balance_sheet.loc["Total Debt", latest_period]
        ) if "Total Debt" in balance_sheet.index else None

        response = {
            "ok": True,
            "symbol": symbol,
            "period": str(latest_period.date()) if hasattr(latest_period, "date") else str(latest_period),
            "revenue": total_revenue,
            "net_income": net_income,
            "total_assets": total_assets,
            "total_debt": total_debt,
            "net_margin": (net_income / total_revenue) if total_revenue and net_income is not None else None,
            "debt_to_assets": (total_debt / total_assets) if total_assets and total_debt is not None else None,
        }
        _log_tool_response("get_financial_statements", response)
        return response
    except Exception as exc:
        logger.exception("get_financial_statements failed")
        response = {"ok": False, "symbol": symbol, "error": str(exc)}
        _log_tool_response("get_financial_statements", response)
        return response


@tool

def get_technical_indicators(symbol: str, period: str = "6mo") -> dict[str, Any]:
    """Calculate a simple technical snapshot using moving averages and RSI."""
    _log_tool_call("get_technical_indicators", {"symbol": symbol, "period": period})
    logger.info("[TOOL] get_technical_indicators(%s, period=%s)", symbol, period)
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            response = {
                "ok": False,
                "symbol": symbol,
                "error": "No historical data was available for the requested period.",
            }
            _log_tool_response("get_technical_indicators", response)
            return response

        hist["SMA_20"] = hist["Close"].rolling(window=20).mean()
        hist["SMA_50"] = hist["Close"].rolling(window=50).mean()

        delta = hist["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        latest = hist.iloc[-1]
        close_price = _to_float(latest["Close"])
        sma_20 = _to_float(latest["SMA_20"])
        sma_50 = _to_float(latest["SMA_50"])
        latest_rsi = _to_float(rsi.iloc[-1])

        trend_signal = "mixed"
        if close_price is not None and sma_20 is not None and sma_50 is not None:
            if close_price > sma_20 > sma_50:
                trend_signal = "bullish"
            elif close_price < sma_20 < sma_50:
                trend_signal = "bearish"

        response = {
            "ok": True,
            "symbol": symbol,
            "period": period,
            "current_price": close_price,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "rsi": latest_rsi,
            "volume": _to_int(latest["Volume"]),
            "trend_signal": trend_signal,
        }
        _log_tool_response("get_technical_indicators", response)
        return response
    except Exception as exc:
        logger.exception("get_technical_indicators failed")
        response = {"ok": False, "symbol": symbol, "error": str(exc)}
        _log_tool_response("get_technical_indicators", response)
        return response


@tool

def get_recent_company_news(symbol: str, limit: int = 5) -> dict[str, Any]:
    """Return a small set of recent Yahoo Finance news headlines for a ticker."""
    _log_tool_call("get_recent_company_news", {"symbol": symbol, "limit": limit})
    logger.info("[TOOL] get_recent_company_news(%s, limit=%s)", symbol, limit)
    try:
        stock = yf.Ticker(symbol)
        raw_news = getattr(stock, "news", []) or []
        items: list[dict[str, Any]] = []
        for article in raw_news[:limit]:
            content = article.get("content") or {}
            items.append(
                {
                    "title": content.get("title") or article.get("title"),
                    "summary": content.get("summary"),
                    "publisher": content.get("provider", {}).get("displayName") or article.get("publisher"),
                    "published": content.get("pubDate") or article.get("providerPublishTime"),
                    "url": content.get("canonicalUrl", {}).get("url") or article.get("link"),
                }
            )
        response = {
            "ok": True,
            "symbol": symbol,
            "news_items": items,
            "source_note": "Yahoo Finance news feed via yfinance",
        }
        _log_tool_response("get_recent_company_news", response)
        return response
    except Exception as exc:
        logger.exception("get_recent_company_news failed")
        response = {"ok": False, "symbol": symbol, "error": str(exc)}
        _log_tool_response("get_recent_company_news", response)
        return response


@tool

def compare_peer_snapshot(symbols: list[str]) -> dict[str, Any]:
    """Compare a small list of tickers using price, market cap, and P/E data."""
    _log_tool_call("compare_peer_snapshot", {"symbols": symbols})
    logger.info("[TOOL] compare_peer_snapshot(%s)", symbols)
    comparisons: list[dict[str, Any]] = []
    for symbol in symbols:
        snapshot = get_stock_price.invoke({"symbol": symbol})
        comparisons.append(snapshot)
    response = {
        "ok": True,
        "symbols": symbols,
        "comparisons": comparisons,
        "methodology": "Simple peer snapshot using the same get_stock_price tool for each ticker.",
    }
    _log_tool_response("compare_peer_snapshot", response)
    return response


# -----------------------------------------------------------------------------
# Prompts aligned to actual tool capabilities
# -----------------------------------------------------------------------------
BASE_CAPABILITIES_NOTE = """
You may use only the tools available to you in this demo.
Available capabilities:
- basic stock price and valuation snapshot
- latest financial statement snapshot
- simple technical indicators (SMA20, SMA50, RSI, volume)
- recent Yahoo Finance news headlines
- simple peer comparison snapshot

Important limitations:
- no direct SEC filing parsing
- no earnings-call transcript tools
- no institutional research database
- news may be incomplete because it comes from the Yahoo Finance feed via yfinance

Pedagogical requirements:
- explain why each tool was used
- prefer clarity over brevity
- state missing data explicitly
- produce structured output that matches the requested schema exactly
"""


FUNDAMENTAL_INSTRUCTIONS = f"""You are the Fundamental Agent.
Your job is to analyze valuation, balance-sheet strength, profitability, peer context, and relevant recent news.
Use tools that help you assess fundamentals and peer comparison.
{BASE_CAPABILITIES_NOTE}
"""


TECHNICAL_INSTRUCTIONS = f"""You are the Technical Agent.
Your job is to analyze trend, momentum, RSI, moving averages, and practical short-term trading context.
Use tools that help you assess technical conditions and any recent headlines that may affect price action.
{BASE_CAPABILITIES_NOTE}
"""


RISK_INSTRUCTIONS = f"""You are the Risk Agent.
Your job is to identify downside cases, valuation risks, data limitations, macro risk, concentration risk, and uncertainty.
Use tools that help you identify where the thesis could fail.
{BASE_CAPABILITIES_NOTE}
"""


SYNTHESIS_INSTRUCTIONS = f"""You are the Synthesis Agent.
You will receive the outputs of three specialist agents: Fundamental, Technical, and Risk.
Your job is to combine them into one balanced, educational final investment report.
You may use tools again when useful, but only if needed.
Produce the final answer in the exact StockResearchReport schema.
{BASE_CAPABILITIES_NOTE}
"""


# -----------------------------------------------------------------------------
# Agent factory + shared checkpointer
# -----------------------------------------------------------------------------
TOOLS = [
    get_stock_price,
    get_financial_statements,
    get_technical_indicators,
    get_recent_company_news,
    compare_peer_snapshot,
]

CHECKPOINTER = InMemorySaver()
_AGENT_CACHE: dict[tuple[str, str], Any] = {}


def get_specialist_agent(agent_label: str, instructions: str):
    """Create and cache a specialist deep agent."""
    cache_key = (agent_label, instructions)
    cached_agent = _AGENT_CACHE.get(cache_key)
    if cached_agent is not None:
        return cached_agent

    agent = create_deep_agent(
        model=openai_model,
        tools=TOOLS,
        system_prompt=instructions,
        response_format=SpecialistReport,
        checkpointer=CHECKPOINTER,
    )
    _AGENT_CACHE[cache_key] = agent
    return agent


def get_synthesis_agent():
    """Create and cache the synthesis deep agent."""
    cache_key = ("Synthesis Agent", SYNTHESIS_INSTRUCTIONS)
    cached_agent = _AGENT_CACHE.get(cache_key)
    if cached_agent is not None:
        return cached_agent

    agent = create_deep_agent(
        model=openai_model,
        tools=TOOLS,
        system_prompt=SYNTHESIS_INSTRUCTIONS,
        response_format=StockResearchReport,
        checkpointer=CHECKPOINTER,
    )
    _AGENT_CACHE[cache_key] = agent
    return agent


# -----------------------------------------------------------------------------
# Specialist execution helpers
# -----------------------------------------------------------------------------
async def run_specialist_agent(
    agent_name: str,
    instructions: str,
    user_query: str,
    thread_id: str,
) -> SpecialistReport:
    """Run one specialist agent and capture visible logs for its panel."""
    add_agent_log(agent_name, "request", user_query)
    token = CURRENT_AGENT_NAME.set(agent_name)
    try:
        agent = get_specialist_agent(agent_name, instructions)
        final_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        structured = final_result.get("structured_response")
        if isinstance(structured, SpecialistReport):
            report = structured
        elif isinstance(structured, dict):
            report = SpecialistReport.model_validate(structured)
        else:
            raise ValueError(
                f"{agent_name} did not return a structured SpecialistReport."
            )
        add_agent_log(agent_name, "result", report.model_dump())
        return report
    finally:
        CURRENT_AGENT_NAME.reset(token)


async def run_synthesis_agent(
    user_query: str,
    specialist_reports: list[SpecialistReport],
    thread_id: str,
) -> StockResearchReport:
    """Combine specialist outputs into one final structured report."""
    agent_name = "Synthesis Agent"
    synthesis_payload = {
        "original_user_request": user_query,
        "specialist_reports": [report.model_dump() for report in specialist_reports],
        "instruction": (
            "Combine the specialist reports into one balanced final report. "
            "Preserve disagreements when they matter, and make the final answer educational."
        ),
    }
    add_agent_log(agent_name, "request", synthesis_payload)
    token = CURRENT_AGENT_NAME.set(agent_name)
    try:
        agent = get_synthesis_agent()
        final_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": str(synthesis_payload)}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        structured = final_result.get("structured_response")
        if isinstance(structured, StockResearchReport):
            report = structured
        elif isinstance(structured, dict):
            report = StockResearchReport.model_validate(structured)
        else:
            raise ValueError(
                "Synthesis Agent did not return a structured StockResearchReport."
            )
        add_agent_log(agent_name, "result", report.model_dump())
        return report
    finally:
        CURRENT_AGENT_NAME.reset(token)


async def run_full_workflow(user_query: str) -> StockResearchReport:
    """Run three specialist agents in parallel, then synthesize their results."""
    base_thread = f"stock-research-{uuid4()}"

    specialist_tasks = [
        run_specialist_agent(
            agent_name="Fundamental Agent",
            instructions=FUNDAMENTAL_INSTRUCTIONS,
            user_query=user_query,
            thread_id=f"{base_thread}-fundamental",
        ),
        run_specialist_agent(
            agent_name="Technical Agent",
            instructions=TECHNICAL_INSTRUCTIONS,
            user_query=user_query,
            thread_id=f"{base_thread}-technical",
        ),
        run_specialist_agent(
            agent_name="Risk Agent",
            instructions=RISK_INSTRUCTIONS,
            user_query=user_query,
            thread_id=f"{base_thread}-risk",
        ),
    ]
    specialist_reports = await asyncio.gather(*specialist_tasks)
    final_report = await run_synthesis_agent(
        user_query=user_query,
        specialist_reports=specialist_reports,
        thread_id=f"{base_thread}-synthesis",
    )
    return final_report

# -----------------------------------------------------------------------------
# Streamlit rendering helpers
# -----------------------------------------------------------------------------

def render_events_into_container(container, agent_name: str, events: list[dict[str, Any]]) -> None:
    """Render a list of agent events into the given Streamlit container."""
    with container.container():
        st.markdown(f"### {agent_name}")
        if not events:
            st.caption("No activity yet.")
            return

        for event in events:
            kind = event["kind"]
            payload = event["payload"]
            if kind == "request":
                with st.expander("Request received", expanded=False):
                    if isinstance(payload, dict):
                        st.json(payload)
                    else:
                        st.text(str(payload))
            elif kind == "tool_call":
                with st.expander("Tool call", expanded=False):
                    st.json(payload)
            elif kind == "tool_response":
                with st.expander("Tool response", expanded=False):
                    st.json(payload)
            elif kind == "result":
                with st.expander("Agent result", expanded=False):
                    st.json(payload)
            else:
                with st.expander(kind, expanded=False):
                    if isinstance(payload, dict):
                        st.json(payload)
                    else:
                        st.text(str(payload))


def render_live_agent_panels(panel_map: dict[str, Any]) -> None:
    """Refresh the visible agent panels from the current log snapshot."""
    snapshot = get_agent_log_snapshot()
    for agent_name, container in panel_map.items():
        render_events_into_container(container, agent_name, snapshot.get(agent_name, []))


def render_live_flow_into_container(container) -> None:
    """Render the current cross-agent flow into a placeholder container."""
    snapshot = get_agent_log_snapshot()
    with container.container():
        st.markdown("### Live Agent / Tool Flow")
        has_events = False
        for agent_name in AGENT_NAMES:
            events = snapshot.get(agent_name, [])
            if not events:
                continue
            has_events = True
            with st.expander(f"{agent_name} flow", expanded=False):
                for idx, event in enumerate(events, start=1):
                    kind = event.get("kind", "event")
                    payload = event.get("payload")
                    st.text(f"{idx}. {kind.replace('_', ' ').title()}")
                    if isinstance(payload, dict):
                        st.json(payload)
                    else:
                        st.text(str(payload))
        if not has_events:
            st.caption("No activity yet.")


def render_main_chat() -> None:
    """Render the main chat UI with conversation history and final report."""
    st.subheader("Main Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.json(message["content"])
            else:
                st.text(message["content"])

    if st.session_state.last_report is not None:
        st.markdown("### Final Structured Response")
        st.json(st.session_state.last_report)

    if st.session_state.last_elapsed_seconds is not None:
        st.markdown(
            f"**Time taken:** {st.session_state.last_elapsed_seconds:.2f} seconds"
        )

    if st.session_state.agent_logs:
        st.markdown("### Agent / Tool Flow")
        for agent_name in AGENT_NAMES:
            events = st.session_state.agent_logs.get(agent_name, [])
            if not events:
                continue
            with st.expander(f"{agent_name} flow", expanded=False):
                for idx, event in enumerate(events, start=1):
                    kind = event.get("kind", "event")
                    payload = event.get("payload")
                    st.text(f"{idx}. {kind.replace('_', ' ').title()}")
                    if isinstance(payload, dict):
                        st.json(payload)
                    else:
                        st.text(str(payload))


def render_agent_panel(agent_name: str) -> None:
    """Render one specialist or synthesis panel from the latest stored logs."""
    events = st.session_state.agent_logs.get(agent_name, [])
    if not events:
        events = get_agent_log_snapshot().get(agent_name, [])
    render_events_into_container(st, agent_name, events)


# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------
def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(
        page_title="Pedagogical Deep Agent Stock Research",
        layout="wide",
    )
    init_session_state()

    st.title("Pedagogical Deep Agent Stock Research App")
    st.caption(
        "Main chat on the left. Parallel visible agent panels on the right. "
        "Each panel shows the request, tool calls, tool responses, and final agent result."
    )

    left_col, right_col = st.columns([1.2, 1.8], gap="large")

    with right_col:
        agent_cols_top = st.columns(2, gap="medium")
        agent_cols_bottom = st.columns(2, gap="medium")

        with agent_cols_top[0]:
            fundamental_panel = st.empty()
            render_events_into_container(
                fundamental_panel,
                "Fundamental Agent",
                st.session_state.agent_logs.get("Fundamental Agent", []),
            )
        with agent_cols_top[1]:
            technical_panel = st.empty()
            render_events_into_container(
                technical_panel,
                "Technical Agent",
                st.session_state.agent_logs.get("Technical Agent", []),
            )
        with agent_cols_bottom[0]:
            risk_panel = st.empty()
            render_events_into_container(
                risk_panel,
                "Risk Agent",
                st.session_state.agent_logs.get("Risk Agent", []),
            )
        with agent_cols_bottom[1]:
            synthesis_panel = st.empty()
            render_events_into_container(
                synthesis_panel,
                "Synthesis Agent",
                st.session_state.agent_logs.get("Synthesis Agent", []),
            )

    panel_map = {
        "Fundamental Agent": fundamental_panel,
        "Technical Agent": technical_panel,
        "Risk Agent": risk_panel,
        "Synthesis Agent": synthesis_panel,
    }

    with left_col:
        render_main_chat()
        live_flow_panel = st.empty()
        user_query = st.chat_input(
            "Ask for a stock analysis, for example: Quick analysis on Apple Inc. (AAPL) - buy, sell, or hold"
        )

        if user_query:
            st.session_state.last_query = user_query
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_query,
                }
            )
            reset_agent_logs()

            result_holder: dict[str, Any] = {}
            worker = threading.Thread(
                target=run_workflow_in_background,
                args=(user_query, result_holder),
                daemon=True,
            )

            start_time = time.perf_counter()
            worker.start()
            with st.spinner(
                "Running specialist agents in parallel and synthesizing the final report...",
                show_time=True,
            ):
                while not result_holder.get("done", False):
                    render_live_agent_panels(panel_map)
                    render_live_flow_into_container(live_flow_panel)
                    time.sleep(0.2)
                worker.join()
                render_live_agent_panels(panel_map)
                render_live_flow_into_container(live_flow_panel)
            elapsed_seconds = time.perf_counter() - start_time

            if "error" in result_holder:
                raise result_holder["error"]

            final_report = result_holder["report"]
            st.session_state.last_elapsed_seconds = elapsed_seconds
            st.session_state.agent_logs = get_agent_log_snapshot()
            final_report_dict = final_report.model_dump()
            
            st.session_state.last_report = final_report_dict
            flow_counts = []
            for agent_name in AGENT_NAMES:
                event_count = len(st.session_state.agent_logs.get(agent_name, []))
                flow_counts.append(f"- {agent_name}: {event_count} events")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"**Recommendation:** {final_report.recommendation}\n\n"
                        f"**Ticker:** {final_report.ticker}\n\n"
                        f"**Time horizon:** {final_report.time_horizon}\n\n"
                        f"**Fundamental summary:** {final_report.fundamental_summary}\n\n"
                        f"**Technical summary:** {final_report.technical_summary}\n\n"
                        f"**Risk summary:** {final_report.risk_summary}\n\n"
                        f"**Time taken:** {elapsed_seconds:.2f} seconds\n\n"
                        f"**Visible flow captured:**\n" + "\n".join(flow_counts)
                    ),
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
