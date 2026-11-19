import os
from typing import Annotated, Any, Dict, List, Tuple
from typing_extensions import TypedDict
from pydantic import SecretStr

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import AGENT_TOOLS, STRATEGY_CONFIG

load_dotenv()

MODEL_NAME = os.getenv("VLLM_MODEL", "cyankiwi/Qwen3.5-9B-AWQ-4bit")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

SYSTEM_PROMPT = """You are an expert cryptocurrency trading strategy assistant.
Your job is to help the user configure a crypto trading strategy by setting:
1. A cryptocurrency trading pair (e.g., BTC/USDT, ETH/USDT, SOL-USD) and timeframe.
2. Buy (long entry) signals using technical indicators or price rules.
3. Sell (exit) signals using technical indicators or price rules.

Rules:
- Only cryptocurrency pairs are supported. Politely decline stocks, forex, or commodities.
- Always call the appropriate tools whenever the user specifies or modifies a crypto pair, buy rule, or sell rule.
- When the user asks to run a backtest, call `run_backtest`.
- Keep responses concise, clear, and focused on the strategy parameters."""


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def _build_graph():
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=VLLM_BASE_URL,
        api_key=SecretStr(VLLM_API_KEY),
        temperature=0.1,
    )
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    def chatbot_node(state: AgentState) -> Dict[str, Any]:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(AGENT_TOOLS)

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", chatbot_node)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


AGENT_GRAPH = _build_graph()


def reset_strategy_state() -> None:
    """Clear the global strategy configuration when starting a new session."""
    STRATEGY_CONFIG["pair"] = None
    STRATEGY_CONFIG["timeframe"] = "4h"
    STRATEGY_CONFIG["buy_signals"].clear()
    STRATEGY_CONFIG["sell_signals"].clear()


def get_dashboard_data() -> Tuple[str, List[List[Any]]]:
    """Return formatted markdown summary of active rules and portfolio/backtest rows."""
    pair = STRATEGY_CONFIG["pair"] or "Not set"
    timeframe = STRATEGY_CONFIG["timeframe"]
    buys = STRATEGY_CONFIG["buy_signals"]
    sells = STRATEGY_CONFIG["sell_signals"]

    buy_md = (
        "\n".join(
            f"- `{r['indicator']} {r['operator']} {r['value']}` — {r['description']}"
            for r in buys
        )
        if buys
        else "*None configured*"
    )
    sell_md = (
        "\n".join(
            f"- `{r['indicator']} {r['operator']} {r['value']}` — {r['description']}"
            for r in sells
        )
        if sells
        else "*None configured*"
    )

    strategy_md = (
        f"### Active Crypto Strategy\n"
        f"**Pair:** `{pair}` | **Timeframe:** `{timeframe}`\n\n"
        f"**Buy Signals:**\n{buy_md}\n\n"
        f"**Sell Signals:**\n{sell_md}"
    )

    # Portfolio / Backtest summary table rows (populated once backtest engine is wired up)
    portfolio_rows = [
        ["Starting Capital", "$10,000.00 USDT"],
        ["Current Equity", "$10,000.00 USDT"],
        ["Total Return", "0.00%"],
        ["Win Rate", "—"],
        ["Total Trades", "0"],
        ["Status", "Awaiting Backtest"],
    ]

    return strategy_md, portfolio_rows


def run_agent_turn(
    user_text: str,
    agent_messages: List[BaseMessage],
) -> Tuple[str, List[BaseMessage], str, List[List[Any]]]:
    """Execute one conversational turn through the LangGraph agent."""
    input_messages = list(agent_messages) + [HumanMessage(content=user_text)]
    result = AGENT_GRAPH.invoke({"messages": input_messages})

    updated_messages: List[BaseMessage] = result["messages"]
    last_message = updated_messages[-1]
    if isinstance(last_message, AIMessage):
        if isinstance(last_message.content, str):
            reply_text = last_message.content or "Done."
        else:
            # Extract text blocks if the model returns structured content blocks
            reply_text = (
                "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in last_message.content
                )
                or "Done."
            )
    else:
        reply_text = "Done."

    strategy_md, portfolio_rows = get_dashboard_data()
    return reply_text, updated_messages, strategy_md, portfolio_rows
