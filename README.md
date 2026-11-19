# Crypto Strategy & Backtest AI Agent with LangGraph

A local cryptocurrency trading strategy and backtesting AI agent built with **LangGraph**, **Gradio**, and powered by **vLLM** running `cyankiwi/Qwen3.5-9B-AWQ-4bit`.

## Prerequisites

1. **Python 3.10+** installed.
2. **CUDA-capable GPU** with **vLLM** installed (`pip install vllm`).

## Setup

1. **Create a virtual environment and install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

2. **Configure environment variables:**

```bash
cp .env.example .env

```

Ensure your `.env` matches your vLLM endpoint:

```env
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
VLLM_MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit

```

## Running the Application

1. **Start the vLLM server with tool-calling enabled:**

```bash
vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --port 8000

```

_(Note: If you are on an older version of vLLM that predates `qwen3_xml`, use `--tool-call-parser hermes`.)_ 2. **Launch the Gradio UI:**

```bash
python app.py
```

Open `http://127.0.0.1:8880` in your browser to configure crypto pairs, set buy/sell signals, and preview your strategy dashboard.
