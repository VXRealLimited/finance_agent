import gradio as gr
from agent.agent import get_dashboard_data, reset_strategy_state, run_agent_turn


def reset_session():
    reset_strategy_state()
    strategy_md, portfolio_rows, chart_section_upd, equity_plot_upd = (
        get_dashboard_data()
    )
    return [], [], strategy_md, chart_section_upd, equity_plot_upd, portfolio_rows


with gr.Blocks(title="Crypto Strategy & Backtest Agent") as demo:
    with gr.Row():
        gr.Markdown("## Crypto Strategy & Backtest Agent (`Qwen3.5-9B-AWQ`)")
        reset_btn = gr.Button("New Strategy / Reset", size="sm", scale=0)

    agent_state = gr.State([])
    initial_strategy_md, initial_portfolio_rows, _, _ = get_dashboard_data()

    with gr.Row():
        # Left Column: Chat Interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600)
            chat_input = gr.Textbox(
                placeholder="Describe your crypto pair, buy/sell signals, or ask to run a backtest...",
                show_label=False,
                autofocus=True,
            )

        # Right Column: Live Strategy Rules, Dynamic Equity Curve & Portfolio Summary
        with gr.Column(scale=2):
            strategy_panel = gr.Markdown(value=initial_strategy_md)

            # Hidden by default until backtest data exists
            with gr.Column(visible=False) as chart_section:
                gr.Markdown("### Portfolio Equity Curve")
                equity_chart = gr.LinePlot(
                    x="datetime",
                    y="equity",
                    x_title="Date",
                    y_title="Equity (USDT)",
                    height=240,
                )

            gr.Markdown("### Backtest Portfolio Summary")
            portfolio_table = gr.Dataframe(
                headers=["Metric", "Value"],
                value=initial_portfolio_rows,
                datatype=["str", "str"],
                interactive=False,
            )

    def respond(user_text: str, history: list, agent_messages: list):
        if not user_text or not user_text.strip():
            return history, agent_messages, ""

        history = history + [{"role": "user", "content": user_text}]

        try:
            reply_text, updated_messages = run_agent_turn(
                user_text=user_text,
                agent_messages=agent_messages,
            )
        except Exception as exc:
            history = history + [{"role": "assistant", "content": f"Error: {exc}"}]
            return history, agent_messages, ""

        history = history + [{"role": "assistant", "content": reply_text}]
        return history, updated_messages, ""

    chat_input.submit(
        fn=respond,
        inputs=[chat_input, chatbot, agent_state],
        outputs=[chatbot, agent_state, chat_input],
    ).then(
        fn=get_dashboard_data,
        outputs=[strategy_panel, portfolio_table, chart_section, equity_chart],
    )

    reset_btn.click(
        fn=reset_session,
        outputs=[
            chatbot,
            agent_state,
            strategy_panel,
            chart_section,
            equity_chart,
            portfolio_table,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=8880,
    )
