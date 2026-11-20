import asyncio
import os
import pandas as pd
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv(override=True)

model_client = OpenAIChatCompletionClient(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    timeout=60,
    max_retries=3,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": "unknown",
        "structured_output": True,
    },
)

# ---------------- TOOLS ----------------
def track_order(order_id: str) -> str:
    """Find an order by its ID, for example 1005"""
    orders = pd.read_csv("orders.csv", dtype=str).fillna("")
    order_id = order_id.replace("#", "").strip()

    result = orders[orders["order_id"] == order_id]
    if result.empty:
        return "Order not found."
    return str(result.to_dict("records"))


def search_product(keyword: str) -> str:
    """Search products by ONE keyword, for example: kurta, shirt, black, hoodie"""
    products = pd.read_csv("products.csv")

    result = products[
        products["name"].str.contains(keyword, case=False)
        | products["category"].str.contains(keyword, case=False)
    ]
    if result.empty:
        return "No product found."
    return str(result.to_dict("records"))


# ---------------- AGENTS ----------------
order_tracker = AssistantAgent(
    name="order_tracker",
    description="Handles questions about order status, courier and delivery date.",
    model_client=model_client,
    tools=[track_order],
    reflect_on_tool_use=True,
    system_message="""You help customers track orders.
Use the track_order tool. Tell the status, courier and delivery date.
Reply in the customer's language. At the end write TERMINATE.""",
)

product_advisor = AssistantAgent(
    name="product_advisor",
    description="Handles questions about products, price, size and stock.",
    model_client=model_client,
    tools=[search_product],
    reflect_on_tool_use=True,
    system_message="""You help customers find products.
Use the search_product tool. Tell the price and which sizes are in stock
(stock_S, stock_M, stock_L, stock_XL; 0 means out of stock).
Reply in the customer's language. At the end write TERMINATE.""",
)

team = SelectorGroupChat(
    participants=[order_tracker, product_advisor],
    model_client=model_client,
    termination_condition=TextMentionTermination("TERMINATE") | MaxMessageTermination(10),
    selector_prompt="""You route customer messages in an online clothing store.
{roles}

Conversation so far:
{history}

Read the customer's request and select ONE agent from {participants} to answer it.
Order questions go to order_tracker. Product, price, size or stock questions go to product_advisor.
Return only the agent name.""",
)


async def main():
    while True:
        question = input("\nCustomer (type exit to quit): ")
        if question.lower() == "exit":
            break
        await Console(team.run_stream(task=question))
        await team.reset()  # start fresh for the next question

    await model_client.close()


asyncio.run(main())