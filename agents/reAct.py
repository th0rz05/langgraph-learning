from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import SystemMessage,BaseMessage,ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]

@tool
def add(a:int,b:int) -> int:
    """This is a addition function that adds two numbers."""
    return a+b

@tool
def subtract(a:int,b:int) -> int:
    """This is a subtraction function that subtracts two numbers."""
    return a-b

@tool
def multiply(a:int,b:int) -> int:
    """This is a multiplication function that multiplies two numbers."""
    return a*b

@tool
def divide(a:int,b:int) -> int:
    """This is a division function that divides two numbers."""
    return a/b

tools = [add, subtract, multiply, divide]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges("our_agent", should_continue, {
    "continue": "tools",
    "end": END
})

graph.add_edge("tools", "our_agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","Add 5 + 12.")]}
print_stream(app.stream(inputs,stream_mode="values"))