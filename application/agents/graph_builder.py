from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, InputState
from HealthBot.application.agents.nodes.analyze_node import analyze_node


# 定义图
checkpointer=MemorySaver()

builder = StateGraph(AgentState,input=InputState)

builder.add_node(analyze_node)

builder.add_edge(START, analyze_node)





graph = builder.compile(checkpointer=checkpointer)
