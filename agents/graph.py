from langgraph.graph import StateGraph, END
from agents.nodes import (
    MeetingState, 
    transcription_node, 
    summary_concept_node, 
    task_milestone_node, 
    mcp_export_node
)

def build_meeting_graph():
    graph = StateGraph(MeetingState)

    # 1. Define Nodes
    graph.add_node("transcriber", transcription_node)
    graph.add_node("summarizer", summary_concept_node)
    graph.add_node("task_planner", task_milestone_node)
    graph.add_node("mcp_exporter", mcp_export_node)

    # 2. Define Execution Edges
    graph.set_entry_point("transcriber")
    graph.add_edge("transcriber", "summarizer")
    graph.add_edge("summarizer", "task_planner")
    graph.add_edge("task_planner", "mcp_exporter")
    graph.add_edge("mcp_exporter", END)

    return graph.compile()

# Instantiated Workflow ready for export
app_graph = build_meeting_graph()