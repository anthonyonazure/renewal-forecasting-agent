"""Linear graph: load → score → summarize → pdf → END."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from renewal.nodes import build_pdf, load_inputs, score_all, summarize
from renewal.state import RState


def build_graph():
    g: StateGraph = StateGraph(RState)
    g.add_node("load_inputs", load_inputs)
    g.add_node("score_all", score_all)
    g.add_node("summarize", summarize)
    g.add_node("build_pdf", build_pdf)
    g.add_edge(START, "load_inputs")
    g.add_edge("load_inputs", "score_all")
    g.add_edge("score_all", "summarize")
    g.add_edge("summarize", "build_pdf")
    g.add_edge("build_pdf", END)
    return g
