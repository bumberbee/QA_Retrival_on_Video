"""Tools callable by the agent, without HTTP wiring or eager model imports."""

from .base import FunctionTool, Tool, ToolContext, ToolRegistry, ToolResult, object_schema
from .budget import Budget, BudgetLimits


def default_registry() -> ToolRegistry:
    from .calc import calc
    from .rerank_tool import RERANK
    from .search_tools import SEARCH_HYBRID, SEARCH_OCR, SEARCH_SPEECH
    from .slide_tools import GET_SLIDE, GET_WINDOW, NEIGHBORS
    from .verify_tool import VERIFY_CLAIMS

    def calculate(args: dict, ctx: ToolContext) -> dict:
        value = calc(args["expr"])
        if not value["ok"]:
            raise ValueError(value["error"])
        return {**value, "evidence": []}

    registry = ToolRegistry()
    for tool in (SEARCH_HYBRID, SEARCH_OCR, SEARCH_SPEECH, RERANK, GET_SLIDE, GET_WINDOW, NEIGHBORS,
                 FunctionTool("calc", "Evaluate bounded arithmetic exactly using the existing calculator.",
                              object_schema({"expr": {"type": "string", "minLength": 1, "maxLength": 2000}}, ["expr"]),
                              calculate), VERIFY_CLAIMS):
        registry.register(tool)
    return registry


__all__ = ["Tool", "ToolContext", "ToolRegistry", "ToolResult", "Budget", "BudgetLimits", "default_registry"]
