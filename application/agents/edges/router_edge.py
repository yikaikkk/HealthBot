

def _ensure_router(router_obj: Any, *, fallback_question: str = "") -> Router:
    """将任意 router 结构转换为 Router 模型，保持字段访问兼容。"""
    if isinstance(router_obj, Router):
        return router_obj
    if isinstance(router_obj, dict):
        try:
            return Router.model_validate(router_obj)
        except Exception:
            pass
    return Router(type="kb-query", logic="missing router", question=fallback_question)


# 路由边，用于根据路由类型将输入状态路由到适当的下一个节点。
def router_edge(state: AgentState) -> Literal["respond_to_general_query", "get_additional_info", "create_research_plan", "create_image_query", "create_file_query", "create_kb_query"]:
    """Route the input state to the appropriate next node based on the router type.

    Args:
        state (AgentState): The input state containing the messages and router.

    Returns:
        str: The next node to route to.
    """
    router= _ensure_router(getattr(state, "router", None), fallback_question=state.messages[-1].content if state.messages else "")
    state.router = router
    _type = router.type or "kb-query"

    if _type == "general-query":
        return "respond_to_general_query"
    elif _type == "additional-query":
        return "get_additional_info"
    elif _type == "kb-query":
        return "create_kb_query"
    elif _type == "graphrag-query":
        return "create_research_plan"
    elif _type == "image-query":
        return "create_image_query"
    elif _type == "file-query":
        return "create_file_query"
    elif _type == "text2sql-query":
        return "create_kb_query"
    else:
       raise ValueError(f"Invalid router type: {_type}")
    
