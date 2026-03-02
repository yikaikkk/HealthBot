
from tkinter import END
from langchain.chat_models import ChatZhipuAI
from langchain.config import RunnableConfig
from HealthBot.application.prompts.base_prompts import BASE_PROMPT
from HealthBot.application.config import settings
from HealthBot.application.entitys.agent_state import AgentState
from HealthBot.application.entitys.agent_state import Router


async def analyze_node(
    state: AgentState, *, config: RunnableConfig
) -> dict:
    """
    分析节点，用于分析输入数据并返回分析结果。

    :param input_data: 输入数据，包含需要分析的信息。
    :return: 分析结果，包含对输入数据的分析信息。
    """

    if not settings.ZHIPU_API_KEY:
        raise RuntimeError("ZHIPU_API_KEY is not set")
    

    # 创建模型实例
    zhipuModel=ChatZhipuAI(
        model_name=settings.ZHIPU_MODEL_NAME,
        api_key=settings.ZHIPU_API_KEY,
    )

    # 构建模型输入
    message=[
        {"role":"system","content":BASE_PROMPT},
    ]+state.messages

    logger.info(f"plan next node")
    logger.info(f"message: {message}")

    # 校验下一节点
    allowed_types: set[str] = {
        "general-query",
        "additional-query",
        "kb-query",
        "graphrag-query",
        "image-query",
        "file-query",
        "text2sql-query",
    }

    # 调用模型
    try:
        raw_response = await zhipuModel.with_structured_output(Router).ainvoke(message)
    except Exception as e:
        logger.error(f"Error invoking model: {e}")
        return {"error": str(e)}
    
    response = raw_response if isinstance(raw_response, Router) else Router.model_validate(raw_response)
    router_type = response.type
    logic = response.logic or ""

    if router_type not in allowed_types:
        logger.error(f"Invalid router type: {router_type}")
        # TODO 添加兜底策略
        router_type = END
        return {"router": Router(
            type=router_type,
            logic=logic,
            question=state.question,
            decision=response.decision,
            confidence=response.confidence,
            reasoning=response.reasoning,
        )}

    final_router=Router(
        type=router_type,
        logic=logic,
        question=state.question,
        decision=response.decision,
        confidence=response.confidence,
        reasoning=response.reasoning,
    )

    logger.info(f"final_router: {final_router}")
    return {"router": final_router}







    
