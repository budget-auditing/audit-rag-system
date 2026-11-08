"""LLM-based fact triple extraction for MSG-RAG instance nodes.

Produces (entity, relation, entity) triples along with entity and relation
descriptions. The graph builder falls back to regex numeric extraction when
the LLM is unavailable.
"""

from typing import Any, Dict, List, Optional

from app.llm.client import chat_complete, extract_json


EXTRACTION_PROMPT_TEMPLATE = """你是一位审计文档信息抽取专家。请从下面的审计文本片段中提取事实三元组。

要求：
1. 实体（entity）可以是项目、科目、资金、指标、部门、时间、金额等语义单元。
2. 关系（relation）描述两个实体之间的归属、关联或说明关系。
3. 对于包含数值的事实，请同时提取 value 和 unit。
4. 每个实体请给出一句简短描述；每个关系请给出一句简短描述。
5. 每个三元组必须附带 support_span（支持该三元组的原始文本片段，50字以内）。
6. 如果文本中没有明确事实，返回空列表，不要编造。
7. 仅输出 JSON，不要任何解释。

输出格式示例：
[
  {
    "subject": "现代农业发展专项资金",
    "subject_desc": "2024年省级财政安排的专项资金",
    "predicate": "安排金额",
    "predicate_desc": "资金总额安排",
    "object": "",
    "object_desc": "",
    "value": "15.00",
    "unit": "亿元",
    "support_span": "2024年省级财政安排现代农业发展专项资金15.00亿元"
  },
  {
    "subject": "现代农业发展专项资金",
    "subject_desc": "用于支持现代农业发展的专项资金",
    "predicate": "支持方向",
    "predicate_desc": "资金具体用途",
    "object": "现代农业产业园建设",
    "object_desc": "产业园建设方向",
    "value": "",
    "unit": "",
    "support_span": "支持现代农业产业园建设，安排资金6.00亿元"
  }
]

文本片段：
$TEXT
"""


def _normalize_triple(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize LLM triple into the internal fact-node shape."""
    return {
        "subject": str(raw.get("subject") or "").strip(),
        "subject_desc": str(raw.get("subject_desc") or "").strip(),
        "predicate": str(raw.get("predicate") or "").strip(),
        "predicate_desc": str(raw.get("predicate_desc") or "").strip(),
        "object": str(raw.get("object") or "").strip(),
        "object_desc": str(raw.get("object_desc") or "").strip(),
        "value": str(raw.get("value") or "").strip(),
        "unit": str(raw.get("unit") or "").strip(),
        "support_span": str(raw.get("support_span") or "").strip(),
    }


def extract_triples_from_text(text: str, context: Optional[str] = None, *, force: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Extract fact triples from a text chunk using the configured LLM.

    Args:
        text: The chunk text to extract from.
        context: Optional surrounding context (e.g., previous chunk sentences).
        force: If True, always try LLM; if False, always skip LLM. If None,
            respect settings.enable_llm_extraction.

    Returns:
        List of normalized triple dicts. Empty list if extraction fails or LLM
        returns no triples.
    """
    from app.config import settings

    if force is False:
        return []
    if force is None and not settings.enable_llm_extraction:
        return []

    if not text or not text.strip():
        return []

    input_text = text
    if context:
        input_text = f"【上下文补充】\n{context}\n\n【当前文本】\n{text}"

    prompt = EXTRACTION_PROMPT_TEMPLATE.replace("$TEXT", input_text)
    call_kwargs: Dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": settings.extract_max_tokens,
    }
    if settings.extract_reasoning_effort:
        # Passed at request-body top-level; disables the thinking chain of
        # reasoning models, which otherwise dominates latency.
        call_kwargs["extra_body"] = {"reasoning_effort": settings.extract_reasoning_effort}
    try:
        raw = chat_complete(
            [
                {
                    "role": "system",
                    "content": "你是信息抽取引擎，只能直接输出 JSON 数组，禁止输出任何思考过程、分析或解释文字。",
                },
                {"role": "user", "content": prompt},
            ],
            **call_kwargs,
        )
    except Exception:
        # If LLM unavailable, return empty; caller may fall back to regex.
        return []

    data = extract_json(raw)
    if not isinstance(data, list):
        return []

    triples: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        triple = _normalize_triple(item)
        # Skip empty subject+object+value triples.
        if not triple["subject"] and not triple["object"] and not triple["value"]:
            continue
        triples.append(triple)
    return triples
