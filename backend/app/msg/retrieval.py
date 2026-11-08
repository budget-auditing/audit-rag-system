"""Top-down structure retrieval for MSG-RAG.

- For each structure level l, select nodes with Rel(s_i^l, Q) > tau.
- Recursively propagate candidates to children via E_ss edges.
- Perform a second-pass Miss(U^l, Q) recovery for nodes missed at first pass.
"""

import json
import logging
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app.llm.client import chat_complete, extract_json
from app.utils.embed import embed_texts

logger = logging.getLogger(__name__)

RELEVANCE_PROMPT = """你是一位审计文档检索助手。请判断下面的文档标题/条目是否与用户查询相关。

用户查询：{query}
标题路径：{path}

请只输出一个 0 到 1 之间的分数，不要任何其他内容：
- 0：完全无关
- 1：高度相关，直接回答查询

输出格式（仅一个数字）：0.85"""

MISS_PROMPT = """你是一位审计文档检索助手。下面是一些在第一轮检索中被遗漏的文档条目，请判断它们是否仍与用户查询相关。

用户查询：{query}
候选条目：
{candidates}

请输出相关的条目序号列表（从 0 开始），格式为 JSON 数组，例如 [1, 3]。如果没有相关条目，输出 []。"""


def _llm_relevance(query: str, path: List[str]) -> float:
    """Ask LLM to score relevance between query and a structure path."""
    try:
        raw = chat_complete(
            [
                {
                    "role": "user",
                    "content": RELEVANCE_PROMPT.format(query=query, path=" > ".join(path)),
                }
            ],
            temperature=0.0,
            max_tokens=16,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = raw.strip()
        # Try to parse first float in response.
        for token in raw.split():
            token = token.strip(".,;:")
            try:
                return float(token)
            except ValueError:
                continue
        return 0.0
    except Exception:
        return 0.0


def _rule_relevance(query: str, path: List[str]) -> float:
    """Fallback keyword-based relevance used when LLM is unavailable."""
    q_tokens = set(query.lower().split())
    path_text = " > ".join(path).lower()
    if not q_tokens:
        return 0.0
    hits = sum(1 for t in q_tokens if t in path_text)
    return hits / len(q_tokens)


def structure_relevance(query: str, path: List[str]) -> float:
    """Compute Relevance between query and a structure node path."""
    score = _llm_relevance(query, path)
    if score == 0.0:
        score = _rule_relevance(query, path)
    return score


def top_down_structure_search(
    G: nx.DiGraph,
    structure_nodes: Dict[str, Dict[str, Any]],
    query: str,
    tau: float = 0.3,
    top_k_per_level: int = 10,
    use_llm: bool = True,
) -> Set[str]:
    """Top-down structure retrieval.

    Returns a set of relevant structure node ids (all levels).
    """
    if not structure_nodes:
        return set()

    # Group nodes by level.
    levels: Dict[int, List[str]] = {}
    for nid, node in structure_nodes.items():
        lvl = node.get("level", 0)
        levels.setdefault(lvl, []).append(nid)
    max_level = max(levels.keys()) if levels else 0

    selected_by_level: Dict[int, Set[str]] = {lvl: set() for lvl in range(max_level + 1)}
    prev_selected: Set[str] = set()

    for lvl in range(max_level + 1):
        level_nodes = levels[lvl]
        candidates: List[Tuple[str, float]] = []

        # If not level 0, only consider children of previously selected nodes.
        if lvl > 0:
            linked = set()
            for sid in prev_selected:
                linked.update(G.successors(sid))
            level_nodes = [n for n in level_nodes if n in linked]

        for nid in level_nodes:
            path = _json_path(structure_nodes[nid].get("path_json"))
            score = structure_relevance(query, path) if use_llm else _rule_relevance(query, path)
            if score > tau:
                candidates.append((nid, score))

        # Sort and keep top-k per level.
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = {nid for nid, _ in candidates[:top_k_per_level]}

        # Miss recovery: ask LLM to re-check unselected nodes at this level.
        unselected = [nid for nid in level_nodes if nid not in selected]
        recovered = _miss_recovery(query, unselected, structure_nodes) if use_llm else set()
        selected.update(recovered)

        selected_by_level[lvl] = selected
        prev_selected = selected

    return set().union(*selected_by_level.values())


def _miss_recovery(query: str, unselected: List[str], structure_nodes: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Second-pass recovery of missed structure nodes."""
    if not unselected:
        return set()
    # Limit recovery batch size to keep LLM call small.
    batch = unselected[:20]
    candidate_text = "\n".join(
        f"{i}. {_json_path(structure_nodes[nid].get('path_json'))}"
        for i, nid in enumerate(batch)
    )
    try:
        raw = chat_complete(
            [
                {
                    "role": "user",
                    "content": MISS_PROMPT.format(query=query, candidates=candidate_text),
                }
            ],
            temperature=0.0,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        data = extract_json(raw)
        if isinstance(data, list):
            return {batch[int(idx)] for idx in data if isinstance(idx, int) and 0 <= idx < len(batch)}
    except Exception:
        pass
    return set()


def _json_path(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


def bottom_up_instance_aggregation(
    G: nx.DiGraph,
    fact_nodes: Dict[str, Dict[str, Any]],
    selected_structure_ids: Set[str],
    query: str,
    top_k: int = 10,
    tau_q: float = 0.3,
) -> List[Dict[str, Any]]:
    """Bottom-up instance aggregation with community detection.

    1. Collect instance nodes anchored under selected bottom-level structures.
    2. Cluster instances into communities (hierarchical layers).
    3. Filter communities/facts by semantic similarity to query.
    4. Return aggregated context facts sorted by relevance.
    """
    if not fact_nodes:
        return []

    # Gather candidate facts under selected structures.
    candidate_ids: Set[str] = set()
    bottom_structures = {sid for sid in selected_structure_ids if G.out_degree(sid) > 0}
    if not bottom_structures:
        bottom_structures = selected_structure_ids
    for sid in bottom_structures:
        for nbr in G.successors(sid):
            if G.nodes[nbr].get("node_type") == "fact":
                candidate_ids.add(nbr)

    # If no structure-anchored facts, fall back to all facts (safety net).
    if not candidate_ids:
        candidate_ids = set(fact_nodes.keys())

    ids = list(candidate_ids)
    texts = [_fact_text(fact_nodes[fid]) for fid in ids]
    embeddings = embed_texts(texts)
    if embeddings.size == 0:
        return []

    # Normalize embeddings.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms

    # Query vector.
    q_emb = embed_texts([query])
    if q_emb.size:
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12
        q_emb = q_emb / q_norm
        sims = (embeddings @ q_emb.T).flatten()
    else:
        sims = np.zeros(len(ids))

    # Community detection: hierarchical agglomerative clustering on fact embeddings.
    n_clusters = max(2, min(len(ids) // 3, 5))
    if len(ids) >= n_clusters * 2:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = clustering.fit_predict(embeddings)
    else:
        labels = np.zeros(len(ids), dtype=int)

    # Build layers; each cluster is one layer.
    layers: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        layers.setdefault(int(label), []).append(idx)

    # Aggregate relevant facts layer by layer (bottom-up via sorted cluster sizes).
    relevant_facts: List[Dict[str, Any]] = []
    for layer_idx in sorted(layers.keys(), key=lambda k: len(layers[k]), reverse=True):
        for idx in layers[layer_idx]:
            fid = ids[idx]
            score = float(sims[idx])
            if score >= tau_q:
                fact = dict(fact_nodes[fid])
                fact["_score"] = round(score, 4)
                # Collect linked fact info as additional context.
                linked_info = []
                for linked in G.successors(fid):
                    if G.nodes[linked].get("node_type") == "fact" and linked in fact_nodes:
                        linked_info.append(_fact_text(fact_nodes[linked]))
                if linked_info:
                    fact["_linked_context"] = linked_info
                relevant_facts.append(fact)

    relevant_facts.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return relevant_facts[:top_k]


def _fact_text(f: Dict[str, Any]) -> str:
    """Alias kept for compatibility; see _fact_text above."""
    return (
        f"{f.get('subject', '')} {f.get('subject_desc', '')} "
        f"{f.get('predicate', '')} {f.get('predicate_desc', '')} "
        f"{f.get('object', '')} {f.get('object_desc', '')} "
        f"{f.get('value', '')} {f.get('unit', '')} "
        f"{(f.get('support_span', '') or '')[:120]}"
    ).strip()
