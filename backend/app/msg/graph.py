"""MSG-RAG multi-layer semantic graph for fact_numeric / fact-based QA.

Multi-layer semantic graph with bidirectional retrieval.

Core data structures:
  G = (V_s, V_f, R), R = E_ss ∪ E_sf ∪ E_ff
  V_s : structure nodes from document hierarchy
  V_f : instance (fact) nodes extracted as triples
  E_ss: parent-child structure edges (parent_id in structure_nodes table)
  E_sf: anchor edges from structure node to instance node
  E_ff: local relation edges between instance nodes (fact_edges table)

Storage strategy (MVP):
  - SQLite as source of truth:
      structure_nodes carries E_ss via parent_id
      fact_nodes carries E_sf via anchored_section_id
      fact_edges carries E_ff
  - NetworkX is rebuilt in-memory from DB at startup.
  - FAISS holds instance-node vectors for bottom-up similarity retrieval.
"""

import json
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx
import faiss
from sqlalchemy.orm import Session

from app.utils.embed import embed_texts
from app.msg.extraction import extract_triples_from_text
from app.msg.retrieval import top_down_structure_search, bottom_up_instance_aggregation
from app.llm.client import chat_complete
from app.models import StructureNode as StructureNodeModel, FactNode as FactNodeModel, FactEdge as FactEdgeModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_RE_FACT_NUMBER = re.compile(
    r"(?P<value>\d[\d,\.']*)\s*(?P<unit>[万亿千百十个元%€$]+|万元|亿元|元|万亿元|%|‰|个|件|人|户|平方米|公顷|万亩|吨|万吨|元/人|万人|千万件)"
)


def _hash(*parts) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _safe_json_load(raw: Any) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _strip_reasoning(text: str) -> str:
    """Remove leaked chain-of-thought so only the final Chinese answer remains."""
    if not text:
        return ""
    s = text.strip()
    # Drop <think>...</think> blocks if present.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()
    # Qwen3 sometimes emits English CoT prefixed like "Here's a thinking process:".
    # Cut everything before the last explicit answer marker if one exists.
    for marker in ("答案：", "答案:", "回答：", "回答:"):
        idx = s.rfind(marker)
        if idx != -1:
            s = s[idx + len(marker):].strip()
            break
    return s.strip()


class MSGGraph:
    """In-memory MSG-RAG multi-layer semantic graph."""

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self.structure_nodes: Dict[str, Dict[str, Any]] = {}
        self.fact_nodes: Dict[str, Dict[str, Any]] = {}
        self.fact_embeddings: Dict[str, np.ndarray] = {}
        self.chunk_map: Dict[str, Any] = {}
        self._fact_index: Optional[faiss.IndexFlatIP] = None
        self._embed_dim: Optional[int] = None
        self._fact_ids: List[str] = []
        self.use_llm_structure_search: bool = True
        self.use_llm_answer: bool = True

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, chunks: List[Any], progress=None) -> Dict[str, int]:
        """Build graph from ORM StandardChunk objects.

        progress: optional callable(current, total, phase) for build progress;
        the per-chunk LLM fact extraction is the slow phase worth reporting.
        """
        self.G.clear()
        self.structure_nodes.clear()
        self.fact_nodes.clear()
        self.fact_embeddings.clear()
        self.chunk_map.clear()
        self._fact_index = None
        self._fact_ids = []

        # 1. Global structure tree: merge structure nodes across chunks by path.
        for ch in chunks:
            path = _safe_json_load(ch.path_json) or []
            self._ensure_structure_path(ch.doc_id, path)
            self.chunk_map[ch.chunk_id] = ch

        # 2. fact nodes from chunks; anchor to deepest structure node.
        #    Fact extraction is the slow phase (one LLM call per chunk), so run
        #    it concurrently and report progress as calls complete.
        from app.config import settings

        all_records: List[Dict[str, Any]] = []
        total = len(chunks)
        extracted: List[List[Dict[str, Any]]] = [[] for _ in range(total)]
        workers = max(1, settings.msg_extract_workers)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(self._extract_facts_from_chunk, ch): i
                for i, ch in enumerate(chunks)
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    extracted[i] = fut.result()
                except Exception:
                    extracted[i] = []
                completed += 1
                if progress is not None:
                    progress(completed, total, "事实抽取")

        for idx, ch in enumerate(chunks):
            facts = extracted[idx]
            path = _safe_json_load(ch.path_json) or []
            anchor_id = self._deepest_structure_node_id(ch.doc_id, path)
            for f in facts:
                fid = _hash(
                    ch.chunk_id,
                    f.get("subject", ""),
                    f.get("predicate", ""),
                    f.get("object", ""),
                    f.get("value", ""),
                )
                f = dict(f)
                f["fact_id"] = fid
                f["doc_id"] = ch.doc_id
                f["chunk_id"] = ch.chunk_id
                f["source_location_json"] = ch.source_location_json or json.dumps({}, ensure_ascii=False)
                f["anchored_section_id"] = anchor_id
                all_records.append(f)
                self.fact_nodes[fid] = f

        # 3. add fact nodes to graph and E_sf edges
        for f in all_records:
            self.G.add_node(f["fact_id"], **f, node_type="fact")
            if f.get("anchored_section_id"):
                self.G.add_edge(f["anchored_section_id"], f["fact_id"], edge_type="E_sf")

        # 4. E_ss edges via parent_id (already stored in structure_nodes)
        for nid, node in self.structure_nodes.items():
            pid = node.get("parent_id")
            if pid and pid in self.structure_nodes:
                self.G.add_edge(pid, nid, edge_type="E_ss")

        # 5. E_ff edges: connect fact nodes sharing the same anchored section
        facts_by_section: Dict[str, List[Dict[str, Any]]] = {}
        for f in all_records:
            sid = f.get("anchored_section_id") or ""
            facts_by_section.setdefault(sid, []).append(f)
        for sid, flist in facts_by_section.items():
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    self.G.add_edge(flist[i]["fact_id"], flist[j]["fact_id"], edge_type="E_ff")

        # 6. embeddings
        if progress is not None:
            progress(total, total, "向量构建")
        self._build_fact_embeddings()

        return {
            "structure_nodes": len(self.structure_nodes),
            "fact_nodes": len(self.fact_nodes),
            "edges": self.G.number_of_edges(),
        }

    def persist(self, db: Session) -> None:
        """Persist the in-memory graph to SQLite as the source of truth."""
        # Clear existing graph rows for simplicity (full rebuild model).
        db.query(FactEdgeModel).delete()
        db.query(FactNodeModel).delete()
        db.query(StructureNodeModel).delete()

        for node in self.structure_nodes.values():
            db.add(
                StructureNodeModel(
                    node_id=node["node_id"],
                    doc_id=node.get("doc_id", ""),
                    title=node.get("title", ""),
                    path_json=node.get("path_json", "[]"),
                    level=node.get("level", 0),
                    parent_id=node.get("parent_id", ""),
                    scope_json=node.get("scope_json", "[]"),
                    local_text=node.get("local_text", ""),
                    summary=node.get("summary", ""),
                    source_location_json=node.get("source_location_json", "{}"),
                )
            )

        for f in self.fact_nodes.values():
            db.add(
                FactNodeModel(
                    fact_id=f["fact_id"],
                    doc_id=f.get("doc_id", ""),
                    chunk_id=f.get("chunk_id", ""),
                    subject=f.get("subject", ""),
                    subject_desc=f.get("subject_desc", ""),
                    predicate=f.get("predicate", ""),
                    predicate_desc=f.get("predicate_desc", ""),
                    object=f.get("object", ""),
                    object_desc=f.get("object_desc", ""),
                    value=f.get("value", ""),
                    unit=f.get("unit", ""),
                    support_span=f.get("support_span", ""),
                    anchored_section_id=f.get("anchored_section_id", ""),
                    source_location_json=f.get("source_location_json", "{}"),
                )
            )

        seen_edges = set()
        for u, v, data in self.G.edges(data=True):
            if data.get("edge_type") != "E_ff":
                continue
            edge_key = (u, v) if u < v else (v, u)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            db.add(
                FactEdgeModel(
                    edge_id=_hash(u, v),
                    source_fact_id=edge_key[0],
                    target_fact_id=edge_key[1],
                    edge_type="E_ff",
                )
            )

        db.commit()

    def load(self, db: Session) -> None:
        """Rebuild the in-memory graph from SQLite."""
        self.G.clear()
        self.structure_nodes.clear()
        self.fact_nodes.clear()
        self.fact_embeddings.clear()
        self.chunk_map.clear()
        self._fact_index = None
        self._fact_ids = []

        for row in db.query(StructureNodeModel).all():
            node = {
                "node_id": row.node_id,
                "doc_id": row.doc_id,
                "title": row.title,
                "path_json": row.path_json,
                "level": row.level,
                "parent_id": row.parent_id,
                "scope_json": row.scope_json,
                "local_text": row.local_text,
                "summary": row.summary,
                "source_location_json": row.source_location_json,
            }
            self.structure_nodes[row.node_id] = node
            self.G.add_node(row.node_id, **node, node_type="structure")

        for row in db.query(FactNodeModel).all():
            f = {
                "fact_id": row.fact_id,
                "doc_id": row.doc_id,
                "chunk_id": row.chunk_id,
                "subject": row.subject,
                "subject_desc": row.subject_desc,
                "predicate": row.predicate,
                "predicate_desc": row.predicate_desc,
                "object": row.object,
                "object_desc": row.object_desc,
                "value": row.value,
                "unit": row.unit,
                "support_span": row.support_span,
                "anchored_section_id": row.anchored_section_id,
                "source_location_json": row.source_location_json,
            }
            self.fact_nodes[row.fact_id] = f
            self.G.add_node(row.fact_id, **f, node_type="fact")

        for node in self.structure_nodes.values():
            pid = node.get("parent_id")
            if pid and pid in self.structure_nodes:
                self.G.add_edge(pid, node["node_id"], edge_type="E_ss")

        for row in db.query(FactNodeModel).all():
            if row.anchored_section_id and row.anchored_section_id in self.structure_nodes:
                self.G.add_edge(row.anchored_section_id, row.fact_id, edge_type="E_sf")

        for row in db.query(FactEdgeModel).all():
            self.G.add_edge(row.source_fact_id, row.target_fact_id, edge_type="E_ff")

        self._build_fact_embeddings()

    def _ensure_structure_path(self, doc_id: str, path: List[str]) -> str:
        """Ensure a path exists in the global structure tree; return deepest node id."""
        parent_id = ""
        for i, title in enumerate(path):
            node_id = _hash(doc_id, " > ".join(path[: i + 1]))
            if node_id not in self.structure_nodes:
                self.structure_nodes[node_id] = {
                    "node_id": node_id,
                    "doc_id": doc_id,
                    "title": title,
                    "path_json": json.dumps(path[: i + 1], ensure_ascii=False),
                    "level": i,
                    "parent_id": parent_id,
                    "scope_json": json.dumps(path, ensure_ascii=False),
                    "local_text": "",
                    "summary": "",
                    "source_location_json": "",
                }
                self.G.add_node(node_id, **self.structure_nodes[node_id], node_type="structure")
            parent_id = node_id
        return _hash(doc_id, " > ".join(path)) if path else ""

    def _deepest_structure_node_id(self, doc_id: str, path: List[str]) -> str:
        if not path:
            return ""
        return _hash(doc_id, " > ".join(path))

    @staticmethod
    def _worth_llm_extraction(text: str) -> bool:
        """A chunk is only worth an LLM call if it carries a number.

        Audit facts are numeric; number-free chunks (pure headings, prose)
        yield nothing from either the LLM or the regex fallback, so skipping
        them avoids a 16s LLM round-trip that returns an empty list.
        """
        t = (text or "").strip()
        return len(t) >= 8 and any(c.isdigit() for c in t)

    def _extract_facts_from_chunk(self, ch: Any) -> List[Dict[str, Any]]:
        text = ch.text or ""
        # Prefer LLM-based triple extraction, but skip the call for
        # chunks that carry no number (nothing extractable anyway).
        if self._worth_llm_extraction(text):
            triples = extract_triples_from_text(text)
            if triples:
                return triples
        # Fallback to regex numeric extraction when LLM is unavailable/skipped.
        path = _safe_json_load(ch.path_json) or []
        subject_hint = path[-1] if path else "未知"
        facts: List[Dict[str, Any]] = []
        for m in _RE_FACT_NUMBER.finditer(text):
            facts.append(
                {
                    "subject": subject_hint,
                    "predicate": "金额/数量",
                    "object": "",
                    "value": m.group("value"),
                    "unit": m.group("unit"),
                    "support_span": text[max(0, m.start() - 20) : m.end() + 20],
                }
            )
        return facts

    def _build_fact_embeddings(self):
        if not self.fact_ids():
            return
        texts = [self._fact_text(f) for f in self.fact_nodes.values()]
        vectors = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embs = embed_texts(batch)
            if embs.size:
                self._embed_dim = embs.shape[1]
                vectors.extend(embs)
        if vectors and self._embed_dim:
            arr = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
            self._fact_index = faiss.IndexFlatIP(self._embed_dim)
            self._fact_index.add(arr)
            self._fact_ids = list(self.fact_nodes.keys())

    def _fact_text(self, f: Dict[str, Any]) -> str:
        return (
            f"{f.get('subject', '')} {f.get('subject_desc', '')} "
            f"{f.get('predicate', '')} {f.get('predicate_desc', '')} "
            f"{f.get('object', '')} {f.get('object_desc', '')} "
            f"{f.get('value', '')} {f.get('unit', '')} "
            f"{(f.get('support_span', '') or '')[:120]}"
        ).strip()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, query_text: str, top_k: int = 5, doc_scope: Optional[List[str]] = None) -> Dict[str, Any]:
        q = query_text.strip()
        allowed_docs = set(doc_scope) if doc_scope else None

        # Top-down structure search with level propagation and miss recovery.
        if allowed_docs:
            scoped_structures = {nid: node for nid, node in self.structure_nodes.items() if node.get("doc_id") in allowed_docs}
            scoped_G = self.G.subgraph(scoped_structures.keys())
        else:
            scoped_structures = self.structure_nodes
            scoped_G = self.G
        selected_structures = list(
            top_down_structure_search(
                scoped_G,
                scoped_structures,
                q,
                tau=0.25,
                top_k_per_level=max(3, top_k),
                use_llm=self.use_llm_structure_search,
            )
        )

        # Bottom-up instance aggregation with community detection.
        selected_structure_set = set(selected_structures)
        top_facts = bottom_up_instance_aggregation(
            self.G,
            self.fact_nodes,
            selected_structure_set,
            q,
            top_k=top_k,
            tau_q=0.2,
        )

        # Also keep a vector-only fallback when structure search yields nothing.
        if not top_facts and self._fact_index is not None:
            qvec = embed_texts([q])
            if qvec.size and self._embed_dim:
                norm = np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12
                qvec = qvec / norm
                scores, idxs = self._fact_index.search(qvec.astype("float32"), min(top_k, len(self._fact_ids)))
                for idx, score in zip(idxs[0], scores[0]):
                    if idx == -1:
                        continue
                    fid = self._fact_ids[int(idx)]
                    f = self.fact_nodes[fid]
                    if allowed_docs and f.get("doc_id") not in allowed_docs:
                        continue
                    f = dict(f)
                    f["_score"] = round(float(score), 4)
                    top_facts.append(f)

        structures = [self.structure_nodes[nid] for nid in selected_structures[:top_k] if nid in self.structure_nodes]

        warnings = []
        if not top_facts:
            warnings.append("未检索到与查询相关的事实节点。")
        if not structures:
            warnings.append("未检索到相关的结构节点。")

        # Generate natural-language answer from retrieved facts if LLM available.
        generated_answer = self._generate_answer(q, top_facts, structures) if self.use_llm_answer else ""

        return {
            "facts": top_facts,
            "structures": structures,
            "generated_answer": generated_answer,
            "warnings": warnings,
        }

    def _generate_answer(self, query: str, facts: List[Dict[str, Any]], structures: List[Dict[str, Any]]) -> str:
        """Call LLM to produce a final answer from aggregated context."""
        context_parts: List[str] = []
        for s in structures:
            path = _safe_json_load(s.get("path_json"))
            context_parts.append("结构路径：" + " > ".join(path) if path else s.get("title", ""))
        for f in facts[:15]:
            line = f"- {f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')} {f.get('value', '')} {f.get('unit', '')}"
            span = f.get("support_span", "")
            if span:
                line += f"（依据：{span[:60]}...）"
            context_parts.append(line)
        context = "\n".join(context_parts)

        prompt = (
            "你是一位审计问答助手。请根据下面检索到的事实片段回答用户问题。"
            "回答必须基于给定事实，不要编造。如果信息不足，请明确说明。\n"
            "要求：直接用中文给出简洁、准确的答案，不要输出任何推理过程、思考步骤或英文内容。\n\n"
            f"用户问题：{query}\n\n"
            f"检索到的事实片段：\n{context}\n\n"
            "答案："
        )
        try:
            raw = chat_complete(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return _strip_reasoning(raw)
        except Exception:
            return ""

    def _structure_relevance(self, query: str, node: Dict[str, Any]) -> float:
        q = query.lower()
        title = node.get("title", "").lower()
        path_text = " ".join(_safe_json_load(node.get("path_json")) or []).lower()
        kw_hits = 0
        for token in set(q.split()):
            if token in title or token in path_text:
                kw_hits += 1
        if not kw_hits:
            return 0.0
        q_tokens = max(1, len(q.split()))
        return round(kw_hits / q_tokens, 3)

    def status(self) -> Dict[str, Any]:
        return {
            "built": bool(self.fact_nodes),
            "count": len(self.fact_nodes),
            "structure_count": len(self.structure_nodes),
            "edges": self.G.number_of_edges(),
        }

    def fact_ids(self) -> List[str]:
        return list(self.fact_nodes.keys())


# global singleton
msg_graph = MSGGraph()
