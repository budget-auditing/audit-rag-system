#!/usr/bin/env python3

from typing import List, Dict, Tuple, Optional, Sequence
import torch
import re
import os
from transformers import AutoTokenizer
from build_index import RAGStore
from c3rag import CCES

class QueryTriplet:
    def __init__(self, time=None, department=None, metric=None):
        self.time, self.department, self.metric = time, department, metric

    def to_query_string(self):
        parts = []
        if self.time:
            parts.append(f"{self.time}年")
        if self.department:
            parts.append(self.department)
        if self.metric:
            parts.append(self.metric)
        return " ".join(parts)

class CARC:
    def __init__(self, rag_store: RAGStore, llm_client):
        self.rag_store = rag_store
        self.llm_client = llm_client

    def decompose_query(self, query: str) -> List[QueryTriplet]:
        years = re.findall(r'20\d{2}', query)
        depts = re.findall(r'[\u4e00-\u9fa5]{2,}(?:厅|局|委|部|处|科|院|会|办|局机关)', query)
        metric_keywords = ['预算', '支出', '收入', '转移支付', '债务', '执行', '绩效', '结转', '拨付', '偿还']
        metrics = [kw for kw in metric_keywords if kw in query]

        years = list(dict.fromkeys(years)) or [None]
        depts = list(dict.fromkeys(depts)) or [None]
        metrics = list(dict.fromkeys(metrics)) or [None]

        triplets: List[QueryTriplet] = []
        for y in years:
            for d in depts:
                for m in metrics:
                    triplets.append(QueryTriplet(y, d, m))

        if not triplets:
            triplets.append(QueryTriplet(metric=query))
        return triplets

    def execute_retrieval(self, triplets: List[QueryTriplet], k=8, doc_scope: Optional[Sequence[str]] = None) -> List[Dict]:
        evidence, seen = [], set()
        # When a doc_scope filter is active, over-fetch so the top-k survivors
        # aren't starved after filtering out off-scope chunks.
        fetch_k = k * 5 if doc_scope else k
        scope = set(doc_scope) if doc_scope else None
        for t in triplets:
            for r in self.rag_store.search(t.to_query_string(), k=fetch_k):
                if scope is not None and not self._in_scope(r, scope):
                    continue
                if r['text'] not in seen:
                    evidence.append(r)
                    seen.add(r['text'])
        return evidence

    @staticmethod
    def _in_scope(record: Dict, scope: set) -> bool:
        doc_id = record.get("doc_id")
        if doc_id and doc_id in scope:
            return True
        src = record.get("source")
        if src and src in scope:
            return True
        return False

    def remove_conflicting(self, evidence: List[Dict], indices: List[int]) -> List[Dict]:
        return [e for i, e in enumerate(evidence) if i not in indices]

class C3RAGPipeline:
    def __init__(self, rag_store, cces_model, tokenizer, llm_client, threshold=0.72, max_iter=4, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.rag_store = rag_store
        self.cces_model = cces_model.to(device)
        self.cces_model.eval()
        for p in self.cces_model.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer
        self.llm_client = llm_client
        self.threshold = threshold
        self.max_iter = max_iter
        self.device = device
        self.carc = CARC(rag_store, llm_client)

    def evaluate_consistency(self, query: str, evidence: List[Dict]) -> float:
        if not evidence:
            return 0.0
        q_tok = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ev_texts = [e.get('span', e.get('text', '')) if isinstance(e, dict) else str(e) for e in evidence]
        ev_tok = self.tokenizer(ev_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            score, _ = self.cces_model(
                q_tok["input_ids"].to(self.device), q_tok["attention_mask"].to(self.device),
                ev_tok["input_ids"].unsqueeze(0).to(self.device), ev_tok["attention_mask"].unsqueeze(0).to(self.device)
            )
        return float(score.item())

    def compute_marginal_contributions(self, query: str, evidence: List[Dict]) -> List[float]:
        if len(evidence) <= 1:
            return [0.0] * len(evidence)
        full_score = self.evaluate_consistency(query, evidence)
        contributions = []
        for i in range(len(evidence)):
            subset = evidence[:i] + evidence[i+1:]
            subset_score = self.evaluate_consistency(query, subset) if subset else 0.0
            contributions.append(subset_score - full_score)
        return contributions

    def _infer_conflict_label(self, query: str, snippet: str) -> str:
        query_years = set(re.findall(r'20\d{2}', query))
        snippet_years = set(re.findall(r'20\d{2}', snippet))
        if query_years and snippet_years and not query_years.intersection(snippet_years):
            return "YEAR_MISMATCH"

        dept_pattern = r'[\u4e00-\u9fa5]{2,}(?:厅|局|委|部|处|科|院|会|办)'
        query_depts = set(re.findall(dept_pattern, query))
        snippet_depts = set(re.findall(dept_pattern, snippet))
        if query_depts and snippet_depts and not query_depts.intersection(snippet_depts):
            return "DEPARTMENT_MISMATCH"

        causal_terms = ['因为', '导致', '因此', '原因', '致使', '以致']
        if any(term in snippet for term in causal_terms) and not any(term in query for term in causal_terms):
            return "CAUSAL_GAP"

        return "UNSPECIFIED_CONFLICT"

    def identify_conflicts(self, query: str, evidence: List[Dict], threshold=0.01) -> List[Dict[str, object]]:
        contributions = self.compute_marginal_contributions(query, evidence)
        conflicts: List[Dict[str, object]] = []
        for idx, delta in enumerate(contributions):
            if delta > threshold:
                text = evidence[idx].get('span') if isinstance(evidence[idx], dict) else str(evidence[idx])
                if isinstance(evidence[idx], dict) and not text:
                    text = evidence[idx].get('text', '')
                label = self._infer_conflict_label(query, text or '')
                conflicts.append({"index": idx, "delta": float(delta), "label": label})
        return conflicts

    def _refine_query(self, query: str, conflicts: Sequence[Dict[str, object]]) -> str:
        labels = {c.get("label") for c in conflicts if c.get("label")}
        if not labels:
            return query + "，补充更精确的审计口径"

        hints: List[str] = []
        if "YEAR_MISMATCH" in labels:
            hints.append("限定相同财政年度和统计口径")
        if "DEPARTMENT_MISMATCH" in labels:
            hints.append("明确部门或单位名称一致")
        if "CAUSAL_GAP" in labels:
            hints.append("补全政策-资金-绩效因果链条")
        if "UNSPECIFIED_CONFLICT" in labels and not hints:
            hints.append("聚焦与问题最相关的数据片段")
        return query + "，" + "，".join(hints)

    def retrieve_and_evaluate(self, query: str, k=8, max_ev=12, doc_scope: Optional[Sequence[str]] = None) -> Tuple[List[Dict], float]:
        triplets = self.carc.decompose_query(query)
        evidence = self.carc.execute_retrieval(triplets, k=k, doc_scope=doc_scope)
        best_score, best_ev = 0.0, evidence.copy()
        last_score = 0.0

        for _ in range(self.max_iter):
            if len(evidence) > max_ev:
                evidence = evidence[:max_ev]
            score = self.evaluate_consistency(query, evidence)
            last_score = score
            if score > best_score:
                best_score, best_ev = score, evidence.copy()
            if score >= self.threshold:
                break

            conflicts = self.identify_conflicts(query, evidence)
            if conflicts:
                indices: List[int] = [int(c["index"]) for c in conflicts][:max(1, len(evidence)//3)]
                evidence = self.carc.remove_conflicting(evidence, indices)
                refined_query = self._refine_query(query, conflicts)
                new_triplets = self.carc.decompose_query(refined_query)
                new_ev = self.carc.execute_retrieval(new_triplets, k=k//2, doc_scope=doc_scope)
                seen = {e.get("text", "") for e in evidence}
                for e in new_ev:
                    if e.get("text", "") not in seen and len(evidence) < max_ev:
                        evidence.append(e)
                        seen.add(e.get("text", ""))

        final_score = best_score
        final_evidence = best_ev
        if last_score > best_score:
            final_score = last_score
            final_evidence = evidence
        return final_evidence, final_score

    def generate_answer(self, query: str, evidence: List[Dict]) -> str:
        ev_parts = []
        for i, e in enumerate(evidence):
            text = e.get('span', e.get('text', '')) if isinstance(e, dict) else str(e)
            ev_parts.append(f"[证据{i+1}] {text}")
        prompt = f"基于以下证据回答问题：\n\n问题: {query}\n\n证据:\n" + "\n\n".join(ev_parts) + "\n\n请基于以上证据给出准确答案。"
        resp = self.llm_client.chat.completions.create(model=os.environ.get("CHAT_MODEL", "gpt-4o-mini"), messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=4096)
        msg = resp.choices[0].message
        content = msg.content or getattr(msg, "reasoning_content", None) or ""
        return content.strip()

    def run(self, query: str, k=8, doc_scope: Optional[Sequence[str]] = None) -> Dict:
        evidence, score = self.retrieve_and_evaluate(query, k=k, doc_scope=doc_scope)
        answer = self.generate_answer(query, evidence)
        return {"query": query, "answer": answer, "consistency_score": score, "evidence_count": len(evidence), "evidence": evidence}

def load_c3rag_pipeline(index_dir, cces_model_path, encoder_name, llm_client, embed_client=None, threshold=0.72, max_iter=4):
    import os
    rag_store = RAGStore(embed_client if embed_client is not None else llm_client)
    rag_store.load(index_dir)
    cces_model = CCES(encoder_name)
    if os.path.exists(cces_model_path):
        cces_model.load_state_dict(torch.load(cces_model_path, map_location="cpu"))
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    return C3RAGPipeline(rag_store, cces_model, tokenizer, llm_client, threshold, max_iter)

if __name__ == "__main__":
    import openai
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    chat_client = openai.OpenAI(base_url=os.environ.get("CHAT_URL", "http://localhost:8000/v1"), api_key=api_key)
    embed_client = openai.OpenAI(base_url=os.environ.get("EMBED_URL", "http://localhost:8081/v1"), api_key=api_key)

    pipeline = load_c3rag_pipeline(
        index_dir="data/index",
        cces_model_path="models/cces_model.pt",
        encoder_name="nghuyong/ernie-3.0-base-zh",
        llm_client=chat_client,
        embed_client=embed_client,
    )

    result = pipeline.run("What is the budget allocation for public health?", k=8)
    print(f"Query: {result['query']}")
    print(f"Consistency Score: {result['consistency_score']:.3f}")
    print(f"Answer:\n{result['answer']}")
