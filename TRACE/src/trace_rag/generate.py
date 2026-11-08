#! /usr/bin/env python3
"""
TRACE generation with OCI structure, claim verification, and conservative fallback.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib import request

import numpy as np

from .retrieval import embed_texts


CHAT_BASE_URL = os.getenv("CHAT_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
DEFAULT_SERVER_URL = os.getenv(
    "EMBED_SERVER_URL",
    "http://localhost:8081/v1/embeddings",
)

NUM_REGEX = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
YEAR_REGEX = re.compile(r"20\d{2}")
UNIT_REGEX = re.compile(
    r"(?i)(trillion|billion|million|thousand|hundred|percent|basis\s+points|bps"
    r"|万亿元|千亿元|百亿元|十亿元|亿元|千万元|百万元|十万元|万元|元|%|％|‰|万|亿|倍|人次|人|户|个|项|件|次)"
)
CAL_REGEX = re.compile(r"\[CAL=([^\]]+)\]")
YEAR_TAG_REGEX = re.compile(r"\[YEAR=(\d{4})\]")

DEFAULT_MULTI_CITE_DELTA = 0.02
DEFAULT_MAX_CITES = 2
DEFAULT_LAMBDA_ATTR = 0.7
DEFAULT_DELTA_A = 0.2
DEFAULT_DELTA_V = 0.2

CALIBER_ALIASES = {
    "一般公共预算收入": "general_public_budget_income",
    "一般公共预算支出": "general_public_budget_expenditure",
    "一般公共预算": "general_public_budget",
    "政府性基金预算": "government_fund_budget",
    "政府性基金": "government_fund_budget",
    "国有资本经营预算": "state_capital_budget",
    "社会保险基金预算": "social_security_fund_budget",
    "社会保险基金": "social_security_fund_budget",
    # English financial categories
    "operating income": "operating",
    "operating expense": "operating",
    "operating revenue": "operating",
    "net income": "net_income",
    "net revenue": "net_income",
    "net earnings": "net_income",
    "net loss": "net_income",
    "gross profit": "gross_profit",
    "gross margin": "gross_profit",
    "total assets": "balance_sheet",
    "total liabilities": "balance_sheet",
    "total equity": "balance_sheet",
    "total revenue": "net_income",
    "total debt": "debt",
    "cash flow": "cash_flow",
    "interest expense": "interest_tax",
    "interest income": "interest_tax",
    "tax expense": "interest_tax",
    "shareholders equity": "equity",
    "stockholders equity": "equity",
    "long-term debt": "debt",
    "short-term debt": "debt",
    "capital expenditure": "capex",
    "earnings per share": "eps",
}

CALIBER_GROUPS = {
    "general_public_budget": "general_public_budget",
    "general_public_budget_income": "general_public_budget",
    "general_public_budget_expenditure": "general_public_budget",
    "government_fund_budget": "government_fund_budget",
    "state_capital_budget": "state_capital_budget",
    "social_security_fund_budget": "social_security_fund_budget",
    # English financial category groups
    "operating": "income_statement",
    "net_income": "income_statement",
    "gross_profit": "income_statement",
    "eps": "income_statement",
    "balance_sheet": "balance_sheet",
    "equity": "balance_sheet",
    "debt": "balance_sheet",
    "cash_flow": "cash_flow",
    "capex": "cash_flow",
    "interest_tax": "income_statement",
}

_CALIBER_EMBED_CACHE: Dict[str, np.ndarray] = {}
_CALIBER_EMBED_LOCK = threading.Lock()


def _empty_oci() -> Dict:
    return {"Observation": [], "Cause": [], "Impact": []}


def _try_parse_json(text: str) -> Optional[Dict]:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_oci_json(content: str) -> Dict:
    parsed = _try_parse_json(content.strip())
    if parsed is not None:
        return parsed

    code_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in code_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        parsed = _try_parse_json(match.group(1).strip())
        if parsed is not None:
            return parsed

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = _try_parse_json(content[start : end + 1])
        if parsed is not None:
            return parsed

    return _empty_oci()


def _detect_lang(text: str) -> str:
    cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return "zh" if cjk > len(text) * 0.1 else "en"


def _build_chat_prompt(query: str, evidence_text: str, dataset_tag: str, lang: str) -> str:
    if dataset_tag == "finqa":
        from trace_rag.utils.finqa_dsl import format_dsl_prompt_block
        dsl_block = format_dsl_prompt_block()
        return (
            f"{dsl_block}\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Q: {query}\nProgram:"
        )
    if lang == "zh":
        return (
            "你是预算审计专家。请严格依据证据回答问题，输出 OCI 结构（Observation/Cause/Impact）。\n"
            "要求：\n"
            "1) 只使用证据中的信息，不编造数值与口径。\n"
            "2) 输出 JSON 格式："
            "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}。\n"
            "3) 不需要输出引用。\n\n"
            f"问题：{query}\n\n证据：\n{evidence_text}\n"
        )
    return (
        "You are a financial analyst. Answer the question strictly based on the provided evidence. "
        "Output an OCI structure (Observation/Cause/Impact).\n"
        "Requirements:\n"
        "1) Only use information from the evidence. Do not fabricate numbers.\n"
        "2) Output JSON format: "
        "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}.\n"
        "3) Include the final numerical answer explicitly in one of the claims.\n"
        "4) Do not include citations.\n\n"
        f"Question: {query}\n\nEvidence:\n{evidence_text}\n"
    )


def call_chat_api(
    query: str,
    evidence: List[Dict],
    model: str,
    max_retries: int = 3,
    dataset_tag: str = "audit-ociqa",
) -> Tuple[Dict, str]:
    evidence_text = "\n\n".join(
        [f"[{e['evidence_id']}]\n{e['text']}" for e in evidence]
    )
    lang = _detect_lang(query + evidence_text[:500])
    prompt = _build_chat_prompt(query, evidence_text, dataset_tag, lang)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "max_tokens": 2048,
        "presence_penalty": 1.0,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        CHAT_BASE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(1, max_retries + 1):
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
            return parse_oci_json(content), content
        except Exception:
            if attempt == max_retries:
                print(f"[WARN] Chat API failed after {max_retries} retries: {query[:50]}...")
                return _empty_oci(), ""
            time.sleep(1.5 * attempt)


def normalize_caliber(caliber: str | None) -> str | None:
    if not caliber:
        return None
    if caliber in CALIBER_ALIASES.values():
        return caliber
    for key in sorted(CALIBER_ALIASES.keys(), key=len, reverse=True):
        if key in caliber:
            return CALIBER_ALIASES[key]
    return caliber


def _caliber_embed(text: str, server_url: str) -> np.ndarray | None:
    key = (text or "").strip()
    if not key:
        return None
    with _CALIBER_EMBED_LOCK:
        cached = _CALIBER_EMBED_CACHE.get(key)
        if cached is not None:
            return cached
    vec = embed_texts(server_url, [key], batch_size=1)[0]
    with _CALIBER_EMBED_LOCK:
        _CALIBER_EMBED_CACHE[key] = vec
    return vec


def extract_caliber(text: str) -> str | None:
    match = CAL_REGEX.search(text)
    if match:
        return normalize_caliber(match.group(1).strip())
    for key in sorted(CALIBER_ALIASES.keys(), key=len, reverse=True):
        if key in text:
            return CALIBER_ALIASES[key]
    return None


def extract_year(text: str) -> int | None:
    tag = YEAR_TAG_REGEX.search(text)
    if tag:
        try:
            return int(tag.group(1))
        except Exception:
            return None
    years = YEAR_REGEX.findall(text)
    if years:
        try:
            return int(years[0])
        except Exception:
            return None
    return None


def _year_positions(text: str) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    tag_spans: List[Tuple[int, int]] = []
    for match in YEAR_TAG_REGEX.finditer(text):
        try:
            year = int(match.group(1))
        except Exception:
            continue
        positions.append((match.start(), year))
        tag_spans.append(match.span())
    for match in YEAR_REGEX.finditer(text):
        if any(span[0] <= match.start() < span[1] for span in tag_spans):
            continue
        try:
            year = int(match.group(0))
        except Exception:
            continue
        positions.append((match.start(), year))
    positions.sort(key=lambda x: x[0])
    return positions


def _nearest_year(pos: int, positions: List[Tuple[int, int]], year_hint: int | None) -> int | None:
    if not positions:
        return year_hint
    left = [(pos - year_pos, year) for year_pos, year in positions if year_pos <= pos]
    if left:
        left.sort(key=lambda x: x[0])
        return left[0][1]
    right = [(year_pos - pos, year) for year_pos, year in positions if year_pos > pos]
    if right:
        right.sort(key=lambda x: x[0])
        return right[0][1]
    return year_hint


def extract_triples(text: str, caliber_hint: str | None = None, year_hint: int | None = None) -> List[Dict]:
    triples = []
    caliber = normalize_caliber(caliber_hint) or extract_caliber(text)
    year_positions = _year_positions(text)
    for match in NUM_REGEX.finditer(text):
        raw = match.group(0)
        end = match.end()
        unit = None
        if end < len(text) and text[end] in ("%","％"):
            unit = "%"
        elif end < len(text) and text[end] == "‰":
            unit = "‰"
        else:
            unit_match = UNIT_REGEX.search(text[end : end + 10])
            unit = unit_match.group(0) if unit_match else None
        try:
            value = float(raw.replace(",", "").replace("%", ""))
        except Exception:
            continue
        year = _nearest_year(match.start(), year_positions, year_hint)
        triples.append({"value": value, "unit": unit, "year": year, "caliber": caliber})
    return triples


def caliber_similarity(
    a: str | None,
    b: str | None,
    server_url: str = DEFAULT_SERVER_URL,
    use_embed: bool = True,
) -> float:
    a_norm = normalize_caliber(a)
    b_norm = normalize_caliber(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.5
    if a_norm == b_norm:
        return 1.0
    group_a = CALIBER_GROUPS.get(a_norm)
    group_b = CALIBER_GROUPS.get(b_norm)
    group_sim = 0.8 if group_a and group_b and group_a == group_b else 0.0
    if not use_embed:
        return group_sim
    emb_a = _caliber_embed(a_norm, server_url)
    emb_b = _caliber_embed(b_norm, server_url)
    if emb_a is None or emb_b is None:
        return group_sim
    sim = float(np.dot(emb_a, emb_b))
    sim01 = max(0.0, min(1.0, (sim + 1.0) / 2.0))
    return max(group_sim, 0.6 * sim01)


def consistency_score(
    claim: str,
    evidence: Dict,
    server_url: str | None = None,
    claim_triples: List[Dict] | None = None,
) -> float:
    claim_triples = claim_triples if claim_triples is not None else extract_triples(claim)
    evidence_text = (
        evidence.get("text_with_meta")
        or evidence.get("text_for_embedding")
        or evidence.get("text", "")
    )
    evidence_triples = extract_triples(
        evidence_text,
        caliber_hint=evidence.get("caliber"),
        year_hint=evidence.get("year"),
    )
    if not claim_triples or not evidence_triples:
        return 0.0
    best = 0.0
    for c in claim_triples:
        for e in evidence_triples:
            if c["unit"] and e["unit"] and c["unit"] != e["unit"]:
                continue
            if c["year"] and e["year"] and c["year"] != e["year"]:
                continue
            sim_kappa = caliber_similarity(
                c.get("caliber"),
                e.get("caliber"),
                server_url=server_url or DEFAULT_SERVER_URL,
            )
            if sim_kappa <= 0:
                continue
            denom = max(abs(c["value"]), abs(e["value"]), 1.0)
            diff = abs(c["value"] - e["value"]) / denom
            score = math.exp(-diff / 0.1) * sim_kappa
            best = max(best, score)
    return best


def triple_score(
    claim: str,
    evidence: Dict,
    server_url: str | None = None,
    claim_triples: List[Dict] | None = None,
) -> float:
    """Backward-compatible alias for the old attribute consistency name."""
    return consistency_score(
        claim,
        evidence,
        server_url=server_url,
        claim_triples=claim_triples,
    )


def semantic_scores(claims: List[str], evidence: List[Dict], server_url: str) -> np.ndarray:
    if not claims:
        return np.zeros((0, len(evidence)), dtype="float32")
    texts = claims + [
        e.get("text_with_meta") or e.get("text_for_embedding") or e.get("text", "")
        for e in evidence
    ]
    vecs = embed_texts(server_url, texts, batch_size=16)
    claim_vecs = vecs[: len(claims)]
    ev_vecs = vecs[len(claims) :]
    return claim_vecs @ ev_vecs.T


def normalize_oci(oci: Dict) -> Dict:
    if not isinstance(oci, dict):
        return _empty_oci()
    sections = ["Observation", "Cause", "Impact"]
    normalized = {}
    for section in sections:
        items = oci.get(section, [])
        if isinstance(items, dict) or isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            items = []
        out_items = []
        for item in items:
            if isinstance(item, dict):
                claim = item.get("claim")
            else:
                claim = item
            if claim is None:
                continue
            claim_str = str(claim).strip()
            if claim_str:
                out_items.append({"claim": claim_str})
        normalized[section] = out_items
    return normalized


def align_claims(
    oci: Dict,
    evidence: List[Dict],
    server_url: str,
    lambda_attr: float = DEFAULT_LAMBDA_ATTR,
    delta_a: float = DEFAULT_DELTA_A,
    delta_v: float = DEFAULT_DELTA_V,
    use_attribute_consistency: bool = True,
    use_local_suppression: bool = True,
    multi_cite_delta: float = DEFAULT_MULTI_CITE_DELTA,
    max_cites: int = DEFAULT_MAX_CITES,
) -> Dict:
    sections = ["Observation", "Cause", "Impact"]
    oci = normalize_oci(oci)
    prediction_oci = _empty_oci()
    claim_support = {}
    suppressed_claims = []

    is_en = _detect_lang(" ".join(
        c.get("claim", "") for s in ("Observation", "Cause", "Impact")
        for c in (oci.get(s) or []) if isinstance(c, dict)
    )) == "en"

    def gap_message(section: str) -> str:
        if is_en:
            return {"Cause": "No direct evidence for the cause",
                    "Impact": "No direct evidence for the impact"
                    }.get(section, "No direct evidence for the observation")
        return {"Cause": "缺少与原因相关的直接证据",
                "Impact": "缺少与影响相关的直接证据"
                }.get(section, "缺少与现象相关的直接证据")

    def unsupported_claim(section: str, claim: str) -> Dict:
        return {
            "claim": claim,
            "cites": [],
            "status": "suppressed",
            "gap": gap_message(section),
            "fallback": "Insufficient evidence." if is_en else "材料未说明/无法判断。",
        }

    if not evidence:
        for section in sections:
            claims = [c.get("claim", "") for c in oci.get(section, []) if c.get("claim")]
            claim_support[section] = [unsupported_claim(section, claim) for claim in claims]
        return {
            "prediction_O": [],
            "prediction_C": [],
            "prediction_I": [],
            "prediction_oci": prediction_oci,
            "claim_support": claim_support,
            "suppressed_claims": suppressed_claims,
            "abstain_flag": True,
            "abstain_reason": "no_evidence",
            "local_metrics": {
                "total_claims": sum(len(claim_support.get(section, [])) for section in sections),
                "supported_claims": 0,
                "numeric_claims": 0,
                "qac_supported_claims": 0,
                "suppressed_claims": sum(len(claim_support.get(section, [])) for section in sections),
            },
        }

    total_claims = 0
    supported_claims = 0
    numeric_claims = 0
    qac_supported_claims = 0
    for section in sections:
        claims = [c.get("claim", "") for c in oci.get(section, []) if c.get("claim")]
        sem = semantic_scores(claims, evidence, server_url)
        section_support = []
        for idx, claim in enumerate(claims):
            total_claims += 1
            sem_row_raw = sem[idx].tolist() if sem.size else []
            sem_row = [max(0.0, min(1.0, (float(s) + 1.0) / 2.0)) for s in sem_row_raw]
            claim_triples = extract_triples(claim) if use_attribute_consistency else []
            has_triples = bool(claim_triples)
            if has_triples:
                numeric_claims += 1
            cons_scores = (
                [
                    consistency_score(
                        claim,
                        e,
                        server_url=server_url,
                        claim_triples=claim_triples,
                    )
                    for e in evidence
                ]
                if use_attribute_consistency and has_triples
                else []
            )
            if use_attribute_consistency and has_triples:
                align_scores = [
                    lambda_attr * cons + (1 - lambda_attr) * float(sem_row[j])
                    for j, cons in enumerate(cons_scores)
                ]
            else:
                align_scores = [float(sem_row[j]) for j in range(len(evidence))]

            if not align_scores:
                support_info = unsupported_claim(section, claim)
                section_support.append(support_info)
                suppressed_claims.append(dict(support_info, section=section))
                continue

            best_idx = int(np.argmax(align_scores))
            best_align = float(align_scores[best_idx])
            best_sup = float(sem_row[best_idx])
            best_cons = float(cons_scores[best_idx]) if cons_scores else None
            max_cons = max(float(score) for score in cons_scores) if cons_scores else None
            if multi_cite_delta and multi_cite_delta > 0:
                candidate_idxs = [
                    i for i, s in enumerate(align_scores) if s >= best_align - multi_cite_delta
                ]
                candidate_idxs.sort(key=lambda i: align_scores[i], reverse=True)
                candidate_idxs = candidate_idxs[: max(1, max_cites)]
                cites = [
                    evidence[i]["evidence_id"]
                    for i in candidate_idxs
                    if evidence[i].get("evidence_id")
                ]
            else:
                best_evidence_id = evidence[best_idx].get("evidence_id")
                cites = [best_evidence_id] if best_evidence_id else []
            support_info = {
                "claim": claim,
                "cites": cites,
                "best_evidence_id": evidence[best_idx].get("evidence_id"),
                "best_evidence_text": evidence[best_idx].get("text"),
                "s_sup": best_sup,
                "Cons": best_cons,
                "qac_cons": max_cons,
                "Align": best_align,
                "delta_a": delta_a,
                "delta_v": delta_v if has_triples else None,
                "has_numeric_attr": has_triples,
                "qac_pass": (max_cons is not None and max_cons >= delta_v),
                "supported": best_align >= delta_a,
                "suppressed": best_align < delta_a and use_local_suppression,
                "retained_unsupported": best_align < delta_a and not use_local_suppression,
            }
            if support_info["qac_pass"]:
                qac_supported_claims += 1
            if support_info["supported"]:
                supported_claims += 1
                prediction_oci[section].append({"claim": claim, "cites": cites})
            elif not use_local_suppression:
                prediction_oci[section].append({"claim": claim, "cites": cites})
            else:
                support_info.update(unsupported_claim(section, claim))
                suppressed_claims.append(dict(support_info, section=section))
            section_support.append(support_info)
        claim_support[section] = section_support

    abstain_flag = not any(prediction_oci[section] for section in sections)
    return {
        "prediction_O": [item["claim"] for item in prediction_oci["Observation"]],
        "prediction_C": [item["claim"] for item in prediction_oci["Cause"]],
        "prediction_I": [item["claim"] for item in prediction_oci["Impact"]],
        "prediction_oci": prediction_oci,
        "claim_support": claim_support,
        "suppressed_claims": suppressed_claims,
        "abstain_flag": abstain_flag,
        "abstain_reason": "all_claims_suppressed" if abstain_flag else None,
        "local_metrics": {
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "numeric_claims": numeric_claims,
            "qac_supported_claims": qac_supported_claims,
            "suppressed_claims": len(suppressed_claims),
        },
    }
