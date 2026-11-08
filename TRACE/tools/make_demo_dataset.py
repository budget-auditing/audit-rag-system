"""Generate the synthetic Audit-OCIQA demo dataset.

Every department, amount and narrative produced here is fabricated. The
output only mirrors the schema of the (non-public) Audit-OCIQA benchmark so
that the TRACE pipeline can be exercised end to end.

Usage (from the TRACE root):
    python -m tools.make_demo_dataset [--output-dir DIR] [--seed N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROVINCE = "示例省"
SALT = "trace-demo-v1"
YEARS = (2023, 2024)
DEFAULT_OUTPUT_DIR = "datasets-public/audit-ociqa-demo"

DEPARTMENTS = [
    ("D01", "205", "教育支出"),
    ("D02", "210", "卫生健康支出"),
    ("D03", "213", "农林水支出"),
    ("D04", "214", "交通运输支出"),
    ("D05", "207", "文化旅游体育与传媒支出"),
    ("D06", "206", "科学技术支出"),
]
HAS_FUND = {"D02", "D04"}
HAS_SPECIAL = {"D01", "D03", "D05"}
SPECIAL_GAP = {("D01", 2024), ("D03", 2023), ("D05", 2024)}
NO_CARRY = {("D06", 2023), ("D03", 2024)}
GROWTH_MISMATCH = {"D02", "D05"}

PROJECT_POOL = [
    "专项业务经费",
    "信息化建设与运维",
    "人才培养与培训",
    "设备购置与更新",
    "绩效评价与课题研究",
    "基础设施维修改造",
]
INCREASE_REASONS = ["新增重点项目支出", "人员经费标准调整", "信息化项目建设投入增加"]
DECREASE_REASONS = ["一次性项目到期退出", "压减一般性支出", "部分项目调整至其他渠道安排"]

CALIBER_BY_PID = {
    1: "unknown",
    2: "unknown",
    3: "general_public_budget",
    4: "general_public_budget",
    5: "general_public_budget",
    6: "unknown",
    7: "gov_fund_budget",
    8: "unknown",
}


def _hash(text: str, n: int) -> str:
    return hashlib.sha256(f"{SALT}|{text}".encode("utf-8")).hexdigest()[:n]


def _fmt(cents: int) -> str:
    return f"{cents / 100:.2f}"


def _split(total: int, parts: int, rng: random.Random) -> List[int]:
    if parts <= 1 or total <= parts:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


@dataclass
class Project:
    name: str
    general: int = 0
    fund: int = 0
    special: int = 0
    carry: int = 0

    @property
    def total(self) -> int:
        return self.general + self.fund + self.special + self.carry


@dataclass
class Budget:
    code: str
    func_code: str
    func_name: str
    year: int
    general: int
    fund: int
    special: int
    special_used: int
    carry: int
    personnel: int
    public: int
    prev_total: int
    stated_rate: float
    reason: str
    projects: List[Project] = field(default_factory=list)

    @property
    def dept(self) -> str:
        return f"部门-{self.code}"

    @property
    def basic(self) -> int:
        return self.personnel + self.public

    @property
    def total(self) -> int:
        return self.general + self.fund + self.special + self.carry

    @property
    def appropriation(self) -> int:
        return self.general + self.fund

    @property
    def project_summary(self) -> int:
        return self.total - self.basic

    @property
    def project_table_total(self) -> int:
        return sum(p.total for p in self.projects)

    @property
    def actual_rate(self) -> float:
        return (self.total - self.prev_total) / self.prev_total * 100

    @property
    def doc_id(self) -> str:
        return _hash(f"{self.code}|{self.year}", 10)

    def evidence_id(self, pid: int) -> str:
        return f"A{self.year}-{self.doc_id}-{pid:04d}"


def _make_budget(
    code: str,
    func_code: str,
    func_name: str,
    year: int,
    prev: Optional[Budget],
    rng: random.Random,
) -> Budget:
    if prev is None:
        general = rng.randint(3000_00, 15000_00)
        personnel = int(general * rng.uniform(0.25, 0.35))
        public = int(general * rng.uniform(0.04, 0.08))
    else:
        growth = rng.uniform(0.9, 1.15)
        general = int(prev.general * growth)
        personnel = int(prev.personnel * rng.uniform(1.0, 1.05))
        public = int(prev.public * rng.uniform(0.92, 1.02))

    fund = rng.randint(200_00, 800_00) if code in HAS_FUND else 0
    special = rng.randint(20_00, 120_00) if code in HAS_SPECIAL else 0
    special_used = special
    if (code, year) in SPECIAL_GAP:
        special_used = special - int(special * rng.uniform(0.15, 0.4))
    carry = 0 if (code, year) in NO_CARRY else rng.randint(50_00, 600_00)

    total = general + fund + special + carry
    if prev is None:
        prev_total = int(total / rng.uniform(0.92, 1.12))
    else:
        prev_total = prev.total
    actual_rate = (total - prev_total) / prev_total * 100
    stated_rate = round(actual_rate, 1)
    if year == YEARS[-1] and code in GROWTH_MISMATCH:
        stated_rate = round(actual_rate + rng.choice([-1, 1]) * rng.uniform(3, 8), 1)
    reasons = INCREASE_REASONS if actual_rate >= 0 else DECREASE_REASONS

    budget = Budget(
        code=code,
        func_code=func_code,
        func_name=func_name,
        year=year,
        general=general,
        fund=fund,
        special=special,
        special_used=special_used,
        carry=carry,
        personnel=personnel,
        public=public,
        prev_total=prev_total,
        stated_rate=stated_rate,
        reason=rng.choice(reasons),
    )
    budget.projects = _make_projects(budget, rng)
    return budget


def _make_projects(b: Budget, rng: random.Random) -> List[Project]:
    names = rng.sample(PROJECT_POOL, rng.randint(2, 3))
    projects = [Project(name=n) for n in names]
    for project, amount in zip(projects, _split(b.general - b.basic, len(projects), rng)):
        project.general = amount
    if b.carry:
        carry_targets = projects[: min(2, len(projects))]
        for project, amount in zip(carry_targets, _split(b.carry, len(carry_targets), rng)):
            project.carry = amount
    if b.fund:
        projects.append(Project(name="政府性基金安排项目", fund=b.fund))
    if b.special_used:
        projects.append(Project(name="专项培训与会议", special=b.special_used))
    return projects


def _paragraphs(b: Budget) -> Dict[int, str]:
    title = f"{PROVINCE}{b.year}年省级部门预算（演示数据）"
    project_rows = "\n".join(
        f"| {p.name} | {_fmt(p.total)} | {_fmt(p.general)} | {_fmt(p.fund)} "
        f"| {_fmt(p.special)} | {_fmt(p.carry)} |"
        for p in b.projects
    )
    gp_projects = b.general - b.basic
    if b.fund:
        fund_text = (
            "| 科目名称 | 合计 | 项目支出 |\n| --- | --- | --- |\n"
            f"| {b.func_name}（政府性基金） | {_fmt(b.fund)} | {_fmt(b.fund)} |"
        )
    else:
        fund_text = f"本部门{b.year}年无政府性基金预算拨款安排的支出。"
    trend = "增长" if b.stated_rate >= 0 else "下降"
    project_names = "、".join(p.name for p in b.projects)

    return {
        1: f"## 封面\n| {title} |\n| --- |\n| {b.dept} |",
        2: (
            "## 一、收支预算总表\n单位：万元\n"
            "| 收入项目 | 预算数 | 支出项目 | 预算数 |\n| --- | --- | --- | --- |\n"
            f"| 一般公共预算拨款 | {_fmt(b.general)} | {b.func_name} | {_fmt(b.total)} |\n"
            f"| 政府性基金预算拨款 | {_fmt(b.fund)} | 基本支出 | {_fmt(b.basic)} |\n"
            f"| 财政专户管理资金 | {_fmt(b.special)} | 项目支出 | {_fmt(b.project_summary)} |\n"
            f"| 上年结转结余 | {_fmt(b.carry)} |  |  |\n"
            f"| 收入总计 | {_fmt(b.total)} | 支出总计 | {_fmt(b.total)} |"
        ),
        3: (
            "## 二、财政拨款收支预算总表\n单位：万元\n"
            "| 收入项目 | 预算数 | 支出项目 | 预算数 |\n| --- | --- | --- | --- |\n"
            f"| 一般公共预算拨款 | {_fmt(b.general)} | {b.func_name} | {_fmt(b.appropriation)} |\n"
            f"| 政府性基金预算拨款 | {_fmt(b.fund)} |  |  |\n"
            f"| 收入合计 | {_fmt(b.appropriation)} | 支出合计 | {_fmt(b.appropriation)} |"
        ),
        4: (
            "## 三、一般公共预算支出表\n单位：万元\n"
            "| 科目编码 | 科目名称 | 合计 | 基本支出 | 项目支出 |\n| --- | --- | --- | --- | --- |\n"
            f"| {b.func_code} | {b.func_name} | {_fmt(b.general)} | {_fmt(b.basic)} | {_fmt(gp_projects)} |"
        ),
        5: (
            "## 四、一般公共预算基本支出表\n单位：万元\n"
            "| 项目 | 金额 |\n| --- | --- |\n"
            f"| 人员经费 | {_fmt(b.personnel)} |\n"
            f"| 公用经费 | {_fmt(b.public)} |\n"
            f"| 合计 | {_fmt(b.basic)} |"
        ),
        6: (
            "## 五、项目支出预算表\n单位：万元\n"
            "| 项目名称 | 合计 | 一般公共预算 | 政府性基金 | 财政专户 | 单位资金（上年结转结余） |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            f"{project_rows}\n"
            f"| 合计 | {_fmt(b.project_table_total)} | {_fmt(gp_projects)} | {_fmt(b.fund)} "
            f"| {_fmt(b.special_used)} | {_fmt(b.carry)} |"
        ),
        7: f"## 六、政府性基金预算支出表\n单位：万元\n{fund_text}",
        8: (
            f"## 七、{b.year}年部门预算情况说明\n"
            f"本部门{b.year}年收入预算{_fmt(b.total)}万元，比上年{trend}{abs(b.stated_rate):.1f}%，"
            f"主要原因是{b.reason}。其中，一般公共预算拨款{_fmt(b.general)}万元，"
            f"政府性基金预算拨款{_fmt(b.fund)}万元，财政专户管理资金{_fmt(b.special)}万元，"
            f"上年结转结余{_fmt(b.carry)}万元。\n"
            f"支出预算{_fmt(b.total)}万元，其中基本支出{_fmt(b.basic)}万元，主要用于保障在职人员工资福利和机构正常运转；"
            f"项目支出{_fmt(b.project_summary)}万元，主要用于{project_names}。"
        ),
    }


def _claim(text: str, cites: List[str]) -> Dict:
    return {"claim": text, "cites": cites}


def _q_basic(b: Budget) -> Dict:
    query = (
        f"请核对{b.dept}的{b.year}年基本支出的构成：一般公共预算支出表列示基本支出{_fmt(b.basic)}万元，"
        "基本支出表中人员经费与公用经费分别为多少？两者合计是否与该数一致？"
    )
    e4, e5 = b.evidence_id(4), b.evidence_id(5)
    oci = {
        "Observation": [
            _claim(f"一般公共预算支出表列示{b.year}年基本支出{_fmt(b.basic)}万元。", [e4]),
            _claim(
                f"基本支出表中人员经费{_fmt(b.personnel)}万元、公用经费{_fmt(b.public)}万元，"
                f"合计{_fmt(b.basic)}万元。",
                [e5],
            ),
        ],
        "Cause": [_claim("两张表的基本支出口径一致，均由一般公共预算拨款安排。", [e4, e5])],
        "Impact": [_claim("基本支出构成勾稽一致，未发现差异风险。", [e4, e5])],
    }
    return {"qid": "Q-demo-basic", "query": query, "oci": oci}


def _q_carry(b: Budget) -> Dict:
    gap = b.total - b.appropriation
    gap_parts = f"上年结转结余{_fmt(b.carry)}万元"
    if b.special:
        gap_parts = f"财政专户管理资金{_fmt(b.special)}万元与{gap_parts}之和"
    carry_projects = "、".join(p.name for p in b.projects if p.carry)
    query = (
        f"请说明{b.dept}的{b.year}年项目支出对上年结转结余资金的依赖情况。收支预算总表显示上年结转结余"
        f"{_fmt(b.carry)}万元，财政拨款收支预算总表支出合计{_fmt(b.appropriation)}万元，较支出总计"
        f"{_fmt(b.total)}万元少{_fmt(gap)}万元。请结合项目支出预算表说明结转资金用于哪些项目、"
        "形成原因以及后续年度资金安排是否可持续。"
    )
    e2, e3, e6, e8 = (b.evidence_id(p) for p in (2, 3, 6, 8))
    oci = {
        "Observation": [
            _claim(
                f"收支预算总表显示{b.year}年上年结转结余{_fmt(b.carry)}万元，收入总计{_fmt(b.total)}万元。",
                [e2],
            ),
            _claim(
                f"财政拨款收支预算总表支出合计{_fmt(b.appropriation)}万元，较支出总计少{_fmt(gap)}万元，"
                f"等于{gap_parts}。",
                [e2, e3],
            ),
            _claim(
                f"项目支出预算表中单位资金（上年结转结余）合计{_fmt(b.carry)}万元，用于{carry_projects}。",
                [e6],
            ),
        ],
        "Cause": [_claim("材料未说明上年结转结余的形成原因（一次性项目或历年累积结余）。", [e8])],
        "Impact": [_claim("材料未说明结转资金是否可持续，后续年度同类项目的资金来源存在不确定性。", [e2, e6])],
    }
    return {"qid": "Q-demo-carry", "query": query, "oci": oci}


def _q_special_gap(b: Budget) -> Dict:
    gap = b.special - b.special_used
    query = (
        f"关于{b.dept}的{b.year}年财政专户管理资金的使用口径：收支预算总表显示财政专户管理资金收入"
        f"{_fmt(b.special)}万元，项目支出预算表中财政专户列示{_fmt(b.special_used)}万元。"
        "请说明两者差额的去向、对应科目以及是否形成结转。"
    )
    e2, e6, e8 = b.evidence_id(2), b.evidence_id(6), b.evidence_id(8)
    oci = {
        "Observation": [
            _claim(f"收支预算总表显示财政专户管理资金收入{_fmt(b.special)}万元。", [e2]),
            _claim(f"项目支出预算表中财政专户资金合计{_fmt(b.special_used)}万元。", [e6]),
            _claim(
                f"两者差额{_fmt(gap)}万元未在项目支出预算表中列示，而收支预算总表的项目支出"
                f"{_fmt(b.project_summary)}万元高于项目支出预算表合计{_fmt(b.project_table_total)}万元。",
                [e2, e6],
            ),
        ],
        "Cause": [_claim(f"材料未说明该{_fmt(gap)}万元专户资金的具体用途或结转安排。", [e8])],
        "Impact": [_claim("专户资金来源与使用的披露不完整，影响对专户资金使用完整性的审查。", [e2, e6])],
    }
    return {"qid": "Q-demo-special-gap", "query": query, "oci": oci}


def _q_growth(prev: Budget, b: Budget) -> Dict:
    trend = "增长" if b.stated_rate >= 0 else "下降"
    query = (
        f"{b.dept}的{b.year}年部门预算情况说明称收入预算比上年{trend}{abs(b.stated_rate):.1f}%。"
        f"请根据{prev.year}年与{b.year}年收支预算总表核对该变动幅度是否准确，并说明增减变动的主要原因。"
    )
    p2, e2, e8 = prev.evidence_id(2), b.evidence_id(2), b.evidence_id(8)
    diff = abs(b.actual_rate - b.stated_rate)
    observations = [
        _claim(
            f"{prev.year}年收入总计{_fmt(prev.total)}万元，{b.year}年收入总计{_fmt(b.total)}万元，"
            f"实际变动率为{b.actual_rate:.1f}%。",
            [p2, e2],
        ),
        _claim(f"预算情况说明所述变动率为{b.stated_rate:.1f}%，原因为{b.reason}。", [e8]),
    ]
    if diff >= 0.1:
        cause = _claim(f"说明所述变动率与报表计算结果相差{diff:.1f}个百分点，材料未解释口径差异。", [e2, e8])
        impact = _claim("预算说明与报表数据不一致，影响预算信息公开的准确性。", [e2, e8])
    else:
        cause = _claim(f"说明所述变动率与报表计算结果一致，主要原因为{b.reason}。", [e8])
        impact = _claim("未发现变动幅度披露不一致风险。", [e2, e8])
    oci = {"Observation": observations, "Cause": [cause], "Impact": [impact]}
    return {"qid": "Q-demo-growth", "query": query, "oci": oci}


def build(seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    budgets: List[Budget] = []
    for code, func_code, func_name in DEPARTMENTS:
        prev: Optional[Budget] = None
        for year in YEARS:
            prev = _make_budget(code, func_code, func_name, year, prev, rng)
            budgets.append(prev)

    corpus: List[Dict] = []
    for b in budgets:
        for pid, text in _paragraphs(b).items():
            corpus.append(
                {
                    "evidence_id": b.evidence_id(pid),
                    "doc_id": b.doc_id,
                    "pid": pid,
                    "year": b.year,
                    "budget_caliber": CALIBER_BY_PID[pid],
                    "budget_level": "provincial",
                    "section": "说明" if pid == 8 else "正文",
                    "text": text,
                    "source_path": f"corpus/dept-{b.code}_{b.year}.md",
                }
            )

    cases: List[tuple] = []
    by_key = {(b.code, b.year): b for b in budgets}
    for b in budgets:
        cases.append((b, _q_basic(b)))
        if b.carry:
            cases.append((b, _q_carry(b)))
        if b.special_used != b.special:
            cases.append((b, _q_special_gap(b)))
        if b.year != YEARS[0]:
            cases.append((b, _q_growth(by_key[(b.code, b.year - 1)], b)))

    qrels, gold, queries = [], [], []
    for idx, (b, case) in enumerate(cases, start=1):
        row_id = f"R{idx:06d}-{_hash(case['query'], 8)}"
        evidence_ids = list(
            dict.fromkeys(eid for claims in case["oci"].values() for c in claims for eid in c["cites"])
        )
        base = {"row_id": row_id, "qid": case["qid"], "split": "test", "query": case["query"]}
        qrels.append({**base, "gold_evidence_ids": evidence_ids})
        queries.append({**base, "year": b.year, "doc_id": b.doc_id})
        gold.append(
            {**base, "year": b.year, "doc_id": b.doc_id, "gold_evidence_ids": evidence_ids, "gold_oci": case["oci"]}
        )
    return {"corpus": corpus, "qrels": qrels, "gold": gold, "queries": queries}


def _validate(data: Dict[str, List[Dict]]) -> None:
    ids = {p["evidence_id"] for p in data["corpus"]}
    assert len(ids) == len(data["corpus"]), "duplicate evidence_id"
    for row in data["gold"]:
        cited = set(row["gold_evidence_ids"])
        for claims in row["gold_oci"].values():
            for claim in claims:
                cited.update(claim["cites"])
        missing = cited - ids
        assert not missing, f"{row['row_id']} cites unknown evidence {sorted(missing)}"
    row_ids = [r["row_id"] for r in data["qrels"]]
    assert len(set(row_ids)) == len(row_ids), "duplicate row_id"


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic Audit-OCIQA demo dataset")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = build(args.seed)
    _validate(data)
    out = Path(args.output_dir)
    _write_jsonl(out / "corpus.paragraphs.jsonl", data["corpus"])
    _write_jsonl(out / "gold.qrels.jsonl", data["qrels"])
    _write_jsonl(out / "qa.gold.jsonl", data["gold"])
    _write_jsonl(out / "splits" / "test.jsonl", data["queries"])
    print(
        f"[make_demo_dataset] wrote {len(data['corpus'])} paragraphs and "
        f"{len(data['qrels'])} queries to {out}"
    )


if __name__ == "__main__":
    main()
