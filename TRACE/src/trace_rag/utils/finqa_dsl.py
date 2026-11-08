"""
FinQA DSL executor and LLM output parser.

DSL operations:
  Math:  add, subtract, multiply, divide, greater, exp
  Table: table_sum, table_average, table_max, table_min
  Constants: const_1 .. const_10, const_100, const_1000, ..., const_m1
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

CONST_MAP: Dict[str, float] = {
    "const_m1": -1,
    **{f"const_{i}": float(i) for i in range(1, 11)},
    "const_100": 100,
    "const_1000": 1000,
    "const_10000": 10000,
    "const_100000": 100000,
    "const_1000000": 1e6,
    "const_1000000000": 1e9,
}

_STEP_RE = re.compile(r"#(\d+)")
_PCT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%$")
_NUM_RE = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?%?$")
_CALL_RE = re.compile(
    r"(add|subtract|multiply|divide|greater|exp"
    r"|table_sum|table_average|table_max|table_min)"
    r"\s*\(\s*([^)]*)\)",
    re.IGNORECASE,
)


def _parse_arg(raw: str, memory: Dict[str, float]) -> Optional[float]:
    """Resolve a single argument to a float."""
    raw = raw.strip()
    if not raw or raw.lower() == "none":
        return None
    step_m = _STEP_RE.fullmatch(raw)
    if step_m:
        return memory.get(f"#{step_m.group(1)}")
    if raw in CONST_MAP:
        return CONST_MAP[raw]
    cleaned = raw.replace(",", "").replace("$", "").strip()
    pct_m = _PCT_RE.match(cleaned)
    if pct_m:
        return float(pct_m.group(1)) / 100.0
    try:
        return float(cleaned)
    except ValueError:
        return None


def execute_program(steps: List[Tuple[str, List[str]]]) -> Optional[float]:
    """Execute a parsed DSL program and return the final result."""
    memory: Dict[str, float] = {}
    result: Optional[float] = None
    for idx, (op, args) in enumerate(steps):
        op_lower = op.lower()
        vals = [_parse_arg(a, memory) for a in args]

        if op_lower in ("table_sum", "table_average", "table_max", "table_min"):
            nums = [v for v in vals if v is not None]
            if not nums:
                return None
            if op_lower == "table_sum":
                result = sum(nums)
            elif op_lower == "table_average":
                result = sum(nums) / len(nums)
            elif op_lower == "table_max":
                result = max(nums)
            elif op_lower == "table_min":
                result = min(nums)
        else:
            a = vals[0] if len(vals) > 0 else None
            b = vals[1] if len(vals) > 1 else None
            if a is None:
                return None
            if op_lower == "add":
                result = a + (b if b is not None else 0)
            elif op_lower == "subtract":
                result = a - (b if b is not None else 0)
            elif op_lower == "multiply":
                result = a * (b if b is not None else 1)
            elif op_lower == "divide":
                if b is None or b == 0:
                    return None
                result = a / b
            elif op_lower == "greater":
                if b is None:
                    return None
                result = 1.0 if a > b else 0.0
            elif op_lower == "exp":
                if b is None:
                    return None
                try:
                    result = math.pow(a, b)
                except (OverflowError, ValueError):
                    return None
            else:
                return None

        if result is not None:
            memory[f"#{idx}"] = result
    return result


def parse_program_string(prog_str: str) -> List[Tuple[str, List[str]]]:
    """Parse a canonical DSL string like 'subtract(5829, 5735), divide(#0, 100)'."""
    steps: List[Tuple[str, List[str]]] = []
    for m in _CALL_RE.finditer(prog_str):
        op = m.group(1).lower()
        raw_args = m.group(2)
        args = [a.strip() for a in raw_args.split(",") if a.strip()]
        steps.append((op, args))
    return steps


def extract_program_from_llm(text: str) -> List[Tuple[str, List[str]]]:
    """
    Extract DSL program from free-form LLM output.
    Looks for lines matching op(arg, arg) patterns.
    """
    steps: List[Tuple[str, List[str]]] = []
    for m in _CALL_RE.finditer(text):
        op = m.group(1).lower()
        raw_args = m.group(2)
        args = [a.strip() for a in raw_args.split(",") if a.strip()]
        steps.append((op, args))
    return steps


def execute_from_string(prog_str: str) -> Optional[float]:
    """Parse and execute a program string."""
    return execute_program(parse_program_string(prog_str))


def execute_from_llm_output(text: str) -> Optional[float]:
    """Extract DSL from LLM output and execute."""
    steps = extract_program_from_llm(text)
    if not steps:
        return None
    return execute_program(steps)


def format_dsl_prompt_block() -> str:
    """Return the DSL instruction block for LLM prompts."""
    return (
        "Your ONLY task is to output a DSL reasoning program. Do NOT output explanations, analysis, or natural language.\n\n"
        "DSL operations: add(a, b), subtract(a, b), multiply(a, b), divide(a, b), greater(a, b), exp(a, b).\n"
        "Use #0, #1, ... to refer to the result of a previous step.\n"
        "Constants: const_2, const_3, const_100, const_1000, const_m1, etc.\n"
        "Percentages: write 23.6% directly (the executor converts it to 0.236).\n"
        "For ratio/percentage questions, use divide(a, b) directly — do NOT multiply by const_100.\n\n"
        "Examples:\n"
        "Q: Net change from $5735M to $5829M?\nProgram: subtract(5829, 5735)\n\n"
        "Q: What percentage did revenue increase from 30 to 164?\nProgram: subtract(164, 30), divide(#0, 30)\n\n"
        "Q: What was the percent change in cost from 2005 ($47.6M) to 2006 ($26.6M)?\n"
        "Program: subtract(26.6, 47.6), divide(#0, 47.6)\n\n"
        "Q: What portion of total debt ($42546) is due in the next 12 months ($1336)?\nProgram: divide(1336, 42546)\n\n"
        "Q: What percentage of total payments ($777K) are due after 2015 ($390K)?\nProgram: divide(390, 777)\n\n"
        "Q: Total operating expenses if fuel $9896M is 23.6% of total?\nProgram: divide(9896, 23.6%)\n\n"
        "Q: What fraction of 2936 is 843?\nProgram: divide(843, 2936)\n\n"
        "Q: What is the average of 84%, 73%, and 65%?\nProgram: add(84%, 73%), add(#0, 65%), divide(#1, const_3)\n\n"
        "Q: What is the ROI from 2008 ($63.00) to 2009 ($79.67)?\nProgram: subtract(79.67, 63.00), divide(#0, 63.00)\n\n"
        "Q: What percent would balance ($127.1) increase if $108.7 and $138.0 were added?\n"
        "Program: add(108.7, 138.0), add(#0, 127.1), subtract(#1, 127.1), divide(#2, 127.1)\n\n"
        "Q: Company A's return ($322.60) was what percent of peer group ($271.10)?\nProgram: divide(322.60, 271.10)\n\n"
        "Q: What was the change in OTTI between 2011 ($9.2M loss) and 2012 ($19.8M loss)?\nProgram: subtract(19.8, 9.2)\n"
        "NOTE: When values are losses/expenses, use their ABSOLUTE magnitudes for change calculations.\n\n"
        "Q: What is the variation in lease obligations from 2011 ($124M) to 2012 ($148M)?\nProgram: subtract(148, 124)\n"
        "NOTE: change/variation from year A to year B = (B_value - A_value). LATER year minus EARLIER year.\n\n"
        "Q: What percentage of $99M in tax benefits would NOT impact the effective rate ($89M would impact)?\n"
        "Program: divide(89, 99), subtract(const_1, #0)\n"
        "NOTE: \"would NOT X\" = 1 - (part that WOULD X / total).\n\n"
        "KEYWORD GUIDE:\n"
        "- \"portion\", \"fraction\", \"percentage of total\", \"what percent of X is Y\" → divide(Y, X)\n"
        "- \"change\", \"variation\", \"difference\" from year A to year B → subtract(B_value, A_value)\n"
        "- \"percent change\" from A to B → subtract(B_value, A_value), divide(#0, A_value)\n"
        "- Values described as losses, deficits, or charges → use POSITIVE magnitudes in subtract/divide\n"
        "- \"would NOT\" / \"excluding\" → compute the complement: 1 - (part / total)\n\n"
        "STRICT RULES:\n"
        "1. You MUST output a program even if evidence seems incomplete — use the best available numbers.\n"
        "2. Output ONLY one line starting with \"Program: \" followed by DSL operations.\n"
        "3. Do NOT write any text, reasoning, or explanation before or after the program.\n"
    )
