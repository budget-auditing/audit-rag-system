"""Method adapter protocol for the three RAG methods."""
from typing import Any, List, Protocol


class MethodAdapter(Protocol):
    route: str

    def answer(self, query: str, doc_scope: List[str] | None = None) -> dict:
        """Return a unified answer envelope."""
        ...


def _first_n(items: List[Any], n: int = 5) -> List[Any]:
    return items[:n]
