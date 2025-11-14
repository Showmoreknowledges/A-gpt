"""LLM-based reranking utilities for candidate alignment lists."""

from __future__ import annotations

import re
from typing import List, Mapping, Sequence

from .prompt_templates import make_rerank_prompt
from .providers import LLMClient


def _combine_description(record: Mapping[str, str]) -> str:
    """Merge structural and semantic descriptions into a single string."""

    struct_desc = str(record.get("struct_desc", ""))
    sem_desc = str(record.get("sem_desc", ""))
    if struct_desc and sem_desc:
        return f"Structural: {struct_desc}\nSemantic: {sem_desc}"
    return struct_desc or sem_desc


class LLMReranker:
    """Use an LLM to reorder coarse candidate lists."""

    def __init__(self, client: LLMClient, model: str):
        self.client = client
        self.model = model

    def rerank_one(
        self,
        source_desc: str,
        candidate_descs: Sequence[str],
        candidate_ids: Sequence[int],
    ) -> List[int]:
        if len(candidate_descs) != len(candidate_ids):
            raise ValueError("candidate_descs and candidate_ids must have the same length")

        prompt = make_rerank_prompt(source_desc, candidate_descs)
        output = self.client.complete(model=self.model, prompt=prompt)
        new_order = self._parse_ranking(output, len(candidate_ids))
        return [candidate_ids[i] for i in new_order]

    def _parse_ranking(self, text: str, k: int) -> List[int]:
        """Parse ``Ranking: 2 > 1 > 3`` style outputs into index order."""

        pattern = re.compile(r"(\d+)")
        seen: list[int] = []
        for match in pattern.findall(text or ""):
            idx = int(match) - 1  # prompt numbers candidates starting at 1
            if 0 <= idx < k and idx not in seen:
                seen.append(idx)
            if len(seen) == k:
                break

        if len(seen) < k:
            seen.extend(i for i in range(k) if i not in seen)
        return seen


def rerank_candidate_lists(
    source_records: Sequence[Mapping[str, str]],
    target_records: Sequence[Mapping[str, str]],
    coarse_indices: Sequence[Sequence[int]],
    reranker: LLMReranker,
) -> list[list[int]]:
    """Convert coarse rank lists into final LLM-refined orderings."""

    final_rankings: list[list[int]] = []
    if len(source_records) != len(coarse_indices):
        raise ValueError("Number of source descriptions must match coarse rank rows")

    num_targets = len(target_records)
    for source_record, candidate_ids in zip(source_records, coarse_indices):
        source_desc = _combine_description(source_record)
        candidate_ids = [int(cid) for cid in candidate_ids]
        for cid in candidate_ids:
            if cid < 0 or cid >= num_targets:
                raise IndexError(f"candidate id {cid} out of bounds for target records")
        candidate_descs = [_combine_description(target_records[cid]) for cid in candidate_ids]
        reranked = reranker.rerank_one(source_desc, candidate_descs, candidate_ids)
        final_rankings.append(reranked)

    return final_rankings