# Authored Q&A seed for 10,000-pair goal

The current batches contain **147** agent-authored Sanskrit Q&A pairs: 60 conversational, 55 general knowledge, 30 simple reasoning, and 2 explicit grammar. `authored_v1.tsv` contains the original 51 examples; `authored_v2.tsv` adds 96. This is **not** the requested completed 10,000-pair set and is not an independently reviewed or independently evaluated dataset. The earlier desired 1% grammar ratio would be 100 grammar and 9,900 other examples in the final 10,000; these drafts currently contain 2/147 grammar, about 1.36%.

`reports/authored_qa_progress.json` records exact normalised duplicates and exact overlaps against four old evaluation lists. There were zero exact matches, but those lists were previously found to be contaminated and exact comparisons do not detect semantic overlap. They do not establish a valid held-out evaluation. `scripts/audit_authored_qa.py` can repeat this audit.

The seed is synthetic, short, and limited in diversity. Correctness and grammatical quality are not guaranteed. Some questions are simplified or assume context. Exact unique questions are checked, but that is not a linguistic or factual review and does not prove distinct underlying facts. This seed should not be used as held-out evaluation. Review, expand substantially, and audit for leakage before interpreting model quality or deploying a candidate.

The separate experimental training process uses one example per attention context and masks prompt/padding loss; it saves the candidate separately from the book checkpoint, preserves the optimizer, and does not auto-deploy. It does not prove that grammar retention or unwanted grammar tails have improved.
