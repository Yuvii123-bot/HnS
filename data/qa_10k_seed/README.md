# Authored Q&A seed for 10,000-pair goal

This initial batch contains **51** agent-authored Sanskrit Q&A pairs: 20 conversational, 20 general knowledge, 10 simple reasoning, and 1 explicit grammar. It is **not** the requested completed 10,000-pair set and is not an independently reviewed or independently evaluated dataset. The earlier desired 1% grammar ratio would be 100 grammar and 9,900 other examples in the final 10,000; this tiny seed contains 1/51 grammar, about 2%.

The seed is synthetic, short, and limited in diversity. Correctness and grammatical quality are not guaranteed. Some questions are simplified or assume context. Exact unique questions are checked, but that is not a linguistic or factual review and does not prove distinct underlying facts. This seed should not be used as held-out evaluation. Review, expand substantially, and audit for leakage before interpreting model quality or deploying a candidate.

The separate experimental training process uses one example per attention context and masks prompt/padding loss; it saves the candidate separately from the book checkpoint, preserves the optimizer, and does not auto-deploy. It does not prove that grammar retention or unwanted grammar tails have improved.
