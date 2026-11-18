# Book-training preparation — in progress, not approved for training

Base: GitHub release `sanskrit-250m-44002147943`, `checkpoint-06-2.pt`, verified step 53,800, 220,364,800 nominal processed positions, optimizer present. The original 250M workflow was cancelled at the user's request. Keep this checkpoint and production model unchanged.

## Data preparation

Five alternative beginner storybooks were fetched through a CC0-declared GitHub mirror of Amarahasa rather than the inaccessible scanned textbooks. Source URLs, exact PDF and extraction hashes, and licensing caveats are recorded in `data/book_candidates/amarahasa/`. Raw extractions remain unchanged.

`scripts/prepare_book_review.py` extracts numbered passages into `data/book_candidates/review/review.jsonl` with source-line provenance and automated warning flags. It does not invent question-answer pairs or create a training split. It found 1,114 numbered passages, of which 133 have at least one automatically detected issue. **All** passages—including unflagged ones—remain unreviewed: correct script, meaning, ordering, source rights, extraction errors, and assessment split need review. Flags do not establish that a passage is correct. These small, repetitive storybooks alone are not adequate instruction tuning data or grammar-preservation rehearsal.

## Preliminary test leakage audit

`scripts/audit_eval_prompt_overlap.py` performs exact substring matching only (requires `pyahocorasick`). In the reconstructed local curriculum's 50 training parts, 35 of 100 `conv_eval_100` prompts, 14 of 100 `questions_100` prompts, and 1 of 100 `phase1_eval` prompts were found verbatim. None of 40 `conversational_eval` prompts were found by this exact test. This audit uses a local reconstruction, not a verified copy of the exact GitHub corpus; matches show leakage in this reconstruction, while non-matches do not prove held-out status. Pretraining, previous training, near duplicates, templates, source overlap and answers remain to be audited. See `reports/preliminary_eval_overlap.json` for matching IDs.

No 250-item held-out evaluation exists yet: the current sets include only 22 multi-turn items in `conv_eval_100` and fewer than 50 verified held-out items in several required categories. Do not report these counts as baseline scores. A separate 500-question post-training gate also remains outstanding.

## Before any candidate training or deployment

1. Correct and independently review book passages, preserving source provenance; include correctly licensed, diverse examples and verified grammar rehearsal. Never treat adjacent story sentences as automatically correct question-answer pairs.
2. Create and lock a genuinely held-out, independently checked set of at least 50 each for conversation, knowledge, grammar, multi-turn and reasoning, and audit it against the actual GitHub corpus and all prior/fine-tuning data.
3. Measure the saved base checkpoint first with fixed decoding and human or validated task-specific scoring. Train a new checkpoint separately with bounded, reproducible settings and optimizer/backup policy; re-evaluate with the exact same prompts and settings, including grammar retention.
4. Promote only if the predeclared quality and grammar-retention checks pass; otherwise retain current production. No book-derived training or deployment has occurred yet.
