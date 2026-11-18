# 25,000 Sanskrit Q&A project — initial draft queues

Target: 250 grammar Q&As and 24,750 other Q&As, following the previously requested 1:99 ratio. This directory is **not a 25,000-item training set**. No item here has been approved for training.

- `sktqa_pairs_review.jsonl`: 1,491 unreviewed existing Q&A records from the SktQA benchmark. All ten rows from the five normalised duplicate-question groups are excluded pending resolution. Answers and multiple-choice fields remain raw. For source license, provenance and risks see `../qa_100k/README.md` and its `sources/sktqa/LICENSE`. Health questions need specialist safety review. Verify data-specific reuse rights and decide benchmark partition policy before any training.
- `book_questions_review.jsonl`: 277 occurrences of source questions (194 distinct normalised Devanagari questions) with surrounding original IAST and automatically transliterated Devanagari. These have **no assigned answer**: nearby text may not answer the question. Chapter-local context is provided solely to help a reviewer. These stories have already been used in experimental fine-tuning and cannot serve as independent held-out evaluation.
- `report.json`: counts, limitations and targets. Counts from the two files cannot simply be added to claim Q&A progress; the books file has questions only.

All material is quarantined, may contain errors, and requires linguistic, answer, source-rights, duplicate, safety and category review. Do not use these files for training or held-out scoring. Grammar and non-grammar quotas cannot be counted until classified and reviewed. More diverse sources must be found; do not pad with duplicate or superficial synthetic questions.

Rebuild from repository root with `python scripts/prepare_25k_qa_drafts.py`. The script checks the five original book files against their manifest, preserves source identifiers, and does not auto-select answers. The separate script `scripts/catalog_qa_sources.py` checks the SktQA source hashes and builds its input catalogue. Run `python -m unittest tests.test_prepare_25k_qa_drafts tests.test_catalog_qa_sources` for the checks.
