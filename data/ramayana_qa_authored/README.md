# Rāmāyaṇa question–answer project

**Current progress: 400 / approximately 100,000 requested pairs.** These are an initial, assistant-authored, Sanskrit-language draft, **not** the complete epic or a reviewed training dataset. They cover source file lines 1–216 only, the opening summary and the start of the next episode in the Bālakāṇḍa. There are five books in this source, not seven: Yuddha and Uttara are absent. The epic has multiple recensions; exhaustive coverage and 100,000 genuinely distinct, useful questions are not guaranteed.

Each record links to a pinned original file and line number and includes a context window ending at its focus line. Line numbers refer to lines in the digitized source file, **not** necessarily canonical verse numbers. Some questions require preceding lines for context. Questions and answers are newly authored; source passages have not been independently proofread and may contain transcription errors. A human Sanskrit and textual expert must check each question, answer, context, source, recensional claim, naturalness, duplicate/near-duplicate status and permission before any use in training. All rows are marked `authored_unreviewed` and `quarantine`; no row is approved for training.

Source: [CLTK Sanskrit Gita Supersite](https://github.com/cltk/sanskrit_text_gitasupersite), commit `d8a6ccec70099d8f2e23aa3d7fdf439d336b4116`, `ramayana/balakanda_sanskrit.txt`. The repository labels its materials Public Domain Mark 1.0; that designation is not a copyright warranty, and underlying transcription and applicable jurisdiction should be checked. Preserve attribution and hashes. The original verse text has previously been used for training in this project; some question content may also overlap earlier QA training.

## Evaluation and future work

Do not call these questions held out or calculate accuracy from them: the source text and potentially similar questions have been seen. Set aside independent, expert-reviewed QA **before** collecting or authoring further training data; audit exact and semantic overlap against all earlier corpora, questions, and evaluation suites. Reserve source passages or episodes by group where possible, rather than splitting near-paraphrases at random. Grammar/conversation retention requires a separate held-out suite. Do not claim an independent holdout without a complete audit. Never reuse the quarantined SktQA benchmark as an independent test.

Scale in traceable batches across all available chapters, using source-specific citations and review, deduplication, topical coverage and licensing records. Acquire and verify suitable Yuddha and Uttara editions before claiming coverage. Avoid padding the target with formulaic paraphrases, fabricated facts or unsupported answers. A 100,000 target is a planning goal, not a reason to weaken quality.

## Explicitly requested experiment

At the user's request, the first 300 unreviewed rows are being used in a separate experimental GitHub fine-tuning run with replay from the prior authored QA and book data. This is not independent review or general approval of the remaining rows. Do not deploy the candidate without the applicable evaluation and approval.
