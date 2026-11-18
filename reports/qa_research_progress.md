# 25,000-Q&A research progress

The requested set is **not complete**. The objective remains 250 grammar and 24,750 other reviewed, distinct Sanskrit Q&As, with independent held-out evaluation. No new training or deployment occurred. `data/qa_25k/README.md` describes the separate draft queues.

## Long audit of reconstructed earlier curriculum

A disk-backed scan of 50 reconstructed files completed; see `reports/qa_pool_long_audit.json` for SHA-256 and per-file counts. It read 9,754,119 lines, of which 7,877,709 were single-turn tagged pairs. Exact NFC/whitespace deduplication found 1,372,696 distinct question-answer pairs and 1,371,977 distinct question strings; 454 questions had multiple distinct answers. Such alternatives may be legitimate or contradictory; no judgment was automated. A crude keyword check flagged 1,193,115 distinct pairs as grammar-related, leaving 179,581 without the specified keywords. This residual is **not** a count of verified non-grammar examples: grammar can be phrased differently, and many rows are mechanically generated arithmetic or paraphrases. Known incorrect time pairings and prior training/validation leakage still need attention.

These 50 files were reconstructed and their identity with the actual training data is unproven. Exact matching does not establish semantic diversity or accuracy, and prior data is not genuinely held out. No pairs have been approved as training-ready by this scan. The disk-backed SQLite audit database resides under `recovery/qa_research/`, not version control. Re-run with `python scripts/audit_qa_pool_long.py --corpus-dir PATH --db PATH --report PATH`.

## Source search

SktQA's 1,501 published benchmark records remain quarantined; see `data/qa_100k/README.md`. Sāmayik is a contemporary Sanskrit-English *parallel text* resource, not an existing Q&A corpus. Indexing four source files found 1,644 lines containing question marks (1,637 unique line hashes); these locators do not provide answers and may include quotations, multiple clauses or statements. Repo source license not established and underlying components may differ. Sāmayik published evaluation material must be kept separate from training. Use the locators solely for provenance and review, not as QA pairs. SANSKRITI, despite its name, is an English-language Indian-culture benchmark, not the requested Sanskrit Q&A source. Other parallel-text or OCR corpora likewise cannot be relabelled as Q&A without careful authoring and review.

Completing the target requires substantially more suitable material or deliberate authoring, review of answers and Sanskrit language, source-specific permissions, deduplication beyond exact strings, category balancing, and a genuinely independent evaluation suite. A simple automated corpus scan cannot guarantee 25,000 correct Q&As or do expert review.
