# 100,000-question project: source catalogue (not a completed dataset)

Goal: 1,000 explicit grammar Q&As and 99,000 other Q&As. The target counts are for reviewed, distinct, relevant Sanskrit pairs, not repeated samples or arbitrary consecutive sentences. There are currently **zero approved pairs**; the catalogue below is not training-ready. Do not start training on it.

## First source: SktQA

- Repository: https://github.com/mahesh-ak/SktQA, commit `7cb06518a019d571c67dab425c34c9e460323143`.
- Paper: Akavarapu et al., *A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs*, Findings of ACL 2025, https://aclanthology.org/2025.findings-acl.141/.
- Original files: `data/qa_set/sanskrit.tsv` and `data/qa_set/ayurveda.tsv`; exact copies and repository's Apache-2.0 LICENSE are in `sources/sktqa/`. The repository declares Apache-2.0; verify that it covers these particular data and any underlying third-party material before training or redistribution.
- 1,501 raw records, 1,496 distinct NFC/whitespace-normalised questions, five repeated-question groups including four with non-identical answer fields. Such differences may be alternative valid answers rather than actual errors; investigate rather than discard automatically.
- This is a published **benchmark**, not a general-purpose conversation dataset. Some entries contain multiple-choice context or multiple alternative answers. Some concern Ayurveda, potentially involving health claims. Categorisation, linguistic review, source and medical-safety checks are essential. Keep quarantined pending rights and partition decisions.
- `python scripts/catalog_qa_sources.py` validates source hashes and regenerates the raw, unreviewed JSONL and counts. It does not pick an answer, infer a category, assign training/test splits or approve a row.

## Work required before training

1. Source substantially more diverse, appropriately licensed Sanskrit Q&A; record provenance and license for each item. Do not represent story sentences or translations as verified Q&A.
2. Reserve independent, reviewed held-out evaluation *before* training. Audit exact and near duplicates across all splits and prior training corpus; benchmark usage requires a deliberate leakage policy. No held-out scores may include items used for training.
3. Review questions, direct answers, relevance, Sanskrit quality, grammar category, safety and conflicting alternatives. Verify expert review of grammar answers and evaluate retention separately.
4. Balance the final reviewed training set to exactly 1,000 grammar and 99,000 other items, tracking counts by category and source. Do not duplicate or generate superficial variations to hit the target.
5. Preserve the current model and optimizer; train a separate candidate with explicit end-of-example boundaries, then compare with the baseline using identical decoding and report failures. No deployment based solely on volume or loss.
