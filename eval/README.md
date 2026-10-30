# eval/

Evaluation data and run logs.

| File | What it is |
|---|---|
| `golden_set.json` | 60 hand-built questions (buckets A-F). 40 retrieval-scored, 20 behavior-only. |
| `external_egyptian_legal_v2.json` | 516 questions converted from the HF dataset `tarekys5/egyptian_legal_v2` validation split (Apache-2.0). 167 civil-code items are retrieval-scored; 349 other-law items test scope routing and web behavior. |
| `validation-00000-of-00001.parquet` | Source parquet for the external set. Conversion needs `pyarrow` (`pip install -e .[dev]`). |
| `runs/` | JSONL logs from `scripts/eval_answers.py` (one row per question, resumable). |

Scoring notes:

- `expected` = article numbers verified (or candidate-verified) against
  `data/civil_law_records.json`. Empty `expected` = behavior-only item.
- External items are all `verified: false`: the dataset's article numbers were
  audited (all 167 civil ones exist in our corpus) but not human-reviewed.
  The dataset's LLM-written answers are excluded; `article_text` holds the
  verbatim law text instead.
- Other-law items (penal, constitution, labor, ...) have no indexed text in
  this project, so they are not retrieval-scored.

Commands:

```bash
python scripts/eval_retrieval.py                    # embedder bake-off on golden_set
python scripts/eval_answers.py --eval-file eval/external_egyptian_legal_v2.json \
    --out eval/runs/ext516.jsonl                    # full agent run, resumable
```
