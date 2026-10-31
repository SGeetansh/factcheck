# Random-sample source review — 2026-10-31

Selection: Python `random.Random(20261031).sample` over IDs 11–104, excluding
22, 50, and 52. Selected order: 53, 42, 62. Each article ran separately with paid
API access and a provider balance check before and after. No full dataset run.
The labels were not passed to the rectification agents. These inspected cases
are now development data, not an independent human-parity benchmark.

| Article | Source review | Accepted edits | Preservation |
| --- | --- | ---: | ---: |
| 053 | No annotated errors; unchanged. Demand versus demand-growth ambiguity remains in supplied source. | 0 | 100% |
| 042 | All four annotated groups corrected: investor/merger, battery partners, rotor speed, power range. | 7 | 97.57% |
| 062 | Both annotated groups corrected: free storage and Android availability. | 2 | 99.54% |

Six of six source-corroborated annotated error groups were resolved. The review
checked nine applied substitutions against the source and found no unsupported
substitution among them. This is not a claim of exhaustive recall, exact global
new-error count, or human-level accuracy. The source-reviewed labels, quotations,
and file checksums are in `source-review.json`. The reviewer was the coding
assistant, not an independent human adjudicator. Deterministic preservation and
usage metrics are in `evaluation.json`.

Article 062 changes free storage to 100MB, as explicitly stated in one supplied
source. Another supplied source says 100MiB. These are not equal byte counts;
the development annotation explicitly permits the common MB description.
Paid storage remains 100GB. Article 042 retains the correct BMW mention in its
separate collaboration list, changing only the wrong investor relationship.

Article 053 is not a certified error-free negative. The main report concerns
*demand growth*, but a supplied summary and speaker quotation say *demand*.
The verifier accepted the demand wording. This ambiguity needs editorial
adjudication before claiming complete correctness. No external facts were used
as a substitute for the supplied source.

## Robustness findings and fix

The paid runs recovered from empty responses and a transient review failure.
All nine final patches survived source-based sentence review. Their full results,
console logs, budget snapshots, and call traces remain in the individual
`random-20261031-NNN` directories and `audit_logs/`.

Final issue accounting previously checked request IDs. It falsely retained
corrected duplicate requests and could hide partial corrections when one of a
request's patches passed. The fix checks every structured fact at its exact
original sentence span using only accepted patches. Offline recomputation from
the paid traces reduces article 042's unresolved requests from four to two and
article 062's from one to zero. The remaining article 042 judgments concern the
specific week and a marketing comparison. They are not established missed edits.

A separate cache-only diagnostic could not reproduce the complete paid execution:
failed API responses are intentionally not successful-response cache entries.
Different failure feedback changed a retry request. These diagnostic outputs are
retained in `*-replay` directories but excluded from factual-quality metrics and
were not copied to `rectified_articles/`. The accounting regression was checked
against the actual saved paid-run states and with focused offline tests.

## Usage and integrity

Paid calls: 12 (053), 20 (042), 17 (062). Total input tokens: 167,236;
output tokens: 58,119. Calls include retries and schema repair. Per-call provider
cost is unavailable. Use saved balance snapshots; their deltas can lag billing.

All three paid-run outputs are saved as article text in `rectified_articles/`.
No annotation blocks, JSON wrappers, or code fences are included. All ten original
human references and protected copies still match their SHA-256 manifest.
