# Source-Grounded Article Rectifier

## 1. Problem statement

The input has two articles.

- The source article is the authority.
- The generated article can contain factual errors.

The system must correct the generated article. It must preserve all text that does
not need a factual change. The system is an editor. It is not an article writer.

The system uses this sequence:

`Detect -> Retrieve -> Verify -> Propose -> Patch -> Validate`

## 2. Assignment requirements

The grading command is:

```bash
python3 rectifier.py rectify-all
```

The command reads 104 mapped article pairs. It writes 104 text files to
`rectified_articles/`. The output filename stays equal to the generated input
filename. An output file contains article text only.

The system uses Python 3.11, Pydantic 2, LangGraph, and LiteLLM. It does not use a
web search, a vector database, or another paid service.

## 3. System design

Python controls the workflow. The language model performs three bounded reasoning
tasks only:

1. It identifies facts to check.
2. It judges a claim against supplied evidence.
3. It proposes a small structured patch.

Python performs all other operations. Python loads files, removes metadata, splits
sentences, retrieves source passages, resolves spans, applies patches, validates
output, and writes files.

The language model cannot write a file. It cannot return the final article as an
accepted edit.

## 4. Architecture diagram

```text
Mapped source + generated article
               |
               v
        Sanitize generated text
               |
               v
     Split text into sentence spans
               |
               v
      Group spans into 3 sentences
               |
               v
  +-------- LangGraph batch workflow --------+
  | planner (LLM)                            |
  |       |                                  |
  |       v                                  |
  | lexical source retrieval (Python)        |
  |       |                                  |
  |       v                                  |
  | evidence verifier (LLM)                  |
  |       |                                  |
  |       v                                  |
  | patch proposer (LLM)                     |
  |       |                                  |
  |       v                                  |
  | span resolver and editor (Python)        |
  |       |                                  |
  |       v                                  |
  | structural and factual guards (Python)   |
  |       |                                  |
  |       +-- retryable -> one proposal retry|
  +------------------------------------------+
               |
               v
  Merge accepted original-offset patches
               |
               v
      Atomic article text write
```

## 5. Module responsibilities

- `evaluate/` is the independent evaluation package.
- `article_rectifier/config.py` validates environment and safety settings.
- `article_rectifier/models/` defines all external and graph contracts.
- `article_rectifier/preprocessing/` sanitizes and segments input text.
- `article_rectifier/retrieval/` builds and searches one local source index.
- `article_rectifier/llm/` makes all LiteLLM calls and validates JSON.
- `article_rectifier/nodes/` contains the bounded workflow stages.
- `article_rectifier/editing/` resolves and applies exact text spans.
- `article_rectifier/graph.py` defines explicit LangGraph transitions.
- `article_rectifier/pipeline.py` processes all batches for one article.
- `article_rectifier/observability/` writes JSONL events and sums observed usage.
- `article_rectifier/io.py` loads mappings and writes output atomically.
- `rectification_system.py` is the thin compatibility entry point.
- `rectifier.py` is the required grading CLI.

## 6. LangGraph state definitions

`BatchState` contains one batch and its bounded results. Important fields are:

- `article_id` and `run_id` identify the work.
- `source_text` and `source_index` are read-only source data.
- `original_text` is the immutable sanitized generated article.
- `batch` contains editable sentence spans and read-only neighbors.
- `fact_requests` contains typed verification questions.
- `retrieved_evidence` maps each request to exact source passages.
- `verification_results` contains the four possible verdicts.
- `proposed_edits` contains structured patch proposals.
- `patch_results` contains accepted, rejected, retryable, or unresolved patches.
- `retry_count` limits the proposal loop.
- `llm_records` contains tokens, latency, retries, and reported cost.
- `errors` contains explicit degraded-operation codes.

`ArticleState` defines article-level fields. The current sequential orchestrator
derives its final result from completed batch states. It does not keep a second
mutable copy of the article.

## 7. Processing sequence

The CLI loads `article_mapping.json`. It loads the source and generated files for
one article. It passes both texts and the article ID to `rectification_system.run`.

The pipeline sanitizes the texts. It creates sentence spans and a source index.
It invokes the compiled batch graph in source order. Each batch uses offsets from
the same immutable generated article. The pipeline collects only validated edits.
It applies all final edits from right to left. It writes the result atomically.

If an unexpected article failure occurs, the CLI writes the sanitized generated
article. It marks the article as failed. It continues with the next mapping.

## 8. Three-sentence batching

The splitter is deterministic. It handles headings, common abbreviations, decimal
numbers, quotes, Unicode punctuation, and paragraph boundaries. Each sentence has
an ID, exact text, start offset, end offset, and paragraph ID.

A batch contains three editable sentences. The final batch can contain fewer than
three. One previous sentence and one next sentence are read-only context. The
patch proposer cannot edit these context sentences.

The splitter has a reconstruction check. Every non-space character must occur in
a sentence span. The reconstructed text must equal the sanitized input.

## 9. Source retrieval strategy

The system removes only the administrative `Source Article(s):` marker. It keeps
useful metadata such as the published date.

The retriever indexes exact source paragraphs and adjacent paragraph windows. It
uses a deterministic BM25 score. It stores the source start and end offset for
each passage. Each returned passage must satisfy this invariant:

```text
source[evidence.start:evidence.end] == evidence.text
```

The planner creates a query from stable subject and action words. A fallback can
use nearby claim anchors when an incorrect value has no source match. If no query
or anchor term occurs, retrieval returns no evidence. It does not invent a
passage.

Deterministic search makes evidence inspectable and repeatable. Semantic
similarity cannot prove a fact. For example, two sentences about different states
can have high similarity and still contradict each other.

## 10. Claim verification

The verifier receives one claim, one question, and exact evidence passages. It
uses one of four verdicts:

- `supported`: The evidence answers the question and entails the claim.
- `contradicted`: The evidence answers the question and gives a conflicting fact.
- `unsupported`: The source does not cover the claim.
- `uncertain`: The evidence is relevant but is not sufficient or is ambiguous.

Supported and contradicted verdicts must cite evidence. An actionable correction
must use a contradicted verdict. Correction tokens must occur in the cited source
passage. Unknown evidence IDs cause an uncertain result.

## 11. Error classification

The fixed categories include named entity, geographic, temporal, numerical,
magnitude, relationship, negation, attribution, quotation, causal, unsupported,
and other errors. One issue can use more than one category.

The planner also has a deterministic omission guard. It creates checks for dates,
numbers, percentages, and magnitude expressions that the model did not include.

## 12. Surgical patch mechanism

The patch proposer returns a `PatchProposal`. It does not return an article or a
paragraph. Each `SurgicalEdit` contains an exact old phrase, a replacement,
sentence ID, request ID, evidence IDs, category, and reason.

Insertions need an exact anchor. Deletions use an explicit delete operation. A
whole-sentence deletion needs an unsupported-claim reason.

Python resolves the old phrase inside the permitted sentence. The match must be
unique in that sentence. Python rejects a missing or ambiguous match. Python also
rejects overlap, a missing evidence record, an invalid source span, a no-op, or a
model-supplied global offset.

Python applies accepted patches from the largest start offset to the smallest
start offset. Earlier changes cannot shift later original offsets.

## 13. Validation rules

Structural validation reconstructs the expected article from the immutable
original and accepted patches. The candidate must equal this reconstruction. It
must not contain error annotations, JSON artifacts, or code fences.

Factual validation rechecks all cited evidence. A replacement value must be an
exact phrase or token subset of the evidence. Numeric replacement values must
also occur in the evidence. A deletion needs exact support for the corrected
sentence. If Python cannot establish this support, it leaves the original text.

The preservation validator reports large edits as warnings. A size threshold is
not proof that an edit is wrong. The system does not reject a grounded patch only
because it is large.

## 14. Error handling and retries

Transport retries are bounded. They use exponential backoff. Invalid JSON gets
one focused schema-repair request. Invalid JSON never becomes a fabricated patch.

Patch validation permits at most one targeted retry by default. The retry prompt
contains the exact validation failure. The graph cannot enter an unlimited loop.

A rejected or unresolved patch does not change the final article. The last known
safe text is always the immutable original plus validated patches.

## 15. LLM configuration

The main environment fields are:

```dotenv
LLM_API_KEY=
LLM_API_BASE=https://recllm.brahmastra.tech/
LLM_MODEL_NAME=groq/openai/gpt-oss-120b
```

The supplied proxy can require `openai/gpt-oss-120b` instead. Do not assume that
both identifiers work. Use this explicit command for one low-cost connection
test:

```bash
python3 rectifier.py smoke-test
```

The smoke test checks the actual provider key budget before and after its call.
It requests one small JSON object. It does not run during import or setup.

Other settings control batch size, retrieval count, timeouts, retries, local
cache, sequential concurrency, cost guard, audit logging, and optional validation.
Temperature is zero where the endpoint supports it.

## 16. Reproducibility limitations

Deterministic Python stages produce the same result for the same inputs and patch
set. Mocked tests produce the same graph result on each run. Cached valid model
responses can make development runs stable.

A live language model can change its judgment for the same prompt. Temperature
zero does not guarantee identical output. Endpoint and model changes can also
change output. Each record stores the model, prompt version, schema version, and
input fingerprint.

## 17. Evaluation metrics

The independent package supports this interface:

```python
evaluate_article(source, original, rectified, human_reference=None)
```

The default path does not call an LLM.

Character preservation is:

```text
P = max(0, 1 - Levenshtein(X, Y) / len(X))
```

Human reference similarity is:

```text
R = max(0, 1 - Levenshtein(Y, H) / len(H))
```

The evaluator also reports raw edit distance, changed-character ratio, output
integrity, annotation detection recall, correction recall, proposed edit
precision, applied edit precision, patch counts, issue counts, and usage.

Annotation recall uses the factual error phrase. It does not match only a sentence
location. Edit precision uses independent justification labels. An agent cannot
approve its own patch for this metric.

The ten human references are copied to
`evaluate/fixtures/human_references/`. `SHA256SUMS` protects them. All ten are an
evaluation holdout. The production pipeline never reads this directory.

No validated fixed NLI judge is configured. Therefore, source attribution and
new-error count are `unavailable`. The evaluator does not invent these scores. An
NLI score or a language-model judgment is not guaranteed ground truth.

## 18. Cost monitoring

Every logical model request stores input tokens, output tokens, latency, retry
count, status, model ID, and provider-reported cost when present. The usage summary
does not replace missing provider cost with zero. It reports the cost as
unavailable.

Use the supplied balance command before a real pilot:

```bash
python3 budget_checker.py
```

The local run-cost guard blocks new uncached calls after the configured observed
cost limit. It does not claim to know the provider balance. The provider balance
and a local cost estimate are different values.

## 19. Testing strategy

Unit tests do not make network calls. They use fixed mock responses. Tests cover
schemas, sanitization, sentence offsets, retrieval, verification, proposal,
editing, validation, usage, and evaluation metrics.

Integration tests run the compiled LangGraph with mock responses. They cover two
errors, no detected errors, malformed output, multiple batches, one patch retry,
and complete reconstruction. A mock full-dataset test writes 104 files to a
temporary directory. It does not overwrite the ten references.

Run the quality gates:

```bash
uv run black --check .
uv run ruff check .
uv run pytest -q
```

## 20. Known limitations

- Lexical retrieval can miss a strong paraphrase with no shared anchor.
- The planner can miss a non-numeric factual proposition.
- A supplied source can omit a generated fact. Omission is not proof of error.
- Deterministic factual validation is conservative. It can reject a valid
  paraphrased correction or deletion.
- The system does not prove that all article claims are correct.
- No independent validated NLI model is included.
- Live model output is not exactly reproducible.
- The default workflow is sequential. This protects state and budget but increases
  runtime for 104 articles.

The system leaves an uncertain claim unchanged. This behavior reduces the risk of
a new unsupported fact. It can also leave an existing error unresolved.

## 21. Setup instructions

Use Python 3.11. Install runtime dependencies:

```bash
python3 -m pip install -r requirements.txt
```

For development with `uv`:

```bash
uv sync
```

Copy `env.example` to `.env`. Set the API key, proxy URL, and verified model name.
Do not commit `.env`.

Run a one-article pilot in a dedicated output directory:

```bash
python3 rectifier.py test --count 1 --output-dir evaluate/reports/pilot-output
```

Check the provider budget before and after a real pilot. Do not use the ten human
reference output paths for experiments.

## 22. Required grading command

After dependencies and environment values are set, run:

```bash
python3 rectifier.py rectify-all
```

This command writes the mapped files to `rectified_articles/`. It continues after
one article failure. A failed article gets its sanitized generated text as a
fallback. The summary does not call this fallback factually verified.

## 23. Example end-to-end correction

The generated `article_004` text says:

> The film was released on August 15, 2024 but is reportedly facing an unofficial ban in Karnataka.

The supplied source says that the film was released on September 5 and was blocked
by most theaters in West Bengal. The source does not state the release year for
this sentence.

The safe geographic patch is:

```text
old_text: Karnataka
new_text: West Bengal
```

The result is:

> The film was released on August 15, 2024 but is reportedly facing an unofficial ban in West Bengal.

This one patch fixes only the verified geographic error. A separate verified patch
can change the date. The system must not add a release year that the supplied
source does not establish.

## 24. Accuracy and preservation trade-off

Factual correctness has priority. The system changes only a fact that source
evidence supports. Minimal editing has second priority. After it identifies the
correct fact, the system changes the smallest unique span that can correct it.

A successful workflow does not prove that the full article is correct. The report
separates observed corrections, unresolved issues, deterministic metrics, and
unavailable factuality judgments.
