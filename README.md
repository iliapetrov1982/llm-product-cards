# llm-product-cards

Batch generation of marketplace product cards with a small self-hosted LLM.

Input: a product catalog (title, category, attributes, customer reviews) as JSONL.
Output: one JSON card per product (`description`, `pros`, `cons`, `tags`) that a storefront
template can render without manual editing. The service runs Qwen2.5-1.5B-Instruct on a vLLM
server, validates every card against the storefront contract, retries with corrections, and
reports quality, speed and cost. The same pipeline is run through an external API for comparison.

```
data/benchmark.jsonl ──> run_batch.py ──> outputs/predictions.jsonl ──> check_metrics.py ──> outputs/report.json
                              │                 outputs/run.json ─────────────┘                      ▲
                              └─ vLLM (Qwen2.5-1.5B-Instruct, T4)                compare_api.py ─────┘
```

## Results

Benchmark: 300 products, `Qwen/Qwen2.5-1.5B-Instruct` in float16 on vLLM 0.31.0, one NVIDIA T4
(16 GB). Hardware: an AWS EC2 `g4dn.xlarge` on-demand instance (1x T4, 4 vCPU, 16 GB RAM,
us-east-1, Ubuntu 24.04 Deep Learning AMI with the NVIDIA driver and CUDA preinstalled) created
with Terraform for the duration of a session and destroyed afterwards. The whole measurement
session (environment setup, dev runs, parameter sweep, concurrency sweep, final run) took
38 minutes of instance time.

| Service requirement                 | Target     | Result                        |
|-------------------------------------|------------|-------------------------------|
| Cards passing format validation     | >= 80%     | **86.3%** (259 / 300)         |
| Time for the 300-card benchmark     | <= 8 min   | **202 s** (3.4 min)           |
| Cost per 1000 cards (own server)    | <= 25 RUB  | **12.2 RUB**                  |
| Prompt length per request           | <= 1500 tokens | enforced by construction  |

All numbers below come from `outputs/report.json` and the per-run files in `outputs/`.

## Final configuration

**Server** (run on the GPU box; weights are downloaded on first start):

```bash
# once per box: Python headers for Triton's JIT (the Deep Learning AMI ships without them)
sudo apt-get install -y python3-dev
uv venv --seed ~/vllm_env && source ~/vllm_env/bin/activate && uv pip install vllm

vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8000
```

- `--dtype half`: the T4 (Turing) has no bfloat16.
- `--max-model-len 4096`: 1500 prompt tokens plus a 350-token answer plus correction retries.
- CUDA graphs are left on (no `--enforce-eager`): on this model it roughly halves per-token
  latency compared with eager mode, at the price of ~1 min extra start-up.
- vLLM picks the Triton attention backend on the T4 (FlashAttention 2 needs compute capability >= 8).

**Generation** (defaults of `GenerationParams`, chosen in the sweep below):

| temperature | top_p | top_k | repetition_penalty | max_tokens | response_format |
|-------------|-------|-------|--------------------|------------|-----------------|
| 0.3         | 0.9   | 50    | 1.1                | 350        | `json_object`   |

**Long-input strategy:** keep the longest prefix of a product's reviews whose rendered prompt
(system + user + chat markup) fits into 1500 tokens, measured with the model's own tokenizer.
Details in [Long inputs](#long-inputs).

**Retries:** up to 3 generation attempts per product. A failed attempt is sent back to the model
together with every broken rule ("pros has 1 item, expected 2-5"); transport errors (429, 5xx,
timeouts) are retried separately with exponential backoff and never count as attempts.
On the benchmark 221 cards were accepted on the first attempt, 37 on the second and only 1 on
the third; 41 products exhausted all three. A third attempt almost never helps, so 3 is the cap:
each extra attempt costs a full prompt (~1200 tokens) and 5-15 s of GPU time.

**Concurrency:** 32 requests in flight (see [Speed](#speed)).

## Generation parameters

Sweep on the first 20 dev products, concurrency 16, `max_tokens` 500 unless stated. One knob
changes per row, the others stay at the baseline. Validity here is after retries.

| run        | change                    | valid | avg attempts | description chars (median) | out tokens / card | cards/s | failure reasons (last attempt)                 |
|------------|---------------------------|------:|-------------:|---------------------------:|------------------:|--------:|------------------------------------------------|
| baseline   | t=0.3 p=0.9 k=50 rp=1.1   | 18/20 | 1.50 | 244 | 376 | 0.46 | truncated 2                                    |
| temp 0.0   | temperature 0.0           | 17/20 | 1.35 | 215 | 336 | 0.62 | tags_count 2, truncated 1                      |
| temp 0.7   | temperature 0.7           | 16/20 | 1.80 | 224 | 412 | 0.72 | cons_count 2, description 2, tags_count 1      |
| top_p 1.0  | top_p 1.0                 | 18/20 | 1.55 | 249 | 370 | 0.54 | description_sentences 1, truncated 1           |
| top_k 20   | top_k 20                  | 17/20 | 1.45 | 208 | 313 | 0.94 | cons_count 1, pros_count 1, sentences 1        |
| top_k off  | top_k -1                  | 17/20 | 1.50 | 190 | 342 | 0.54 | tags_count 2, sentences 1                      |
| rp 1.0     | repetition_penalty 1.0    | 10/20 | 2.00 | 253 | 629 | 0.50 | truncated 4, sentences 4, cons_count 3, tags 1 |
| rp 1.2     | repetition_penalty 1.2    | 17/20 | 1.60 | 196 | 361 | 0.72 | description_length 2, sentences 1, tags 1      |
| max 250    | max_tokens 250            | 16/20 | 1.50 | 244 | 331 | 1.01 | truncated 3, cons_count 1                      |
| **max 350**| **max_tokens 350**        | **19/20** | **1.25** | 210 | 266 | **1.01** | truncated 1                             |
| no JSON mode | prompt only, no `response_format` | 0/20 | 3.00 | - | 673 | 0.51 | invalid_json 20 (Markdown-fenced JSON)   |

What the sweep showed:

- **JSON mode is not optional for this model.** Without `response_format={"type": "json_object"}`
  every answer comes wrapped in a ```` ```json ```` fence despite explicit instructions, and the
  correction retry does not fix it. With JSON mode the fence disappears entirely.
- **`repetition_penalty` is the second most important knob.** At 1.0 the model loops inside
  `tags`/`pros` until it hits `max_tokens` (half the cards truncated); 1.1 fixes that; 1.2 starts
  to hurt description length.
- **`max_tokens` 350 beats 500.** Answers that run past 350 tokens are runaway loops, not longer
  cards: they fail anyway, and waiting for 500 tokens doubles the batch time. At 250 the longest
  legitimate cards get cut.
- `temperature`, `top_p` and `top_k` move the result by at most one card on 20 products, which is
  inside the noise of this sample size. The baseline values were kept.
- During prompt tuning the main lever for validity was not a sampling parameter but the sample
  card in the system prompt: the model copies its length, so a 190-character sample produced
  descriptions right at the 120-character minimum; a 330-character sample moved the median to
  ~220 characters and removed `description_length` from the failure list.

## API comparison

The same prompt, validator and retry loop were run through OpenRouter (`openai/gpt-oss-20b`)
on 25 dev products with concurrency 4 (`compare_api.py`).

| metric                       | local: Qwen2.5-1.5B on vLLM (T4) | API: gpt-oss-20b via OpenRouter |
|------------------------------|---------------------------------:|--------------------------------:|
| valid cards                  | 86.3% (300 products)             | 100% (25 products)              |
| tokens per card, in / out    | 1666 / 317 (summed over attempts) | 979 / 290                      |
| time per card (avg latency)  | 20.4 s at concurrency 32 (5.6 s at concurrency 1) | 11.3 s at concurrency 4 |
| cost per 1000 cards          | 12.2 RUB                         | 37.0 RUB                        |

Notes:

- API latency includes network transfer and the provider's own queue, so it is not directly
  comparable with the local number; the local p50 latency is 14.6 s only because 32 requests
  share one T4.
- Token counts are not comparable across models either: the two tokenizers split the same
  Russian text differently (979 vs ~1190 prompt tokens per attempt), which is why the API cost
  is computed from the API's own `usage` and never from the local tokenizer.
- The 20B model needed one correction retry (a copied `product_id`) and otherwise produced
  richer, more fluent descriptions; the 1.5B model is good enough for the format contract but
  its texts are plainer.

**API errors met and how they are handled:**

- The model is a reasoning model: by default it spent the whole `max_tokens` budget on hidden
  reasoning and returned an empty `content` with `finish_reason="length"` (0 / 3 valid, 118 RUB
  per 1000). `compare_api.py --reasoning off` sends `reasoning: {exclude: true, effort: low}`;
  with it 25 / 25 cards are valid.
- 429 and 5xx are retried with exponential backoff and jitter (max 3 tries); 400 is not retried
  and ends up in the product's `error` field; a product that fails is recorded and the batch
  continues. None of these occurred in the final run, but the no-server dry runs exercise the
  same path (`APIConnectionError` -> 3 retries -> failed card, batch completes).

**Verdict for this workload:** the API is better on text quality and needs no infrastructure; the
own server is three times cheaper per card and, at the required volume, the only option that
fits the budget. See [Cost and decision](#cost-and-decision).

## Benchmarks

### Quality

259 / 300 valid cards (`valid_rate` 0.863). Failure reasons on the last attempt of the 41
failed products (a card can break several rules):

| reason                 | count | meaning                                      |
|------------------------|------:|----------------------------------------------|
| cons_count             | 11    | empty or too long `cons` list                |
| tags_count             | 11    | fewer than 3 tags                            |
| description_sentences  | 10    | 1 sentence or more than 5                    |
| invalid_json / truncated | 9   | runaway generation cut at `max_tokens`       |
| pros_count             | 3     | fewer than 2 pros                            |

In `report.json` these 41 products appear as `missing_fields`: when generation fails the
prediction file keeps only `product_id` for that product, so the storefront-side check sees a
card without the content fields.

### Speed

Throughput and latency vs concurrency on the first 40 benchmark products (final parameters):

| concurrency | elapsed, s | cards/s | avg latency, s | projected time for 300, s | valid |
|------------:|-----------:|--------:|---------------:|--------------------------:|------:|
| 1           | 222.8      | 0.18    | 5.6            | 1671                      | 31/40 |
| 4           | 67.9       | 0.59    | 6.2            | 509                       | 34/40 |
| 16          | 39.5       | 1.01    | 10.2           | 297                       | 32/40 |
| 32          | 32.7       | 1.22    | 13.7           | 246                       | 35/40 |

Throughput keeps growing up to 32 (the KV cache on the T4 holds ~83 concurrent 4096-token
sequences, so the server is not the limit yet), latency grows with the queue, and validity does
not degrade. 32 was chosen as the working point.

Full benchmark at concurrency 32: **300 cards in 202.1 s, 1.48 cards/s**, average latency
20.4 s (p50 14.6 s, p95 56.4 s), average 1.40 attempts per card.

### Cost

- Own server: `202.1 s / 3600 * 65 RUB/h / 300 * 1000 = 12.2 RUB per 1000 cards`.
- API: `(24468 * 20 + 7262 * 60) / 1e6 / 25 * 1000 = 37.0 RUB per 1000 cards`, from the real
  `usage` of the 25-card API run.

Pricing assumptions (`llm_product_cards/config.py`): GPU hour 65 RUB; API 20 RUB per 1M input
and 60 RUB per 1M output tokens.

## Long inputs

Most products have 4-25 reviews, but 45 of the 300 benchmark products (15%) have 90-220
reviews and do not fit into 1500 tokens: the longest raw prompt is ~4700 tokens.

Strategy: **truncation by whole reviews, in catalog order.** `fit_reviews()` renders the prompt
without reviews, adds reviews one by one while the estimated token count fits, then verifies
the real rendered prompt with the model tokenizer and corrects by one review in either
direction. Long products keep a median of 39 of their 167 reviews; prompts land at 1465-1490
tokens; validity on long products (38 / 45 = 84%) matches the rest of the set (221 / 255 = 87%).

Why this and not the alternatives:

- Selecting "the most useful" reviews needs a relevance or sentiment signal the data does not
  have; with plain truncation the first ~40 reviews already cover the recurring pros and cons.
- Summarizing the reviews with an extra model call would double the GPU time for 15% of the
  catalog and add a second place where facts can be lost, for no measurable gain on the format
  metric.

The 1500-token limit is applied to the full request as the server sees it (system prompt, user
message and chat template markup), not just to the review text.

## Optimizations

**float16, not INT4.** The weights of a 1.5B model take ~3.1 GB in float16, which leaves ~10 GB
of the T4 for the KV cache (340k tokens, 83 concurrent sequences). Memory is not the bottleneck,
so weight-only INT4 would buy nothing here and would cost speed: on this GPU class dequantizing
weights before every matmul makes small-batch decoding slower, and INT4 also changes answer
style (longer, more verbose outputs), which is exactly what the format contract punishes.
INT4 becomes relevant for a 7B+ model or a smaller GPU; this was not measured in the project.

**Prefix caching** is a free win for this workload: the 525-token system prompt is identical for
every request. vLLM 0.31 enables it by default (it was explicitly off in older releases), so the
measured numbers already include it.

## Cost and decision

Per 1000 cards the own server costs 12.2 RUB and the API 37.0 RUB. At the planned 50 000 cards
per month that is ~610 RUB versus ~1850 RUB. Both are within the 25 RUB per 1000 budget only for
the own server.

The GPU, however, is paid for while it runs, including the ~10 minutes it takes to start the
machine, install the environment and load the model. (The 65 RUB/h figure is an input of
the cost model, not a quote; the actual on-demand price of the `g4dn.xlarge` used here was about
$0.53/h, i.e. of the same order.) If cards are generated in weekly batches
(4 sessions a month, ~43 RUB of overhead), the monthly cost is:

| cards / month | own server, RUB | API, RUB |
|--------------:|----------------:|---------:|
| 1 000         | 56              | 37       |
| 2 000         | 68              | 74       |
| 5 000         | 104             | 185      |
| 10 000        | 165             | 370      |
| 50 000        | 651             | 1 850    |

Break-even is at roughly **1 700 cards per month** (`43 / (37 - 12.2) * 1000`); below it the API
is cheaper because the GPU would mostly sit idle, above it the own server wins and the gap grows
linearly.

**Decision:** for the planned volume (50 000 cards/month, growing) run the own vLLM server,
started per batch and shut down afterwards. Keep the API client as a fallback for small ad-hoc
batches and for products where text quality matters more than cost.

## Repository layout

```
run_batch.py            # batch generation: prompts, long-input strategy, concurrency -> predictions.jsonl, run.json
check_metrics.py        # validity, speed and cost from the run artifacts -> report.json
compare_api.py          # the same pipeline through OpenRouter -> api_comparison section of report.json
llm_product_cards/
  config.py             # model id, token budget, pricing assumptions, server URLs
  models.py             # Product / Card / CardResult types
  data.py               # JSONL I/O
  prompts.py            # system prompt, user message, correction retry message
  validation.py         # storefront contract -> list of Violation(rule, detail)
  tokenization.py       # model tokenizer, prompt length, fit_reviews()
  clients.py            # OpenAI-compatible client, transient retries, generate_card() loop
  metrics.py            # validity, speed, cost formulas
  batch.py              # bounded-concurrency runner
data/                   # dev.jsonl (40 products), benchmark.jsonl (300 products)
outputs/                # predictions.jsonl, report.json (final run); other runs are not committed
tests/                  # pytest, 78 tests, no network needed
```

## How to run

Local environment (Python 3.13, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest
```

On the GPU box, start the server (see [Final configuration](#final-configuration)), then:

```bash
uv run run_batch.py --input data/benchmark.jsonl --concurrency 32 --max-tokens 350
uv run check_metrics.py
```

`run_batch.py --help` lists every flag: input/output paths, `--limit` for sweeps, server URL and
model, concurrency, attempts, all sampling parameters and `--no-json-mode`.

API comparison (needs `OPENROUTER_API_KEY` in `.env`, see `.env.example`):

```bash
uv run compare_api.py --limit 25 --concurrency 4
```

`check_metrics.py` and `compare_api.py` each rewrite only their own sections of
`outputs/report.json`, so they can be run in any order.
