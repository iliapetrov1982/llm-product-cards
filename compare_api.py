"""Run the same card generation through an external OpenAI-compatible API (OpenRouter)
on a small subset and store the comparison section of report.json.

The prompt, validator and retry logic are shared with run_batch.py, so the two
numbers in the report differ only in the model and where it runs.
"""

import argparse
import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from llm_product_cards.batch import run_concurrent
from llm_product_cards.clients import GenerationParams, generate_card, make_client, make_request_fn
from llm_product_cards.config import OPENROUTER_BASE_URL
from llm_product_cards.data import load_jsonl
from llm_product_cards.metrics import cost_api_per_1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Free OpenRouter models come and go; override with --model when this one is gone
DEFAULT_API_MODEL = "openai/gpt-oss-20b"


def parse_args() -> argparse.Namespace:
    defaults = GenerationParams()
    parser = argparse.ArgumentParser(
        description="Generate cards for a subset through OpenRouter and record the API comparison."
    )
    parser.add_argument("--input", type=Path, default=Path("data/dev.jsonl"),
                        help="input catalog, one product per line")
    parser.add_argument("--limit", type=int, default=25, help="how many products to send (free tiers are rate-limited)")
    parser.add_argument("--report", type=Path, default=Path("outputs/report.json"),
                        help="report to update; only the api_comparison section is replaced")
    parser.add_argument("--results", type=Path, default=Path("outputs/api_run.json"),
                        help="where to keep per-product facts of this run")

    parser.add_argument("--model", default=DEFAULT_API_MODEL, help="OpenRouter model id")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="max requests in flight; free models return 429 above a few")
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="generation attempts per product, including correction retries")

    parser.add_argument("--temperature", type=float, default=defaults.temperature, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=defaults.top_p, help="nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=None,
                        help="top-k cutoff; off by default, not every provider accepts it")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="repetition penalty; off by default, not every provider accepts it")
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens, help="answer length cap in tokens")
    parser.add_argument("--reasoning", choices=["off", "on"], default="off",
                        help="reasoning models spend the whole max_tokens budget thinking and return an "
                             "empty answer; 'off' asks OpenRouter to skip it")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set: put it into .env or export it")

    products = load_jsonl(args.input)[: args.limit]
    logging.info("loaded %d products from %s", len(products), args.input)

    params = GenerationParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        extra={"reasoning": {"exclude": True, "effort": "low"}} if args.reasoning == "off" else {},
    )
    client = make_client(OPENROUTER_BASE_URL, api_key)
    request_fn = make_request_fn(client, args.model, params)

    async def process(product):
        return await generate_card(product, request_fn, max_attempts=args.max_attempts)

    results, elapsed = await run_concurrent(products, args.concurrency, process)

    n = len(results)
    valid = sum(r.card is not None for r in results)
    input_tokens = sum(r.prompt_tokens for r in results)
    output_tokens = sum(r.completion_tokens for r in results)
    reasons = Counter(rule for r in results for rule in r.violations)
    api_errors = Counter(r.error.split(":")[0] for r in results if r.error)

    comparison = {
        "model": args.model,
        "n": n,
        "valid": valid,
        "valid_rate": valid / n if n else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_sec": elapsed,
        "avg_latency": sum(r.latency_sec for r in results) / n if n else 0.0,
        "avg_attempts": sum(r.attempts for r in results) / n if n else 0.0,
        "cost_per_1000": cost_api_per_1000(input_tokens, output_tokens, n) if n else 0.0,
        "errors_by_reason": dict(reasons),
        "api_errors": dict(api_errors),
        "params": asdict(params),
    }

    # Keep whatever check_metrics.py has written; replace only our section
    report = {}
    if args.report.exists():
        with open(args.report, encoding="utf-8") as f:
            report = json.load(f)
    report["api_comparison"] = comparison

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(args.results, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "cards": [asdict(r) for r in results]}, f, ensure_ascii=False, indent=2)

    logging.info(
        "%s: n=%d valid_rate=%.3f tokens in/out=%d/%d elapsed=%.1fs avg_latency=%.1fs cost=%.1f rub/1000 -> %s",
        args.model, n, comparison["valid_rate"], input_tokens, output_tokens, elapsed,
        comparison["avg_latency"], comparison["cost_per_1000"], args.report,
    )
    if reasons:
        logging.info("errors_by_reason: %s", dict(reasons.most_common()))
    if api_errors:
        logging.warning("API errors: %s", dict(api_errors))


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
