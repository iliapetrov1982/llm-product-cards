import json
import asyncio
import logging
import argparse
from pathlib import Path
from dataclasses import asdict

from llm_product_cards.data import load_jsonl, save_jsonl
from llm_product_cards.batch import run_concurrent
from llm_product_cards.config import VLLM_BASE_URL, MODEL
from llm_product_cards.clients import GenerationParams, make_client, make_request_fn, generate_card

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    defaults = GenerationParams()
    parser = argparse.ArgumentParser(
        description="Generate product cards for a JSONL catalog via an OpenAI-compatible server."
    )
    # Files
    parser.add_argument(
        "--input", type=Path, default=Path("data/dev.jsonl"), help="input catalog, one product per line"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/predictions.jsonl"),
        help="where to write generated cards, one per line"
    )
    parser.add_argument(
        "--run-info", type=Path, default=Path("outputs/run.json"),
        help="where to write run facts: elapsed time, tokens, latency, attempts"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="process only the first N products (e.g. for a parameter sweep)"
    )

    # Server and batching
    parser.add_argument("--base-url", default=VLLM_BASE_URL, help="OpenAI-compatible server base URL")
    parser.add_argument("--model", default=MODEL, help="model name as the server reports it")
    parser.add_argument("--concurrency", type=int, default=16, help="max requests in flight")
    parser.add_argument(
        "--max-attempts", type=int, default=3, help="generation attempts per product, including correction retries"
    )

    # Sampling (defaults come from GenerationParams)
    parser.add_argument(
        "--temperature", type=float, default=defaults.temperature, help="sampling temperature; 0 is greedy"
    )
    parser.add_argument("--top-p", type=float, default=defaults.top_p, help="nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=defaults.top_k, help="top-k cutoff; -1 disables it")
    parser.add_argument(
        "--repetition-penalty", type=float, default=defaults.repetition_penalty,
        help="penalty for repeated tokens; 1.0 disables it"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=defaults.max_tokens, help="answer length cap in tokens"
    )

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    products = load_jsonl(args.input)
    if args.limit is not None:
        products = products[: args.limit]
    logging.info("loaded %d products from %s", len(products), args.input)

    params = GenerationParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    client = make_client(args.base_url, api_key="EMPTY")
    request_fn = make_request_fn(client, args.model, params)

    async def process(product):
        return await generate_card(product, request_fn, max_attempts=args.max_attempts)

    results, elapsed = await run_concurrent(products, args.concurrency, process)
    logging.info("done: %d products in %.1fs", len(results), elapsed)

    cards = [r.card if r.card is not None else {"product_id": r.product_id} for r in results]
    save_jsonl(args.output, cards)

    run_info = {
        "model": args.model,
        "base_url": args.base_url,
        "input": str(args.input),
        "n": len(results),
        "concurrency": args.concurrency,
        "max_attempts": args.max_attempts,
        "params": asdict(params),  # temperature, top_p, top_k, repetition_penalty, max_tokens
        "elapsed_sec": elapsed,
        "input_tokens": sum(r.prompt_tokens for r in results),
        "output_tokens": sum(r.completion_tokens for r in results),
        "avg_latency": sum(r.latency_sec for r in results) / len(results),
        "avg_attempts": sum(r.attempts for r in results) / len(results),
        "cards": [asdict(r) for r in results],  # per-product facts for errors_by_reason etc.
    }

    args.run_info.parent.mkdir(parents=True, exist_ok=True)
    with open(args.run_info, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    logging.info("saved %s and %s", args.output, args.run_info)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))