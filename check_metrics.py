import argparse
import json
import logging
from pathlib import Path

from llm_product_cards.data import load_jsonl
from llm_product_cards.metrics import count_validity, speed, cost_own_server_per_1000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

RUN_FIELDS = (
    "model", "concurrency", "params", "elapsed_sec", "input_tokens", "output_tokens", "n", "avg_latency", "avg_attempts"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute quality, speed and cost metrics of a batch run and write report.json."
    )

    parser.add_argument(
        "--input", type=Path, default=Path("data/benchmark.jsonl"),
        help="input catalog the cards were generated for, one product per line"
    )
    parser.add_argument(
        "--predictions", type=Path, default=Path("outputs/predictions.jsonl"),
        help="generated cards from run_batch.py, one per line"
    )
    parser.add_argument(
        "--run-info", type=Path, default=Path("outputs/run.json"),
        help="run facts from run_batch.py: elapsed time, tokens, latency, attempts"
    )
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/report.json"),
        help="where to write the report; an existing api_comparison section is kept"
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    products = load_jsonl(args.input)
    cards = load_jsonl(args.predictions)
    with open(args.run_info, encoding="utf-8") as f:
        run = json.load(f)

    attempted = {c["product_id"] for c in run["cards"]}
    products = [p for p in products if p["product_id"] in attempted]

    if len(products) != run["n"]:
        logging.warning(
            "run.json reports n=%d but %d of its products are present in %s: are the files from the same run?",
            run["n"], len(products), args.input,
        )

    valid, reasons = count_validity(products, cards)
    throughput, time_for_300 = speed(run["n"], run["elapsed_sec"])
    cost = cost_own_server_per_1000(run["elapsed_sec"], run["n"])

    existing = {}
    if args.report.exists():
        with open(args.report, encoding="utf-8") as f:
            existing = json.load(f)

    report = {
        "run": {k: run[k] for k in RUN_FIELDS},
        "metrics": {
            "valid": valid,
            "valid_rate": valid / len(products) if products else 0.0,
            "throughput": throughput,
            "time_for_300_sec": time_for_300,
            "cost_vllm_per_1000": cost,
            "errors_by_reason": dict(reasons),
        },
        "api_comparison": existing.get("api_comparison", {}),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logging.info(
        "n=%d valid=%d valid_rate=%.3f throughput=%.2f cards/s time_for_300=%.0fs cost=%.1f rub/1000 -> %s",
        len(products), valid, report["metrics"]["valid_rate"], throughput, time_for_300, cost, args.report,
    )
    if reasons:
        logging.info("errors_by_reason: %s", dict(reasons.most_common()))


if __name__ == "__main__":
    main(parse_args())
