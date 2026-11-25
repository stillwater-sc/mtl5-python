"""The ``mtl5-bench`` command line."""

from __future__ import annotations

import argparse
import sys

from .ingest import ingest


def _cmd_ingest(args: argparse.Namespace) -> int:
    con, report = ingest(args.root, args.db, rebuild=args.rebuild)
    try:
        print(report)
        problems = report.problems()
        if problems and (args.verbose or len(problems) <= 20):
            print("\nrejected and incomplete files:")
            for o in problems:
                print(f"  {o.status:<10} {o.path}: {'; '.join(o.reasons)}")
        elif problems:
            print(f"\n{len(problems)} rejected or incomplete files; -v lists them")
        if args.verbose:
            for o in report.outcomes:
                if o.status in ("deferred", "ignored", "removed", "duplicate") and o.reasons:
                    print(f"  {o.status:<10} {o.path}: {'; '.join(o.reasons)}")
        n_runs, n_results, n_series = con.execute(
            "SELECT count(DISTINCT run_id), count(*), count(DISTINCT series_key) FROM trend_results"
        ).fetchone()
        print(
            f"\ntrend-eligible: {n_runs} runs, {n_results} results, {n_series} series "
            f"(database: {args.db})"
        )
    finally:
        con.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mtl5-bench",
        description="Ingest and analyse MTL5 benchmark data (mtl5.bench).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "ingest",
        help="ingest a benchmark data tree into a DuckDB database",
        description=(
            "Walk ROOT (normally <mtl5 checkout>/benchmarks/data) and account for "
            "every file: mtl5-bench/1 JSON runs are ingested with their .sysinfo "
            "sidecars, and everything else is recorded with the reason it was not."
        ),
    )
    p.add_argument("root", help="the benchmark data directory")
    p.add_argument("--db", default="bench.duckdb", help="database file (default: %(default)s)")
    p.add_argument("--rebuild", action="store_true", help="delete the database and start over")
    p.add_argument("-v", "--verbose", action="store_true", help="list every file with a reason")
    p.set_defaults(func=_cmd_ingest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
