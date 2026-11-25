"""The ``mtl5-bench`` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ingest import ingest


def _cmd_ingest(args: argparse.Namespace) -> int:
    con, report = ingest(
        args.root, args.db, rebuild=args.rebuild, time_from_git=not args.no_git_time
    )
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
        m_runs, m_values, m_series = con.execute(
            "SELECT count(DISTINCT run_id), count(*), count(DISTINCT series_key) "
            "FROM trend_measurements"
        ).fetchone()
        print(
            f"\ntrend-eligible: {n_runs} runs, {n_results} results, {n_series} series "
            f"(database: {args.db})"
        )
        print(f"trend-eligible legacy: {m_runs} runs, {m_values} measurements, {m_series} series")
    finally:
        con.close()
    return 0


def _cmd_figure(args: argparse.Namespace) -> int:
    from . import analyze, plot

    tiers = ("complete", "incomplete") if args.include_incomplete else ("complete",)
    db = args.db
    if not Path(db).exists():
        print(f"mtl5-bench: no database {db}; run 'mtl5-bench ingest' first", file=sys.stderr)
        return 2
    try:
        if args.kind == "card":
            df = analyze.capability(
                db,
                args.machine,
                label=args.label,
                threads=args.threads,
                run_id=args.run,
                tiers=tiers,
            )
            fig = plot.capability_card(df)
        elif args.kind == "roofline":
            df = analyze.roofline(
                db,
                args.machine,
                label=args.label,
                threads=args.threads,
                run_id=args.run,
                areas=args.area,
                tiers=tiers,
            )
            fig = plot.roofline(df, label_entries=args.entry or ())
        elif args.kind == "scaling":
            df = analyze.thread_scaling(
                db,
                args.machine,
                label=args.label,
                entries=args.entry or SCALING_ENTRIES,
                tiers=tiers,
            )
            fig = plot.thread_scaling(df)
        else:
            df = analyze.rate_vs_size(
                db,
                args.machine,
                operations=args.operation,
                backends=args.backend,
                metric=args.metric,
                tiers=tiers,
            )
            if df.empty and not args.include_incomplete:
                raise LookupError(
                    "no complete bench_all sweep matches; most sweeps predate the "
                    "sidecar rules, and --include-incomplete reads them"
                )
            fig = plot.rate_vs_size(df)
    except (LookupError, ValueError) as exc:
        print(f"mtl5-bench: {exc}", file=sys.stderr)
        return 1
    out = args.out or f"{args.kind}-{args.machine or 'all'}"
    for path in fig.save(out, args.format.split(","), data=not args.no_data):
        print(path)
    return 0


#: The entries a scaling figure shows when none are named.
SCALING_ENTRIES = (
    "blas1/dot/f64",
    "blas1/axpy/f64",
    "blas2/gemv/f64",
    "blas3/gemm/f64",
    "blas3/gemm/f32",
    "sparse/spmv/f64",
    "factor/lu/f64",
)


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
            "every file: mtl5-bench/1 JSON, MTL5's CSV formats and bench_blas.py "
            "JSON are ingested with their .sysinfo sidecars, and everything else is "
            "recorded with the reason it was not."
        ),
    )
    p.add_argument("root", help="the benchmark data directory")
    p.add_argument("--db", default="bench.duckdb", help="database file (default: %(default)s)")
    p.add_argument("--rebuild", action="store_true", help="delete the database and start over")
    p.add_argument("-v", "--verbose", action="store_true", help="list every file with a reason")
    p.add_argument(
        "--no-git-time",
        action="store_true",
        help="do not date files outside a timestamp directory by the commit that added them",
    )
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser(
        "figure",
        help="draw a figure from the database",
        description=(
            "Draw one of the core figures and write it as PDF, SVG and PNG, with "
            "the DataFrame it was drawn from as CSV. card: a machine's capability "
            "card. roofline: one run's entries on the roofline. scaling: speedup "
            "against threads. rate: a rate against problem size, from the "
            "bench_all sweeps."
        ),
    )
    p.add_argument("kind", choices=["card", "roofline", "scaling", "rate"])
    p.add_argument("--db", default="bench.duckdb", help="database file (default: %(default)s)")
    p.add_argument(
        "--machine", help="machine label, data directory or fingerprint (rate: optional)"
    )
    p.add_argument("--label", default="native", help="the run's label (default: %(default)s)")
    p.add_argument("--threads", type=int, help="the run's thread count (card, roofline)")
    p.add_argument("--run", help="a run id, instead of the latest matching run")
    p.add_argument("--area", action="append", help="roofline: keep this area (repeatable)")
    p.add_argument(
        "--entry",
        action="append",
        help="scaling: entries to show; roofline: entries to label (id or prefix, repeatable)",
    )
    p.add_argument("--operation", action="append", help="rate: keep this operation (repeatable)")
    p.add_argument("--backend", action="append", help="rate: keep this backend (repeatable)")
    p.add_argument("--metric", default="gflops", help="rate: the metric (default: %(default)s)")
    p.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also read runs whose provenance is incomplete (the footer says so)",
    )
    p.add_argument("--out", help="output path without suffix (default: KIND-MACHINE)")
    p.add_argument("--format", default="pdf,svg,png", help="formats (default: %(default)s)")
    p.add_argument("--no-data", action="store_true", help="do not write the CSV")
    p.set_defaults(func=_cmd_figure)

    args = parser.parse_args(argv)
    if args.command == "figure" and args.kind != "rate" and not args.machine:
        parser.error(f"figure {args.kind} needs --machine")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
