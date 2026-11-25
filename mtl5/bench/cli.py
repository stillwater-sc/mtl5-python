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
    from .style import check_formats

    tiers = ("complete", "incomplete") if args.include_incomplete else ("complete",)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    try:
        check_formats(formats)
    except ValueError as exc:
        print(f"mtl5-bench: {exc}", file=sys.stderr)
        return 2
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
                threads=_threads(args.threads, None),
                run_id=args.run,
                tiers=tiers,
            )
            fig = plot.capability_card(df)
        elif args.kind == "roofline":
            df = analyze.roofline(
                db,
                args.machine,
                label=args.label,
                threads=_threads(args.threads, 1),
                run_id=args.run,
                areas=args.area,
                tiers=tiers,
            )
            fig = plot.roofline(df, label_entries=args.entry)
        elif args.kind == "scaling":
            df = analyze.thread_scaling(
                db,
                args.machine,
                label=args.label,
                entries=args.entry or SCALING_ENTRIES,
                tiers=tiers,
            )
            fig = plot.thread_scaling(df)
        elif args.kind == "balance":
            df = analyze.machines(
                db,
                label=args.label,
                threads=_threads(args.threads, "max"),
                machines=_machine_list(args.machines),
                tiers=tiers,
            )
            fig = plot.machine_balance(df)
        elif args.kind == "compare":
            df = _compare_frame(analyze, args, tiers)
            fig = plot.cross_machine(df)
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
    try:
        paths = fig.save(out, formats, data=not args.no_data)
    except (OSError, ValueError) as exc:
        print(f"mtl5-bench: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


def _threads(value: str | None, default):
    """``--threads``: an integer, ``max`` (the cross-machine kinds), or the default."""
    if value is None:
        return default
    if value == "max":
        if default != "max":
            raise ValueError("--threads max applies to the cross-machine figures and tables")
        return value
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"--threads must be a positive integer or max, not {value!r}") from None
    if n < 1:
        raise ValueError(f"--threads must be a positive integer or max, not {value!r}")
    return n


def _machine_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [m.strip() for m in value.split(",") if m.strip()] or None


def _compare_frame(analyze, args, tiers):
    df = analyze.cross_machine(
        args.db,
        args.entry or SCALING_ENTRIES,
        metric=args.metric,
        label=args.label,
        threads=_threads(args.threads, "max"),
        machines=_machine_list(args.machines),
        tiers=tiers,
    )
    if df.empty:
        raise LookupError("no results match the entries on these machines")
    return df


def _cmd_table(args: argparse.Namespace) -> int:
    from . import analyze, plot, tables

    tiers = ("complete", "incomplete") if args.include_incomplete else ("complete",)
    if not Path(args.db).exists():
        print(f"mtl5-bench: no database {args.db}; run 'mtl5-bench ingest' first", file=sys.stderr)
        return 2
    try:
        if args.kind == "machines":
            df = analyze.machines(
                args.db,
                label=args.label,
                threads=_threads(args.threads, "max"),
                machines=_machine_list(args.machines),
                tiers=tiers,
            )
            table = tables.machine_table(df)
        else:
            df = _compare_frame(analyze, args, tiers)
            table = tables.compare_table(df, spread=args.spread and args.format == "md")
    except (LookupError, ValueError) as exc:
        print(f"mtl5-bench: {exc}", file=sys.stderr)
        return 1
    if args.format == "md":
        text = tables.to_markdown(table, footer=plot.provenance(df))
    else:
        text = table.to_csv(index=False)
    if args.out:
        try:
            # UTF-8 everywhere: the provenance line has non-ASCII separators.
            Path(args.out).write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"mtl5-bench: {exc}", file=sys.stderr)
            return 1
        print(args.out)
    else:
        sys.stdout.write(text)
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
            "bench_all sweeps. balance: FMA peak against bandwidth across machines. "
            "compare: entries across machines."
        ),
    )
    p.add_argument("kind", choices=["card", "roofline", "scaling", "rate", "balance", "compare"])
    p.add_argument("--db", default="bench.duckdb", help="database file (default: %(default)s)")
    p.add_argument(
        "--machine", help="machine label, data directory or fingerprint (rate: optional)"
    )
    p.add_argument("--label", default="native", help="the run's label (default: %(default)s)")
    p.add_argument(
        "--threads",
        help="the run's thread count (card, roofline); balance and compare: an integer "
        "or max, each machine's most (default)",
    )
    p.add_argument(
        "--machines", help="balance, compare: the machines to show, comma-separated (all)"
    )
    p.add_argument("--run", help="a run id, instead of the latest matching run")
    p.add_argument("--area", action="append", help="roofline: keep this area (repeatable)")
    p.add_argument(
        "--entry",
        action="append",
        help="scaling, compare: entries to show; roofline: entries to label "
        "(id or prefix, repeatable)",
    )
    p.add_argument("--operation", action="append", help="rate: keep this operation (repeatable)")
    p.add_argument("--backend", action="append", help="rate: keep this backend (repeatable)")
    p.add_argument(
        "--metric",
        default="gflops",
        help="rate, compare: the metric (default: %(default)s; compare also takes "
        "roofline_pct, gbytes_per_s, median_s)",
    )
    p.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also read runs whose provenance is incomplete (the footer says so)",
    )
    p.add_argument("--out", help="output path without suffix (default: KIND-MACHINE)")
    p.add_argument("--format", default="pdf,svg,png", help="formats (default: %(default)s)")
    p.add_argument("--no-data", action="store_true", help="do not write the CSV")
    p.set_defaults(func=_cmd_figure)

    p = sub.add_parser(
        "table",
        help="write a comparison table across machines",
        description=(
            "machines: one row per machine, the median of its runs' probes. "
            "compare: entries by machines, each machine's median over its runs. "
            "A machine's runs are its latest commit's sessions at --threads."
        ),
    )
    p.add_argument("kind", choices=["machines", "compare"])
    p.add_argument("--db", default="bench.duckdb", help="database file (default: %(default)s)")
    p.add_argument("--label", default="native", help="the runs' label (default: %(default)s)")
    p.add_argument("--threads", help="an integer, or max: each machine's most (default)")
    p.add_argument("--machines", help="the machines to show, comma-separated (all)")
    p.add_argument(
        "--entry", action="append", help="compare: entries to show (id or prefix, repeatable)"
    )
    p.add_argument(
        "--metric",
        default="gflops",
        help="compare: gflops (default), roofline_pct, gbytes_per_s or median_s",
    )
    p.add_argument(
        "--spread", action="store_true", help="compare, md: write median (min–max) per cell"
    )
    p.add_argument(
        "--include-incomplete",
        action="store_true",
        help="also read runs whose provenance is incomplete (the footer says so)",
    )
    p.add_argument("--format", choices=["md", "csv"], default="md", help="(default: md)")
    p.add_argument("--out", help="write to this file instead of standard output")
    p.set_defaults(func=_cmd_table)

    args = parser.parse_args(argv)
    if args.command == "figure" and args.kind in ("balance", "compare") and args.machine:
        parser.error(f"figure {args.kind} compares machines: select them with --machines")
    if (
        args.command == "figure"
        and args.kind not in ("rate", "balance", "compare")
        and not args.machine
    ):
        parser.error(f"figure {args.kind} needs --machine")
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
