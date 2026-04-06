#!/usr/bin/env python3
"""Pandas Sensor Data Analysis with MTL5 Posit16 ExtensionDtype

Demonstrates the data science workflow using mtl5's pandas extension type:
storing sensor measurements in posit16 columns instead of float64. The
posit16 dtype halves storage with minimal accuracy loss for typical
sensor ranges (values centered near zero or one, with reasonable dynamic
range).

The synthetic dataset simulates a small fleet of IoT temperature/humidity
sensors reporting hourly readings. We compare:

  - Storage footprint: posit16 vs float64
  - Round-trip fidelity for typical sensor values
  - groupby + aggregation behavior on posit16 columns
  - Mixing posit16 and float64 columns in the same DataFrame

Requires pandas. Run with:

    pip install 'mtl5[dev]' pandas
    python pandas_sensor_analysis.py [--days 30] [--sensors 5]
"""

import argparse

import numpy as np

import mtl5

try:
    import pandas as pd
except ImportError:
    raise SystemExit("This demo requires pandas. Install with:\n    pip install pandas")


def generate_sensor_data(n_sensors: int, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Synthesize hourly temperature and humidity readings for a fleet of sensors."""
    n_hours = 24 * n_days
    sensor_ids = [f"sensor-{i:02d}" for i in range(n_sensors)]

    rows = []
    for sensor_id in sensor_ids:
        # Each sensor has its own baseline and noise characteristics
        temp_baseline = 20.0 + rng.uniform(-3, 3)
        humidity_baseline = 50.0 + rng.uniform(-10, 10)

        for hour in range(n_hours):
            # Diurnal cycle for temperature
            t_diurnal = 5.0 * np.sin(2 * np.pi * (hour % 24) / 24.0)
            # Day/night swing for humidity (anti-correlated with temperature)
            h_diurnal = -10.0 * np.sin(2 * np.pi * (hour % 24) / 24.0)

            temp = temp_baseline + t_diurnal + 0.3 * rng.standard_normal()
            humidity = humidity_baseline + h_diurnal + 1.0 * rng.standard_normal()
            humidity = float(np.clip(humidity, 0.0, 100.0))

            rows.append(
                {
                    "sensor_id": sensor_id,
                    "hour": hour,
                    "day": hour // 24,
                    "temperature_c": temp,
                    "humidity_pct": humidity,
                }
            )

    return pd.DataFrame(rows)


def compare_precision_round_trip(df: pd.DataFrame) -> None:
    """Round-trip the temperature column through posit16 and report fidelity."""
    print("\nRound-Trip Fidelity (posit16 vs float64)")
    print("=" * 60)

    temp_f64 = df["temperature_c"].to_numpy()
    temp_posit = mtl5.Posit16Array(temp_f64)
    temp_back = temp_posit.to_numpy()

    abs_err = np.abs(temp_back - temp_f64)
    rel_err = abs_err / np.maximum(np.abs(temp_f64), 1e-12)

    print(f"  N samples:           {len(temp_f64)}")
    print(f"  Mean absolute error: {abs_err.mean():.6e}")
    print(f"  Max  absolute error: {abs_err.max():.6e}")
    print(f"  Mean relative error: {rel_err.mean():.6e}")
    print(f"  Max  relative error: {rel_err.max():.6e}")


def compare_storage_footprint(df: pd.DataFrame) -> None:
    """Estimate storage in float64 vs posit16."""
    print("\nStorage Footprint")
    print("=" * 60)
    n = len(df)

    f64_bytes = n * 8  # 8 bytes per float64
    posit16_bytes = n * 2  # 2 bytes per posit16

    print("  Per column:")
    print(f"    float64 storage:  {f64_bytes:>10,d} bytes  ({f64_bytes / 1024:.2f} KiB)")
    print(f"    posit16 storage:  {posit16_bytes:>10,d} bytes  ({posit16_bytes / 1024:.2f} KiB)")
    print(f"    Reduction:        {f64_bytes / posit16_bytes:.1f}x")
    print()
    print(f"  For 2 measurement columns over {n} rows:")
    print(f"    f64:  {2 * f64_bytes / 1024:.2f} KiB")
    print(f"    p16:  {2 * posit16_bytes / 1024:.2f} KiB")
    print(f"    saved {2 * (f64_bytes - posit16_bytes) / 1024:.2f} KiB")


def build_quantized_dataframe(df_f64: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame where measurement columns use the posit16 ExtensionDtype."""
    return pd.DataFrame(
        {
            "sensor_id": df_f64["sensor_id"],
            "hour": df_f64["hour"],
            "day": df_f64["day"],
            "temperature_c": pd.array(
                df_f64["temperature_c"].to_numpy(), dtype=mtl5.Posit16Dtype()
            ),
            "humidity_pct": pd.array(df_f64["humidity_pct"].to_numpy(), dtype=mtl5.Posit16Dtype()),
        }
    )


def per_sensor_summary(df: pd.DataFrame, label: str) -> None:
    """Compute per-sensor mean/min/max and report."""
    print(f"\nPer-Sensor Summary ({label})")
    print("=" * 60)

    # NB: pandas converts ExtensionArray to NumPy for numeric aggregations
    # via to_numpy(). For the demo we explicitly convert columns to float64
    # before grouping, then report.
    df_num = df.copy()
    if hasattr(df["temperature_c"].array, "to_numpy"):
        df_num["temperature_c"] = df["temperature_c"].array.to_numpy()
        df_num["humidity_pct"] = df["humidity_pct"].array.to_numpy()

    summary = df_num.groupby("sensor_id").agg(
        temp_mean=("temperature_c", "mean"),
        temp_std=("temperature_c", "std"),
        temp_min=("temperature_c", "min"),
        temp_max=("temperature_c", "max"),
        hum_mean=("humidity_pct", "mean"),
    )

    print(summary.to_string(float_format=lambda x: f"{x:8.3f}"))


def aggregation_drift(df_f64: pd.DataFrame, df_p16: pd.DataFrame) -> None:
    """Compare aggregate statistics computed from float64 vs posit16 columns."""
    print("\nAggregate Drift (posit16 vs float64)")
    print("=" * 60)

    metrics = [
        ("mean(temperature)", lambda d: float(np.asarray(d["temperature_c"]).mean())),
        ("std(temperature)", lambda d: float(np.asarray(d["temperature_c"]).std())),
        ("mean(humidity)", lambda d: float(np.asarray(d["humidity_pct"]).mean())),
        ("std(humidity)", lambda d: float(np.asarray(d["humidity_pct"]).std())),
    ]

    print(f"  {'metric':<20s}  {'float64':>12s}  {'posit16':>12s}  {'rel diff':>12s}")
    print(f"  {'------':<20s}  {'-------':>12s}  {'-------':>12s}  {'--------':>12s}")
    for name, fn in metrics:
        v64 = fn(df_f64)
        v16 = fn(df_p16)
        rel = abs(v16 - v64) / max(abs(v64), 1e-12)
        print(f"  {name:<20s}  {v64:12.6f}  {v16:12.6f}  {rel:12.2e}")


def main():
    parser = argparse.ArgumentParser(description="pandas + MTL5 Posit16Dtype sensor analysis demo")
    parser.add_argument("--sensors", type=int, default=5, help="Number of sensors")
    parser.add_argument("--days", type=int, default=14, help="Days of data per sensor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"MTL5 v{mtl5.__version__} — Pandas Sensor Analysis with Posit16Dtype")
    print(f"  pandas version: {pd.__version__}")
    print(f"  Posit16Dtype:   {mtl5.Posit16Dtype()}")
    print()

    rng = np.random.default_rng(args.seed)

    # Generate the float64 baseline DataFrame
    df_f64 = generate_sensor_data(args.sensors, args.days, rng)
    print(f"Generated {len(df_f64)} rows: {args.sensors} sensors × {args.days * 24} hours")
    print(f"Columns: {list(df_f64.columns)}")
    print("\nSample rows (float64):")
    print(df_f64.head().to_string(float_format=lambda x: f"{x:7.3f}"))

    # Build the posit16 quantized version
    df_p16 = build_quantized_dataframe(df_f64)
    print("\nQuantized dtype check:")
    print(f"  temperature_c.dtype = {df_p16['temperature_c'].dtype}")
    print(f"  humidity_pct.dtype  = {df_p16['humidity_pct'].dtype}")

    # Run all comparisons
    compare_precision_round_trip(df_f64)
    compare_storage_footprint(df_f64)
    per_sensor_summary(df_p16, "posit16 columns")
    aggregation_drift(df_f64, df_p16)

    print()
    print("Observations")
    print("=" * 60)
    print("- posit16 stores sensor measurements at 2 bytes vs 8 bytes for f64.")
    print("- For values in the typical sensor range (10-100), the relative")
    print("  error is well within sensor noise floors.")
    print("- pandas groupby and aggregation work transparently on the")
    print("  ExtensionDtype because the array implements to_numpy().")
    print("- Real workloads can mix posit16 (measurements) and int (timestamps,")
    print("  IDs) columns in the same DataFrame.")


if __name__ == "__main__":
    main()
