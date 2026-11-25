| Machine | CPU | Arch | Threads | FMA unit | f64 peak, GFLOP/s | f32 peak, GFLOP/s | Bandwidth, GB/s | f64 ridge, FLOP/byte | LLC, MiB | Runs | Commit |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| arduino-ventuno-q | 0xd4b | aarch64 | 4 | NEON | 67.7 | 136 | 24.6 | 2.75 |  | 1 | bfa1c28 |
| i7-12700k | 12th Gen Intel(R) Core(TM) i7-12700K | x86_64 | 8 | AVX2+FMA | 601 | 1202 | 83.8 | 7.17 | 25 | 2 | bfa1c28 |
| jetson-orin | ARMv8 Processor rev 1 (v8l) | aarch64 | 6 | NEON | 64.3 | 129 | 34.6 | 1.86 | 2 | 2 | bfa1c28 |

*machines arduino-ventuno-q, i7-12700k, jetson-orin · commit bfa1c28 · 5 runs, 2026-11-11 to 2026-11-12*
