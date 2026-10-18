# Results

## test_in (n=150)

| system | category acc | macro-F1 | merchant exact | self-transfer F1 | refund F1 | valid JSON | input tok/row | p50 latency | $/1k rows |
|---|---|---|---|---|---|---|---|---|---|
| local | 92.7% | 0.862 | 82.7% | 0.933 | 0.000 | 99.3% | 77 | 1956 ms | - |

## test_unseen (n=57)

| system | category acc | macro-F1 | merchant exact | self-transfer F1 | refund F1 | valid JSON | input tok/row | p50 latency | $/1k rows |
|---|---|---|---|---|---|---|---|---|---|
| local | 86.0% | 0.824 | 77.2% | 1.000 | 0.000 | 100.0% | 75 | 1989 ms | - |

Latency for Tinker systems is remote API latency; `local` is on-device.
Cost uses Tinker sampling prices; the local model costs nothing per row.