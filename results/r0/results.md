# Results

## test_in (n=150)

| system | category acc | macro-F1 | merchant exact | self-transfer F1 | refund F1 | valid JSON | input tok/row | p50 latency | $/1k rows |
|---|---|---|---|---|---|---|---|---|---|
| rules | 65.3% | 0.369 | 72.0% | 0.560 | 0.000 | 100.0% | 0 | 0 ms | - |
| finetuned | 94.0% | 0.934 | 77.3% | 0.778 | 0.000 | 100.0% | 77 | 2725 ms | $0.056 |

## test_unseen (n=57)

| system | category acc | macro-F1 | merchant exact | self-transfer F1 | refund F1 | valid JSON | input tok/row | p50 latency | $/1k rows |
|---|---|---|---|---|---|---|---|---|---|
| rules | 75.4% | 0.683 | 73.7% | 1.000 | 0.000 | 100.0% | 0 | 0 ms | - |
| finetuned | 82.5% | 0.773 | 73.7% | 0.667 | 0.000 | 100.0% | 75 | 5192 ms | $0.055 |

Latency for Tinker systems is remote API latency; `local` is on-device.
Cost uses Tinker sampling prices; the local model costs nothing per row.