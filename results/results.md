# Results

## test_in (n=150)

| system | category acc | macro-F1 | merchant exact | self-transfer F1 | refund F1 | valid JSON | input tok/row | p50 latency | $/1k rows |
|---|---|---|---|---|---|---|---|---|---|
| rules | 65.3% | 0.369 | 72.0% | 0.560 | 0.000 | 100.0% | 0 | 0 ms | - |
| student-few | 78.7% | 0.646 | 71.3% | 0.842 | 0.000 | 99.3% | 1360 | 2619 ms | $0.479 |
| teacher-few | 86.7% | 0.820 | 78.7% | 0.778 | 0.000 | 100.0% | 1360 | 2371 ms | $2.472 |
| finetuned | 92.7% | 0.828 | 82.0% | 0.933 | 0.000 | 99.3% | 77 | 3384 ms | $0.056 |

## test_unseen (n=57)

| system | category acc | macro-F1 | merchant exact | self-transfer F1 | refund F1 | valid JSON | input tok/row | p50 latency | $/1k rows |
|---|---|---|---|---|---|---|---|---|---|
| rules | 75.4% | 0.683 | 73.7% | 1.000 | 0.000 | 100.0% | 0 | 0 ms | - |
| student-few | 80.7% | 0.745 | 73.7% | 1.000 | 0.000 | 100.0% | 1358 | 5497 ms | $0.479 |
| teacher-few | 87.7% | 0.847 | 77.2% | 1.000 | 0.000 | 100.0% | 1358 | 3755 ms | $2.469 |
| finetuned | 87.7% | 0.844 | 77.2% | 1.000 | 0.000 | 100.0% | 75 | 4564 ms | $0.055 |

Latency for Tinker systems is remote API latency; `local` is on-device.
Cost uses Tinker sampling prices; the local model costs nothing per row.