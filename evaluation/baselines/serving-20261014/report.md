# Local HTTP performance baseline

Latency quantiles use HTTP 200 responses only. Small successful samples are exploratory, not SLO evidence.

| Mode | Concurrency | Success/total | P50 s | P95 s | P99 s | Success req/s | Output tok/s | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnostic | 1 | 3/3 | 20.736 | 20.745 | 20.745 | 0.0482 | 0.772 | 0.0% |
| diagnostic | 2 | 3/6 | 20.775 | 20.824 | 20.829 | 0.0481 | 0.770 | 50.0% |
| diagnostic | 4 | 3/12 | 20.808 | 20.816 | 20.816 | 0.0481 | 0.769 | 75.0% |
| diagnostic | 8 | 3/24 | 20.878 | 20.910 | 20.912 | 0.0479 | 0.767 | 87.5% |
| diagnostic | 16 | 3/48 | 21.922 | 22.739 | 22.811 | 0.0457 | 0.731 | 93.8% |
| strict | 1 | 0/3 | N/A | N/A | N/A | 0.0000 | 0.000 | 100.0% |

Diagnostic requests are capped at 16 tokens by default and do not establish complete-output quality.

## Complete-output requests

| Clip | HTTP status | Client seconds | Contract validation |
|---|---:|---:|---|
| 1 | 502 | 432.823 | Summary has 38 words; maximum is 35 |
| 3 | 502 | 525.770 | Summary has 57 words; maximum is 35; Reference must be one of: Frame 00, Frame 01, Frame 02, Frame 03, Frame 04, Frame 05, Frame 06, Frame 07; Reference must be one of: Frame 00, Frame 01, Frame 02, Frame 03, Frame 04, Frame 05, Frame 06, Frame 07; Reference must be one of: Frame 00, Frame 01, Frame 02, Frame 03, Frame 04, Frame 05, Frame 06, Frame 07 |
| 14 | 502 | 595.870 | Summary has 74 words; maximum is 35 |

HTTP 502 responses expose validation issues but no generated token counts; failed-generation throughput is unavailable.

## Runtime checks

Measurement status: measured. Stable instance: True.

GPU summary includes warmup and whole-device activity:

```json
{
  "valid_samples": 921,
  "mean_utilization_percent": 70.04234527687296,
  "peak_used_memory_mib": 5168.0
}
```

Published request status/latency, frame fingerprints and metric inputs: [results.json](results.json). Runtime verification: [environment-verification.json](environment-verification.json). Measurement client: [client-used.py](client-used.py). Full raw responses and GPU logs remain local under the ignored evaluation/runs directory. Frozen configuration: [environment capsule](../../../environment/serving-20261014/manifest.json).
