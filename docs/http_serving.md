# Resident VideoThinker HTTP service

The 2026-10-14 reproducible runtime capsule, complete model fingerprints and
restore instructions are documented in [environment freeze](environment_freeze.md).
Use its frozen Compose configuration to replay that exact baseline.

The reproducible local workload, measurement definitions and results are in
[performance baseline](performance_baseline.md).

The local service binds to `127.0.0.1:8000`. One Uvicorn process owns one
resident vLLM AsyncLLMEngine. Inputs reuse the existing manifests and worker-v5
prompt. Model and dataset directories are mounted read-only. The 6 GiB laptop
configuration uses 6 GB CPU offload and admits one active request, rejecting
additional requests with 429 instead of creating an unbounded queue.

## Start and stop

Run from the repository root with Docker Desktop's Linux GPU engine running:

The pinned base and frozen runtime images are local artifacts, not images
published to a public registry. Exact replay requires the separately retained
image archive described in the environment-freeze document; cloning this
repository alone does not supply it. The base Dockerfile records the original
build recipe. Rebuilding it can produce a different digest and requires a new
explicit image reference and baseline rather than the old frozen identity.

```powershell
# Bootstrap the local base once, if it is not already present.
docker build -f scripts/Dockerfile.vllm-validation -t videothinker-vllm-validation:0.8.5 .
docker compose -f compose.serving.yml up -d --build
docker compose -f compose.serving.yml logs -f
docker compose -f compose.serving.yml down
```

The Compose defaults point to this machine's model and data directories.
Override `VIDEO_MODEL_DIR` and `VIDEO_DATA_DIR` for another machine. Docker
restarts the container after process failure or Docker restart unless explicitly
stopped. Healthcheck failure alone does not trigger a restart; the application
now independently detects fatal engine failures and exits with code 1 so the
restart policy takes effect. Start with exactly
one Uvicorn worker; additional workers would load additional model copies.

## Requests

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
$body = @{episode='ep02'; clip_id=1; max_new_tokens=500; timeout_seconds=600} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/infer -Method Post -ContentType application/json -Body $body -TimeoutSec 660
```

`episode` is `ep01` or `ep02`, with `ep02` as the default. Clients select an
existing clip, rather than supplying arbitrary paths or image URLs. A successful
response includes a server-generated request ID, instance ID, generated text,
parsed result, token counts and elapsed time. The same instance ID and
`model_load_count=1` across requests establish model reuse within that process.

`require_json` defaults to true. Generated JSON must satisfy the complete
deterministic contract described below. This validates structure, length,
provided frame references and explicit uncertainty fields, not factual accuracy. Set
`require_json=false` only for diagnostic short generations; then truncated text
can return 200 with `structured_output_valid=false`, `result=null` and detailed
`validation_errors`. Passing responses include `validation_errors=[]`.

| Status | Meaning |
|---|---|
| 422 | Invalid request fields or bounds |
| 404 | Episode/clip is unavailable |
| 429 | One inference request is already active; retry later |
| 503 | Model loading or engine unhealthy |
| 504 | Inference deadline exceeded |
| 502 | Model output failed contract validation; see `detail.validation_errors` |
| 500 | Unexpected inference error |

Timeouts and detected client disconnects cancel the generation task and abort
the vLLM request. Abort releases scheduling state at the engine's next step;
an already running GPU kernel cannot be preempted. Timeouts range from 0.1 to
600 seconds (default 600) and include image preparation. Requests are not automatically
retried; request IDs provide traceability, not persistent idempotency.

`/health/live` checks the HTTP process. `/health/ready` checks the initialized
engine and reports active/completed/failed requests. Startup can take minutes.
A stopped/errored engine causes automatic process exit and container restart.
Repeated health-check failures also trigger recovery. Ordinary 422/429/502/504
responses and client cancellation do not themselves trigger a restart. A restart
loads a new model instance and resets in-memory counters. This is a local development service without authentication or
durable job storage; it currently binds only to localhost.

## Actual GPU acceptance checks

```powershell
& 'C:\Users\uxangy\miniconda3\envs\videothinker\python.exe' scripts/verify_http_serving.py --output-dir evaluation/runs/http_serving_new_run
```

Choose a new output directory. The script checks live/readiness, malformed
requests, missing clips, repeated short generations, strict rejection of
truncated JSON, deadline cancellation and subsequent recovery, overload
rejection, full structured output and unchanged model instance/load count.
Results are saved after each check. This is acceptance validation, not a
concurrency performance benchmark.

Final deployment defaults and real client disconnect/recovery have a separate
bounded check:

```powershell
& 'C:\Users\uxangy\miniconda3\envs\videothinker\python.exe' scripts/verify_http_disconnect.py --output-dir evaluation/runs/http_disconnect_new_run
```

## Acceptance results — 2026-10-10

The actual GPU HTTP checks passed in
`evaluation/runs/http_serving_20261010_retry/results.json`:

| Check | Result |
|---|---|
| Live/readiness | 200; readiness was 503 during model loading |
| Invalid fields / missing clip | 422 / 404 |
| Clip 1 short generation | 200, 16 tokens, 22.63 seconds |
| Strict truncated JSON | 502 |
| Two-second inference deadline | 504 |
| Clip 3 after timeout | 200, 16 tokens, 21.41 seconds |
| Extra request while busy | 429 |
| Clip 1 complete generation | 200, 346 tokens, 437.62 seconds, finish reason stop |
| Final health | 200, active 0, model load count 1, same instance |

Complete output passed the structural and metadata checks available on October 10.
The generated summary exceeds the prompt's 35-word instruction, which that version
did not enforce. The October 14 validator rejects that saved output as documented
below. Model instruction adherence and semantic quality remain separate concerns.
These laptop timings include CPU offload and are not production GPU performance
measurements. The final deployment changes only the default timeout from 300 to
600 seconds, with a new process/instance as expected after image replacement.

The final deployment check passed in
`evaluation/runs/http_disconnect_20261010/results.json`: OpenAPI default timeout
600 verified, initialization 103.38 seconds, client disconnect released admission
capacity in 0.22 seconds, and a subsequent two-token request completed in
4.60 seconds. Readiness remained 200 with active requests 0 and model load count
1. The service is left running as `videothinker-serving`.

## Automatic recovery — 2026-10-14

`src/serving/recovery.py` adds an engine health monitor that runs without client
requests, plus a separate watchdog thread. On fatal failure, readiness fails and
new inference requests are rejected. A one-second timer terminates the process
with exit code 1, abandoning the broken engine and releasing its CUDA allocations
through process teardown. Docker's existing `unless-stopped` policy then starts
a fresh process. Healthy operation still loads the model once per process.

| Setting | Default | Behavior |
|---|---|---|
| `ENGINE_CHECK_INTERVAL_SECONDS` | 5 | Autonomous engine check interval |
| `ENGINE_CHECK_TIMEOUT_SECONDS` | 5 | Deadline for an async health check |
| `ENGINE_CHECK_FAILURE_LIMIT` | 3 | Consecutive check failures before exit; a success resets the count |
| `MODEL_STARTUP_TIMEOUT_SECONDS` | 600 | Startup deadline enforced by the watchdog |
| `SERVICE_UNRESPONSIVE_TIMEOUT_SECONDS` | 60 | Maximum silence from the health loop before watchdog exit |

Initialization exceptions and vLLM stopped/errored states trigger exit directly,
without waiting for three failed checks. Transient check failures alone do not.
A fatal startup decision cannot be overwritten by a late initializer completing.
The watchdog detects an unresponsive service health loop; it is not a complete
GPU hardware diagnostic. Persistent bad paths, incompatible weights or driver
problems can still cause a restart loop and require configuration repair. Use
`docker compose -f compose.serving.yml logs --tail 100` to inspect the reason.
An explicit Compose stop/down remains stopped as requested.

In-flight requests may receive an error or lose their connection when the engine
dies. There is no automatic replay, durable result storage or exactly-once
guarantee. Clients decide whether to retry after readiness recovers. This change
does not alter model output validation or model quality.

### Recovery validation

The final policy suite passed 10 tests, including an unresponsive asyncio loop,
health-check deadlines, startup deadline, late-initializer race, transient
failure recovery and normal shutdown:

```powershell
docker run --rm --entrypoint python videothinker-serving:local -m unittest src.tests.test_serving_recovery -v
```

The isolated actual-GPU test used the normal application and a test-only external
launcher, mounted read-only. It injected one missing-model initialization failure
and one vLLM errored state while a request was running. The launcher is not in
the serving image and exposes no HTTP fault endpoint. This does not constitute
an actual CUDA OOM or driver-crash test.

| Real GPU check | Result |
|---|---|
| Initialization failure | Nonzero exit, automatic restart, readiness recovered |
| Inference after initialization recovery | HTTP 200, 2 output tokens |
| Runtime engine failure | Detected by autonomous monitor, nonzero exit, automatic restart |
| Runtime recovery time | About 145 seconds including model reload |
| Interrupted request | Connection closed; no success response |
| Inference after runtime recovery | HTTP 200, new instance ID, load count 1 |
| Ordinary timeout / invalid output / invalid input | 504 / 502 / 422, same instance remains ready |
| Docker restart count | Exactly 2 |

Measurements and container logs are in
`evaluation/runs/serving_recovery_20261014/http_checks_corrected/results.json`,
`corrected_container.log` and `restart_count.txt` under the same run folder.
An initial test omitted `MODEL_PATH=/model` and inherited the base image's
`/work/model`, producing a repeated configuration failure. That run was aborted
and its logs retained as `wrong_model_path.log`; the corrected run above explicitly
sets both model and data roots. The final revision adds the startup-deadline race
guard, covered by the final 10-test suite and a separate deployment smoke check.

Reproduce in a quiet maintenance window with a fresh output directory:

```powershell
./scripts/run_serving_recovery.ps1 -PythonExe 'C:\Users\uxangy\miniconda3\envs\videothinker\python.exe'
```

The runner temporarily stops the existing service to free the 6 GiB GPU, saves
test evidence, removes its test container, and restores a previously running
service in a `finally` block. It does not wait for that restored service to finish
loading; check `/health/ready`. The PowerShell runner has been syntax checked;
this session executed its constituent Docker and Python test steps directly.

The final normal-entrypoint deployment also passed the real disconnect/recovery
smoke check in
`evaluation/runs/serving_recovery_20261014/final_deployment/results.json`:
initialization 134.77 seconds, disconnect admission release 0.25 seconds, next
two-token generation 6.54 seconds, unchanged instance ID and model load count 1.
The normal service was restored; the isolated fault-injection container was removed.

## Output validation — 2026-10-14

`src/serving/output_validation.py` now enforces these rules without rewriting
model text or silently retrying generation:

- Parse a single JSON object. Reject malformed/truncated JSON, markdown fences,
  duplicate object keys and non-finite JSON numbers.
- Require the exact declared fields and strict types at both clip and event
  levels; reject extra fields, invalid enums and blank required text/list items.
- Match clip ID, filename and start/end timestamps to the source record.
- Limit summary to 35 words and each event's visual evidence to 20 words.
  Words are nonempty whitespace-separated tokens; hyphenated words and
  contractions each count as one token. Text is not trimmed or rewritten.
- Allow zero to three events. Every event must provide one to three unique
  references to actual input frames. The existing prompt labels frames starting
  at **Frame 00**: eight frames permit Frame 00 through Frame 07. Fewer input
  frames narrow the valid set. Labels must match exactly.
- Low-confidence events require nonempty `uncertain_parts` and
  `possible_hallucination_risks`. Any explicitly listed uncertain part requires
  at least one risk. Low visual information alone is not treated as uncertainty.

This does not verify that the described event actually occurred, that cited
frames prove it, that all implicit uncertainties are listed, or that prose is
English. Those require separate model-quality evaluation. These rules apply to
the HTTP service; the original AWS Batch worker's output handling is unchanged.

Strict rejection returns HTTP 502 with a request ID, finish reason and issues:

```json
{
  "detail": {
    "error": "Model output failed validation",
    "request_id": "server-generated-id",
    "finish_reason": "stop",
    "validation_errors": [
      {"path": ["summary"], "code": "word_limit", "message": "Summary has 38 words; maximum is 35"}
    ]
  }
}
```

`path` identifies the field, including event/list indices when relevant. Once
schema validation passes, all detected contract issues are returned together.
Schema errors are reported first if field types/shape are invalid. Diagnostic
mode (`require_json=false`) returns the unmodified `text` with the same issues,
`result=null` and `structured_output_valid=false`. Validation failures never
trigger engine recovery by themselves.

The 28-test suite passed: 15 output boundary tests, 3 actual FastAPI route tests
with synthetic model outputs, and 10 recovery regressions. The route tests also
exposed and verified a fix for a cleanup race: Starlette's cancellation-scoped
disconnect probe could absorb task cancellation. An explicit stop event now
ends the watcher after a response, releasing the admission slot.

```powershell
docker run --rm --entrypoint python videothinker-serving:local -m unittest src.tests.test_output_validation src.tests.test_serving_output_api src.tests.test_serving_recovery -v
```

Revalidation of the saved real complete output found exactly the known summary
violation (38 > 35), with valid metadata and frame references. Evidence is in
`evaluation/runs/output_validation_20261014/saved_outputs_audit.json`. This audit
does not regenerate or modify the old result. `scripts/audit_serving_outputs.py`
supports repeating this check against saved HTTP checks and their source manifest.

The deployed real-GPU checks also passed:

- `evaluation/runs/output_validation_20261014/final_deployment/results.json`:
  readiness recovered, client disconnect released admission in 0.22 seconds,
  subsequent two-token generation succeeded, same instance and model load count 1.
- `evaluation/runs/output_validation_20261014/http_errors/results.json`:
  diagnostic truncated generation returned 200 with `result=null` and
  `invalid_json` details; strict truncated generation returned 502 with the same
  issue structure; readiness remained healthy with no model reload.

```powershell
& 'C:\Users\uxangy\miniconda3\envs\videothinker\python.exe' scripts/verify_output_validation_http.py --output-dir evaluation/runs/output_validation_http_new_run
```

This request-level GPU check uses one-token generations to exercise the failure
path. Passing complete-output HTTP behavior is covered with synthetic outputs;
no new full-length GPU generation or multi-clip acceptance-rate study was performed
in this validation change. The previously saved complete model output fails the
new contract, so improved generation adherence remains follow-up work.
