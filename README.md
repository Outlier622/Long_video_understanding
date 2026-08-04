# Long-Video Inference Optimization

A long-video inference pipeline built on top of **VideoThinker-R1-3B**. The project improves frame selection, adds bounded cross-clip context, and supports resumable GPU inference on AWS.

This repository focuses on **inference optimization and systems engineering**. It does not retrain the base model.

---

## Overview

Uniform frame sampling can waste a fixed frame budget on low-information moments, miss short-lived motion events, and provide weak continuity across adjacent clips.

This project addresses those limitations through three components:

1. **Selective keyframe sampling**
2. **Bounded cross-clip context**
3. **Resumable cloud-backed inference**

The base model is **VideoThinker-R1-3B**, derived from the Qwen2.5-VL model family.

---

## Key Results

### Selective Keyframe Sampling

The final sampler combines:

- content-aware frame scoring
- optical-flow motion scoring
- adaptive flow weighting
- temporal coverage constraints
- source-frame traceability

Under the same 12-frame budget across 74 clips:

| Metric | Fixed Sampling | Improved Sampling |
|---|---:|---:|
| Low-information frame ratio | 14.0% | 11.4% |
| Proxy motion-event coverage | 98.7% | 95.6% |

The improved sampler retained **96.9%** of fixed-sampling motion-event coverage while reducing the proportion of low-information frames.

Under the final 8-frame-per-clip setting:

- processed a 24:09 episode split into 74 clips
- generated 592 structured keyframe inputs
- reduced frame inputs by 33.3% compared with 12-frame inference
- completed with zero sampler failures

### Cross-Clip Context

The pipeline adds bounded context propagation using:

- structured memory
- confidence-based updates
- source tracking
- bounded history
- explicit protection against stale or unsupported context

A transition-level evaluation was built across 73 adjacent clip boundaries to inspect:

- event ordering
- temporal consistency
- cross-clip contradictions
- stale-context propagation
- unsupported event carryover

### AWS Batch GPU Validation

The inference worker was containerized and deployed with:

- AWS Batch
- Amazon EC2 `g5.xlarge`
- Amazon ECR
- Amazon S3
- Amazon DynamoDB
- Amazon CloudWatch Logs

A BF16 validation run processed 12 clips on one `g5.xlarge` instance:

| Metric | Result |
|---|---:|
| Completed clips | 12 / 12 |
| Structured JSON parse success | 100% |
| Average inference time | 266.8 seconds per clip |
| Median inference time | 264.1 seconds per clip |
| Minimum inference time | 217.9 seconds |
| Maximum inference time | 302.7 seconds |

The AWS Batch compute environment scales to zero when idle.

---

## Architecture

```text
Video or Pre-Split Clips
          |
          v
Selective Keyframe Sampler
          |
          v
Clip Manifest and Frame Metadata
          |
          +-----------------------------+
          |                             |
          v                             v
Local Inference Worker            Amazon S3 Inputs
                                        |
                                        v
                                  AWS Batch Job
                                        |
                                        v
                               Docker Image from ECR
                                        |
                           +------------+------------+
                           |                         |
                           v                         v
                  DynamoDB Job State          S3 JSONL Outputs
                           |
                           v
                  Resume, Retry, Recovery
```

---

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── .dockerignore
├── Dockerfile.worker
├── requirements_worker.txt
├── build_and_push_worker.ps1
├── run_container_dry_test.ps1
├── content_aware_sampler_adaptive_flow.py
├── make_clip_manifest.py
├── make_review_csv.py
├── run_episode_keyframe_pipeline_v5.py
├── evaluation/
│   ├── evaluate_long_video_pipeline.py
│   ├── prepare_eval_inputs.py
│   ├── inputs/
│   ├── outputs/
│   └── scripts/
│       └── generate_fixed_keyframes.py
└── src/
    ├── cloud/
    │   ├── batch_worker_entrypoint.py
    │   ├── materialize_s3_model_bundle.py
    │   ├── prepare_s3_inference_run.py
    │   ├── prepare_s3_model_bundle.py
    │   ├── setup_dynamodb_job_table.py
    │   ├── setup_ecr_repository.py
    │   └── setup_s3_bucket.py
    ├── orchestration/
    │   └── aggregate_worker_outputs.py
    ├── state/
    │   ├── dynamodb_job_state_store.py
    │   ├── job_state_store.py
    │   └── test_job_state_store.py
    ├── storage/
    │   └── s3_artifact_store.py
    ├── tests/
    │   ├── test_dynamodb_job_state_store.py
    │   └── test_inference_worker_recovery.py
    └── workers/
        └── inference_worker_v5.py
```

---

## Local Pipeline

The local pipeline performs:

1. clip discovery
2. selective keyframe sampling
3. manifest generation
4. structured multimodal inference
5. output aggregation
6. evaluation export

Main entry point:

```powershell
python .\run_episode_keyframe_pipeline_v5.py --help
```

The exact episode paths and model paths should be configured before execution.

---

## Cloud Worker

The AWS Batch container entrypoint is:

```text
src/cloud/batch_worker_entrypoint.py
```

The production worker used by the entrypoint is:

```text
src/workers/inference_worker_v5.py
```

The worker supports:

- explicit clip selection
- deterministic sharding
- local or S3 manifests
- S3 output persistence
- DynamoDB-backed job state
- retryable and terminal failure states
- resume behavior
- per-worker JSONL outputs
- structured model responses
- source-frame metadata

### Job States

```text
PENDING
RUNNING
SUCCEEDED
RETRYABLE_FAILED
FINAL_FAILED
```

---

## Docker

Build and push the worker image:

```powershell
.\build_and_push_worker.ps1
```

The image is built from:

```text
Dockerfile.worker
```

Run a local container dry test:

```powershell
.\run_container_dry_test.ps1
```

---

## AWS Components

### Amazon S3

Used for:

- model bundle storage
- inference manifests
- selected keyframes
- worker outputs
- run artifacts

### Amazon DynamoDB

Used for persistent clip-level job state, including:

- atomic job claims
- lease ownership
- retry tracking
- success state
- retryable failures
- terminal failures

### AWS Batch

Used for:

- GPU-backed job execution
- container scheduling
- job retries
- array-job shard selection
- automatic scale-down when idle

### Amazon ECR

Used to store the Docker worker image.

### Amazon CloudWatch Logs

Used to collect AWS Batch container logs.

---

## Evaluation

Primary evaluation scripts:

```text
evaluation/prepare_eval_inputs.py
evaluation/evaluate_long_video_pipeline.py
```

Evaluation artifacts include:

- fixed-sampling inputs
- improved-sampling inputs
- isolated clip summaries
- bounded-context summaries
- transition-level consistency checks
- 8-frame and 12-frame comparisons

Generated evaluation spreadsheets are stored in:

```text
evaluation/outputs/
```

---

## Reliability Tests

The project includes tests for:

- atomic job claims
- retryable failures
- final failures
- DynamoDB state transitions
- resume behavior
- S3 output recovery
- duplicate-work prevention
- worker-output aggregation

---

## Excluded Artifacts

The repository intentionally excludes:

- model weight files
- generated keyframe image directories
- local caches
- local databases
- temporary AWS Batch override files
- local inference JSONL outputs
- AWS credentials
- environment files
- one-off development scripts

Large runtime artifacts should be generated locally or stored in S3 instead of committed to Git.

---

## Base Model and Attribution

This project uses **VideoThinker-R1-3B** as the base multimodal model.

Original project:

- VideoThinker: <https://github.com/falonss703/VideoThinker>

Base model family:

- Qwen2.5-VL-3B-Instruct

Paper citation:

```bibtex
@inproceedings{wu2026videothinker,
  title={Beyond Perceptual Shortcuts: Causal-Inspired Debiasing Optimization for Generalizable Video Reasoning in Lightweight MLLMs},
  author={Wu, Jingze and Zhang, Quan and Suo, Hongfei and Cai, Zeqiang and Chen, Hongbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

---

## License

The original VideoThinker repository is released under the MIT License. Review the original project and model license terms before redistribution or commercial use.
