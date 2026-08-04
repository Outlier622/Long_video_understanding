Long-Video Inference Optimization

A long-video inference pipeline built on top of VideoThinker-R1-3B that improves frame selection, adds bounded cross-clip context, and supports resumable cloud execution with AWS Batch, S3, DynamoDB, Docker, and Amazon ECR.

This repository focuses on inference optimization and systems engineering. It does not retrain the base model.

Project Overview

Long videos are difficult to process efficiently because uniform frame sampling can waste the frame budget on low-information moments, miss short-lived motion events, and provide weak temporal continuity across adjacent clips.

This project addresses those problems through three components:

Selective keyframe sampling

Bounded cross-clip context

Resumable cloud-backed inference workers

The base model is VideoThinker-R1-3B, derived from Qwen2.5-VL-3B-Instruct.

Key Results

Selective Keyframe Sampling

The final sampler combines:

content-aware frame scoring

optical-flow motion scoring

adaptive flow weighting

temporal anchor frames

source-frame traceability

Under the same 12-frame budget across 74 clips:

low-information frame ratio decreased from 14.0% to 11.4%

proxy motion-event coverage changed from 98.7% to 95.6%

the improved sampler retained 96.9% of fixed-sampling coverage

The final 8-frame configuration:

generated 592 structured keyframe inputs

reduced frame inputs by 33.3% compared with 12-frame inference

completed with zero sampler failures

Cross-Clip Context

The pipeline adds bounded context propagation using:

structured memory

confidence-based updates

source tracking

bounded history instead of unrestricted full-history accumulation

A transition-level evaluation was built across 73 adjacent clip boundaries to inspect:

event ordering

temporal consistency

cross-clip contradictions

stale or unsupported context propagation

AWS Batch GPU Validation

The inference worker was containerized and deployed with:

AWS Batch for job execution

Amazon EC2 g5.xlarge for GPU inference

Amazon ECR for the worker image

Amazon S3 for manifests, model artifacts, and outputs

Amazon DynamoDB for persistent job state

CloudWatch Logs for execution logs

A BF16 cloud validation run processed 12 clips on one g5.xlarge instance:

12/12 clips completed successfully

100% structured JSON parse success

average inference time: 266.8 seconds per clip

minimum: 217.9 seconds

maximum: 302.7 seconds

The compute environment scales to zero when idle.

Architecture

Video / Clips
    |
    v
Selective Keyframe Sampler
    |
    v
Clip Manifest + Structured Frame Metadata
    |
    +----------------------+
    |                      |
    v                      v
Local Worker          S3 Input Artifacts
                           |
                           v
                    AWS Batch Job
                           |
                           v
                 Docker Worker from ECR
                           |
              +------------+------------+
              |                         |
              v                         v
       DynamoDB Job State          S3 JSONL Output
              |
              v
     Resume / Retry / Recovery

Repository Structure

.
├── content_aware_sampler_adaptive_flow.py
├── make_clip_manifest.py
├── make_review_csv.py
├── run_episode_keyframe_pipeline_v5.py
├── Dockerfile.worker
├── requirements_worker.txt
├── evaluation/
│   ├── evaluate_long_video_pipeline.py
│   ├── prepare_eval_inputs.py
│   ├── inputs/
│   ├── outputs/
│   └── scripts/
└── src/
    ├── cloud/
    │   ├── batch_worker_entrypoint.py
    │   ├── prepare_s3_inference_run.py
    │   ├── prepare_s3_model_bundle.py
    │   ├── materialize_s3_model_bundle.py
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

The AWS Batch container entrypoint is:

src/cloud/batch_worker_entrypoint.py

Local Pipeline

The local pipeline performs:

clip discovery

selective keyframe sampling

manifest creation

structured inference

output aggregation

evaluation export

Main entry point:

python run_episode_keyframe_pipeline_v5.py --help

The exact input and output paths should be configured for the local episode directory before execution.

Cloud Worker Design

The cloud worker supports:

explicit clip selection

deterministic sharding

local or S3 manifests

S3 output persistence

DynamoDB-backed job state

retries and terminal failure states

resume behavior

per-worker JSONL outputs

source-frame metadata

structured model responses

The worker state model includes:

PENDING
RUNNING
SUCCEEDED
RETRYABLE_FAILED
FINAL_FAILED

Docker

Build the worker image with:

.\build_and_push_worker.ps1

The image is built from:

Dockerfile.worker

A local container dry run can be executed with:

.\run_container_dry_test.ps1

Evaluation

Evaluation artifacts include:

fixed-sampling inputs

improved-sampling inputs

isolated clip summaries

bounded-context summaries

transition-level consistency checks

8-frame and 12-frame comparisons

Primary scripts:

evaluation/prepare_eval_inputs.py
evaluation/evaluate_long_video_pipeline.py

Reliability Tests

The project includes tests for:

atomic job claims

retryable failures

terminal failures

resume behavior

DynamoDB state transitions

S3 output recovery

duplicate-work prevention

worker output aggregation

Notes on Generated Artifacts

The repository intentionally excludes:

model weight files

generated keyframe image directories

local caches

temporary Batch override files

local JSONL inference outputs

AWS credentials

environment files

one-off development scripts

These artifacts should be generated locally or stored in S3 rather than committed to Git.

Base Model and Attribution

This project uses VideoThinker-R1-3B as the base multimodal model.

Original project:

VideoThinker: https://github.com/falonss703/VideoThinker

Base model family: Qwen2.5-VL-3B-Instruct

Paper:

@inproceedings{wu2026videothinker,
  title={Beyond Perceptual Shortcuts: Causal-Inspired Debiasing Optimization for Generalizable Video Reasoning in Lightweight MLLMs},
  author={Wu, Jingze and Zhang, Quan and Suo, Hongfei and Cai, Zeqiang and Chen, Hongbo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

License

The original VideoThinker model and repository are released under the MIT License. Check the original project and model license terms before redistribution or commercial use.