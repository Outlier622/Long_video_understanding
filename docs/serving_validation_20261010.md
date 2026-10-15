# Serving readiness validation — 2026-10-10

## Result

Local Transformers and Docker vLLM sequential short-generation smoke tests passed.
Production serving readiness and complete structured-output quality remain unverified.

## Verified environment and inputs

- Windows 10; Conda environment `videothinker`, Python 3.10.20.
- PyTorch 2.7.1+cu118; Transformers 5.8.0.dev0; Accelerate 1.13.0;
  qwen-vl-utils 0.0.14. This differs from the worker image's pinned dependencies.
- NVIDIA RTX 3060 Laptop GPU, 6 GiB VRAM, driver 546.29.
- Checkpoint architecture: `Qwen2_5_VLForConditionalGeneration`.
- Both safetensors shards opened successfully; all 824 tensor names match the
  weight index. Stored elements: 3,754,622,976, all BF16; 7,509,245,952 bytes
  (approximately 6.99 GiB). This checks headers/index, not checksum integrity.
- Processor loaded from local files without network access.
- ep02 `clip_manifest_flow.jsonl`: 73 records, unique IDs 0–72.
  Selected clips 1, 3, 14 each have 8 accessible frame images.
- ep01 `clip_manifest.jsonl`: 74 records, IDs 0–73, all referenced images exist.
  ep01 is the dataset described by the README's 74-clip result; it was not used
  for this generation run.

## Runtime smoke results

One model load took 12.36 seconds. The visual module, embeddings, output head,
and language layers 0–13 were assigned to GPU; layers 14–35 and language norm
were assigned to CPU. This was deliberate to leave room for activations.

| Clip | KV cache | Frames | Input tokens | Output tokens | Generation seconds | Output tokens/s | Peak GPU allocated GiB |
|---|---|---|---|---|---|---|---|
| 1 | Off | 8 | 2131 | 16 | 52.91 | 0.302 | 4.61 |
| 1 | On | 8 | 2131 | 16 | 17.71 | 0.904 | 4.67 |
| 3 | On | 8 | 2131 | 16 | 17.23 | 0.929 | 4.68 |
| 14 | On | 8 | 2132 | 16 | 17.14 | 0.933 | 4.67 |

All four calls completed in one process using one model instance. All reached
the intentional 16-token limit and returned incomplete JSON prefixes. Therefore
JSON correctness, semantic quality, TTFT/TPOT, concurrency and serving throughput
have not been validated. Generation time includes prefill and CPU offload; the
table is not a production-GPU benchmark.

The checkpoint's effective text and generation configurations both have
`use_cache=false`. The existing worker does not override it. The first cache-off
call was not warmed up, whereas the cache-on call was subsequent; the observed
2.99x ratio is a hypothesis for a controlled follow-up, not an isolated causal
performance claim. Clip 1's two output prefixes were identical. Model files and
existing worker code were not changed.

An initial probe stopped after successful loading because its logging assumed
an older Transformers config layout. The probe was fixed to inspect the nested
text config; the subsequent run completed successfully.

## vLLM status

After Docker Desktop was started, its Linux engine and GPU passthrough passed
an actual CUDA tensor-operation test. The existing worker image has PyTorch
2.6.0+cu124 and Transformers 4.51.3. A derived local image,
`videothinker-vllm-validation:0.8.5`, adds vLLM 0.8.5 and GCC/libc development
files. The initial attempt loaded both weight shards but failed in Triton
warmup because the runtime base lacked a C compiler; the corrected image passed.
Final image ID: `sha256:421bcd51d2db69a916c2bd375367bb51b4f7e8c28200ee4091414f8d539de2df`.
Initialization took 132.49 seconds. Only vLLM itself is version-pinned in the
validation Dockerfile; retain this image for exact reruns, since rebuilding can
resolve different transitive dependencies.

One V0 engine loaded the local checkpoint once and processed clips 1 and 3
sequentially, with 8 images each and 16 output tokens. Engine settings: BF16,
CPU offload 6 GB, eager execution, GPU memory utilization 0.8, context 4096,
one concurrent sequence, batch token budget 4096, image pixel budget 100352.
Initialization includes weights, profiling and KV cache creation.

| Clip | Input tokens | Output tokens | Generate call seconds | Finish reason |
|---|---|---|---|---|
| 1 | 2091 | 16 | 22.60 | length |
| 3 | 2091 | 16 | 20.62 | length |

Both returned JSON prefixes identifying the correct clip. Outputs are truncated;
this establishes checkpoint/multimodal runtime compatibility, not full JSON
validity, semantic accuracy, HTTP serving or concurrency. Input token counts
differ from the Transformers probe (2131); tokenizer/processor behavior must be
aligned before any comparative performance claim. The engine's reported 16.43x
KV-cache capacity estimate is not a measured concurrency result; max_num_seqs
was 1.

Runtime memory profiling reported weights 1.95 GiB, peak activations 0.50 GiB,
non-Torch memory 0.03 GiB and KV cache 2.31 GiB. These values exclude CPU-resident
weights. Model/data/code mounts were read-only; inference had network disabled.
The test container exited successfully and was removed. No cloud resources,
host drivers or Conda packages were changed.

Artifacts: `scripts/Dockerfile.vllm-validation`,
`scripts/validate_vllm_checkpoint.py`, and ignored runtime measurements in
`evaluation/runs/vllm_smoke_20261010/results.json`. The initial compiler failure
is preserved in `missing_compiler.json` beside those measurements.

Official references:
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md

The exact local checkpoint and processor now have an execution smoke result.

## Reproduce the completed smoke run

Use a fresh output directory each time:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& 'C:\Users\uxangy\miniconda3\envs\videothinker\python.exe' `
  scripts/validate_serving_baseline.py `
  --model 'D:\projects\VideoThinker-R1-3B' `
  --manifest 'D:\projects\longvideo\episodes\ep02\keyframes\clip_manifest_flow.jsonl' `
  --output-dir evaluation/runs/serving_smoke_new_run
```

Completed measurements are in
`evaluation/runs/serving_smoke_20261010_retry/results.json` (git-ignored runtime data).

Docker vLLM reproduction from the project directory (choose a fresh run folder):

```powershell
docker build -f scripts/Dockerfile.vllm-validation -t videothinker-vllm-validation:0.8.5 .
$runDir = Join-Path (Get-Location) 'evaluation/runs/vllm_smoke_new_run'
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
docker run --rm --network none --gpus all --shm-size 1g `
  --mount 'type=bind,source=D:\projects\VideoThinker-R1-3B,target=/model,readonly' `
  --mount 'type=bind,source=D:\projects\longvideo,target=/data,readonly' `
  --mount 'type=bind,source=D:\projects\Long_video_understanding_clean,target=/workspace,readonly' `
  --mount "type=bind,source=$runDir,target=/results" `
  videothinker-vllm-validation:0.8.5 /workspace/scripts/validate_vllm_checkpoint.py
```

## Next validation

The resident local HTTP service was subsequently implemented and validated;
see [HTTP service instructions and acceptance results](http_serving.md).
Complete generated JSON structure and clip metadata passed, but semantic model
quality still requires evaluation. The remaining production GPU work below is
a performance/quality baseline rather than a prerequisite for local HTTP use.

Use a Linux GPU with enough VRAM for weights, vision activations and KV cache;
the previously used 24 GiB AWS GPU is a practical candidate. Pin the environment,
use the same ep01 keyframes and prompt, and establish a warmed, full-output
Transformers baseline. Compare cache off/on using repeated alternating runs
and full structured-output validation. Then execute the actual checkpoint
through vLLM with identical frame count, pixel budget and generation limit.
Only after those pass should API and concurrency acceptance tests be claimed.
