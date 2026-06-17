# Long Video Understanding Optimization

## Overview

This project explores inference-time optimization methods for long-video understanding using a lightweight multimodal video reasoning model. The work is based on **VideoThinker-R1-3B** and focuses on improving the video input pipeline rather than retraining or fine-tuning the model.

The core idea is simple: for long videos, model performance depends heavily on which visual frames are selected as evidence. Uniform frame sampling is easy to implement, but it can miss short-lived events, over-sample static scenes, and provide weak visual support for reasoning. This project improves the sampling pipeline through:

* fixed / uniform frame sampling baseline
* content-aware keyframe selection
* optical-flow-based motion scoring
* adaptive optical-flow weighting based on clip dynamics
* structured clip-level outputs for later analysis and evaluation

The final goal is to make long-video understanding more stable, more interpretable, and more suitable for lightweight multimodal video reasoning models under limited visual input budgets.


## Project Motivation

Long-video understanding is challenging because a full video contains far more visual information than a multimodal model can process directly. Most lightweight video-language models must compress a long video into a small number of frames or keyframes before inference.

This creates a key problem:

> If the selected frames do not contain the right visual evidence, the model may miss important events or infer actions that are not strongly supported by the video.

Uniform sampling treats all clips in the same way, regardless of whether the clip contains a static dialogue scene, a fast scene transition, an explosion, a monster movement, or a combat sequence. This project investigates whether better frame selection can improve clip-level understanding without modifying the model weights.


## Base Model

This project uses **VideoThinker-R1-3B** as the base video reasoning model.

VideoThinker-R1 is designed for reasoning with lightweight multimodal large language models. In this repository, the model itself is not retrained. All improvements are made at the inference pipeline level by changing how video clips are sampled, represented, and passed into the model.


## Method

The pipeline processes long videos at the clip level.

```text
Long video
→ split into short clips
→ sample candidate frames
→ score candidate frames
→ select keyframes
→ run multimodal model inference
→ save structured clip-level outputs
→ compare sampling strategies
```

The project evolved through four main versions.


## Version 1: Fixed / Uniform Sampling

The initial version used fixed frame sampling. Each clip was represented by a fixed number of frames selected at regular intervals.

This approach was simple and stable, but it had several limitations:

* It could miss short-lived visual events.
* It treated low-information and high-information moments similarly.
* It did not distinguish between static dialogue scenes and motion-heavy scenes.
* It sometimes encouraged the model to infer actions that were not strongly supported by visible frames.

Fixed sampling served as the baseline, but it was not sufficient for more complex long-video understanding tasks.


## Version 2: Content-Aware Keyframe Sampling

The second version introduced content-aware keyframe selection. Instead of selecting frames only by time interval, each candidate frame was scored using visual signals.

The scoring signals included:

* frame difference
* histogram / scene change
* brightness
* sharpness
* temporal coverage

The sampler selected a fixed number of keyframes per clip while trying to preserve both temporal coverage and high-information visual moments.

### Full-Episode Processing Result

On `ep02`, the content-aware sampling pipeline successfully processed the full episode into clip-level keyframe inputs.

| Metric                      |     Value |
| --------------------------- | --------: |
| Episode                     |      ep02 |
| Number of clips             |        73 |
| Total video duration        | 24:09.760 |
| Selected keyframes per clip |         8 |
| Sampler errors              |         0 |

This showed that the content-aware sampler could run stably across a full episode and produce a consistent number of selected keyframes for each clip.

Compared with fixed sampling, this version made the selected frames depend on visual information density rather than only fixed time intervals.


## Version 3: Optical-Flow-Based Motion Scoring

The third version added optical flow as an explicit motion signal. The purpose was to improve keyframe selection for dynamic scenes such as:

* creature movement
* aircraft motion
* explosions
* transformations
* combat scenes
* fast scene transitions

Optical flow helped the sampler become more sensitive to dynamic visual changes. However, applying the same optical-flow weight to every clip also introduced a trade-off.

In action-heavy clips, optical flow helped prioritize frames with stronger movement cues. But in dialogue-heavy, dark, or subtitle-heavy clips, a fixed optical-flow weight could over-emphasize small movements, camera motion, or low-level visual changes.

This suggested that optical flow was useful, but a single global motion weight was not ideal for all clip types.


## Version 3.1: Fixed Flow Weight Ablation

To understand the effect of optical flow, two fixed optical-flow weights were tested:

```text
flow_weight = 0.25
flow_weight = 0.15
```

The higher weight was more sensitive to dynamic scenes, but it could become over-sensitive in dialogue or dark scenes. The lower weight produced more conservative outputs, but it sometimes weakened the benefit of optical flow in highly dynamic scenes.

### Ablation Observations

| Clip | Scene Type                       | Observation                                                                              |
| ---: | -------------------------------- | ---------------------------------------------------------------------------------------- |
|   22 | screen transition / control room | Higher flow weight was more sensitive to screen changes.                                 |
|   23 | transformation + meeting         | Lower flow weight produced a more conservative result.                                   |
|   48 | aircraft / explosion             | Motion cues were important, but all versions still had imperfect event coverage.         |
|   59 | monster confrontation            | Higher flow weight helped with action sensitivity, but over-inference remained possible. |
|   60 | combat / stance                  | Lower flow weight was more conservative for stance-heavy action.                         |
|   70 | monster / fire / credits         | Lower flow weight reduced some over-sensitive behavior in dark scenes.                   |

The ablation results suggested that neither `0.25` nor `0.15` was universally better. Different clip types required different levels of motion sensitivity.

This motivated adaptive optical-flow weighting.


## Version 4: Adaptive Optical-Flow Sampling

The current version uses adaptive optical-flow weighting. Instead of applying the same motion weight to every clip, the sampler first estimates a clip-level dynamic score.

Based on this dynamic score, each clip is assigned one of three flow weights:

* static or low-motion clips: lower optical-flow weight
* moderately dynamic clips: medium optical-flow weight
* motion-heavy clips: higher optical-flow weight

Current adaptive parameters:

```text
static_flow_weight = 0.05
medium_flow_weight = 0.15
dynamic_flow_weight = 0.25
dynamic_score_low = 0.40
dynamic_score_high = 0.75
```

This design reduces optical-flow influence in dialogue-heavy or low-motion clips while preserving stronger motion sensitivity for explosions, monsters, aircraft, fire, combat, and fast transitions.


## Adaptive Flow Results

Two groups of clips were tested to verify whether the adaptive sampler assigned different flow weights based on clip dynamics.

### Dialogue / Meeting-Oriented Clips

| Clip | Dynamic Score | Category           | Flow Weight |
| ---: | ------------: | ------------------ | ----------: |
|   32 |         0.700 | moderately_dynamic |        0.15 |
|   33 |         0.422 | moderately_dynamic |        0.15 |
|   34 |         0.406 | moderately_dynamic |        0.15 |
|   35 |         0.553 | moderately_dynamic |        0.15 |
|   36 |         0.567 | moderately_dynamic |        0.15 |
|   65 |         0.722 | moderately_dynamic |        0.15 |

All six dialogue or meeting-oriented clips were assigned the medium optical-flow weight of `0.15`. This shows that the adaptive sampler avoided applying the highest motion weight to dialogue-heavy clips.

### Dynamic / Action-Oriented Clips

| Clip | Dynamic Score | Category           | Flow Weight |
| ---: | ------------: | ------------------ | ----------: |
|   22 |         0.641 | moderately_dynamic |        0.15 |
|   23 |         0.541 | moderately_dynamic |        0.15 |
|   48 |         0.794 | motion_heavy       |        0.25 |
|   59 |         0.805 | motion_heavy       |        0.25 |
|   60 |         0.584 | moderately_dynamic |        0.15 |
|   70 |         0.803 | motion_heavy       |        0.25 |

Among the six dynamic or action-oriented clips, three were assigned the higher optical-flow weight of `0.25`. These clips involved stronger visual motion, such as explosions, monster scenes, fire, or heavy action.

This result shows that the adaptive sampler does not rely on a single global optical-flow weight. Instead, it adjusts motion sensitivity based on the estimated dynamic level of each clip.


## Summary of Iterations

| Version | Sampling Strategy               | Main Purpose                              | Observation                                                                    |
| ------- | ------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------ |
| v1      | Fixed / uniform sampling        | Simple baseline sampling strategy         | Easy to implement, but content-agnostic                                        |
| v2      | Content-aware sampling          | Select visually informative frames        | Stable full-episode processing with 73 clips and 0 sampler errors              |
| v3      | Fixed optical-flow sampling     | Add motion sensitivity                    | More sensitive to dynamic events, but sometimes over-sensitive                 |
| v3.1    | Lower fixed optical-flow weight | Test more conservative motion scoring     | Reduced over-sensitivity in some clips, but weakened dynamic-event sensitivity |
| v4      | Adaptive optical-flow sampling  | Adjust flow weight based on clip dynamics | More flexible across dialogue, moderate-motion, and action-heavy clips         |

Overall, the project evolved from a simple fixed sampling baseline to a more adaptive keyframe selection strategy. The final adaptive optical-flow version provides a more flexible sampling pipeline by adjusting motion sensitivity according to clip-level dynamic scores.


## Key Findings

The experiments suggest that keyframe sampling has a meaningful effect on long-video understanding quality.

The main findings are:

1. **Fixed sampling is stable but content-agnostic.**
   It can miss short-lived events and does not adapt to different clip types.

2. **Content-aware sampling improves visual evidence selection.**
   It uses frame-level visual signals to select more informative keyframes.

3. **Optical flow improves sensitivity to motion-heavy scenes.**
   It is useful for explosions, monster movement, transformations, aircraft motion, and combat.

4. **Fixed optical-flow weighting is not ideal.**
   A high flow weight can over-emphasize small movements in dialogue-heavy, subtitle-heavy, or dark scenes.

5. **Adaptive optical-flow weighting provides a better balance.**
   It allows the sampler to use stronger motion sensitivity only when the clip appears dynamic enough.


## Example Usage

### Run the Episode-Level Keyframe Pipeline

```bash
python src/run_episode_keyframe_pipeline_v2.py \
  --video ep02.mp4 \
  --episode ep02 \
  --run-mode all \
  --num-frames 8 \
  --force-keyframes \
  --force-manifest
```

### Generate Clip Manifest Files

```bash
python src/make_clip_manifest.py \
  --summary sampling_summary_flow.jsonl \
  --output-jsonl clip_manifest_flow.jsonl \
  --output-csv clip_manifest_flow.csv
```


## Example Output

The adaptive sampler produces structured clip-level metadata.

```json
{
  "clip_id": "ep02_0048",
  "dynamic_score": 0.794,
  "dynamic_category": "motion_heavy",
  "flow_weight": 0.25,
  "selected_keyframes": 8,
  "low_information": false
}
```

This output can be used for later analysis, comparison, visualization, and potential clip-level memory retrieval.


## Limitations

This project does not fine-tune or retrain VideoThinker-R1-3B. The improvement is focused on inference-time input sampling.

Current limitations include:

* Evaluation is still mostly based on clip-level qualitative comparison.
* A larger manually labeled test set is needed.
* Event recall and hallucination reduction are not yet fully quantified.
* Some static or dialogue-heavy clips may still receive moderate dynamic scores.
* Optical flow can still be affected by camera motion, subtitles, credits, or dark scenes.
* The model may still infer actions that are not fully supported by selected keyframes.

A more complete evaluation would require:

* human-labeled event annotations
* quantitative event recall scoring
* hallucination rate measurement
* comparison against more sampling strategies
* improved handling of subtitles, credits, dark frames, and camera motion


## Future Work

Planned improvements include:

* clip-level memory and retrieval
* episode-level timeline construction
* quantitative event recall evaluation
* hallucination and over-inference analysis
* scene-aware dynamic thresholding
* better subtitle and low-information frame filtering
* separating camera motion from object motion more robustly
* automatic clip-type classification
* query-conditioned keyframe selection
* larger manually reviewed evaluation tables


## References

* Jingze Wu, Quan Zhang, Hongfei Suo, Zeqiang Cai, and Hongbo Chen.
  **Beyond Perceptual Shortcuts: Causal-Inspired Debiasing Optimization for Generalizable Video Reasoning in Lightweight MLLMs**.
  arXiv:2605.01324 / CVPR 2026.
  https://arxiv.org/abs/2605.01324

* VideoThinker official repository.
  **falonss703/VideoThinker**.
  https://github.com/falonss703/VideoThinker

* Falconss1.
  **VideoThinker-R1-3B**. Hugging Face model page.
  https://huggingface.co/Falconss1/VideoThinker-R1-3B

* Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye.
  **Adaptive Keyframe Sampling for Long Video Understanding**.
  arXiv:2502.21271, 2025.
  https://arxiv.org/abs/2502.21271

* Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, and Yang You.
  **FOCUS: Efficient Keyframe Selection for Long Video Understanding**.
  arXiv:2510.27280, 2025.
  https://arxiv.org/abs/2510.27280

* Yifeng Yao, Yike Yun, Jing Wang, Huishuai Zhang, Dongyan Zhao, Ke Tian, Zhihao Wang, Minghui Qiu, and Tao Wang.
  **K-frames: Scene-Driven Any-k Keyframe Selection for Long Video Understanding**.
  arXiv:2510.13891, 2025.
  https://arxiv.org/abs/2510.13891

* Berthold K. P. Horn and Brian G. Schunck.
  **Determining Optical Flow**.
  Artificial Intelligence, 1981.
  https://www.cmor-faculty.rice.edu/~zhang/caam699/opt-flow/horn81.pdf

* Gunnar Farnebäck.
  **Two-Frame Motion Estimation Based on Polynomial Expansion**.
  Scandinavian Conference on Image Analysis, 2003.
  https://link.springer.com/chapter/10.1007/3-540-45103-X_50
