# Long Video Understanding Optimization

## Overview

This project explores practical optimization methods for long-video understanding using a lightweight multimodal video reasoning model. The work is based on VideoThinker-R1-3B and focuses on improving the input sampling pipeline rather than retraining the model itself.

The main goal is to improve how long videos are converted into representative visual evidence for model inference. Instead of relying only on fixed or uniform frame sampling, this project experiments with content-aware keyframe selection, optical-flow-based motion scoring, and adaptive optical-flow weighting.

## Motivation

Long-video understanding is difficult because a full video contains far more visual information than a multimodal model can process within a limited token budget. If frames are sampled uniformly, short but important events may be missed. If too many visually redundant or low-information frames are selected, the model may waste context on unimportant content.

This project investigates whether better keyframe sampling can improve clip-level understanding for long videos, especially for scenes involving action, screen changes, transformations, monsters, explosions, and dialogue-heavy segments.

## Base Model

This project uses VideoThinker-R1-3B as the base video reasoning model.

VideoThinker-R1 focuses on improving reasoning in lightweight multimodal large language models. In this project, the model weights are not modified. The optimization is performed at the inference pipeline level, mainly by changing how video clips are sampled and represented as keyframes.

## Research Process

### 1. Initial Approach: Fixed / Uniform Sampling

The first version of the pipeline used a fixed sampling strategy. Each video clip was represented by a fixed number of frames selected at regular intervals. This approach was simple and easy to implement, but it had several limitations:

* It could miss short-lived visual events.
* It treated low-information and high-information moments similarly.
* It did not distinguish between static dialogue scenes and motion-heavy scenes.
* It sometimes encouraged the model to infer actions that were not strongly supported by visible frames.

Therefore, fixed sampling served as the initial version of the pipeline, but it was not sufficient for more complex long-video understanding tasks.

### 2. Content-Aware Keyframe Sampling

The next step was to build a content-aware keyframe sampler. Each long video was split into short clips, and candidate frames inside each clip were scored using multiple visual signals.

The scoring signals included:

* frame difference
* histogram / scene change
* brightness
* sharpness
* temporal coverage

The sampler selected a fixed number of keyframes per clip while trying to preserve both temporal coverage and high-information visual moments.

This version improved the pipeline by selecting more meaningful frames instead of sampling only at fixed intervals.

### 3. Optical-Flow-Based Motion Scoring

After content-aware sampling, optical flow was added as an additional motion signal. The purpose was to better capture dynamic events such as:

* creature movement
* aircraft motion
* explosions
* transformations
* combat scenes
* fast scene transitions

The optical-flow score was combined with the existing content-aware scores. This version improved some motion-heavy clips, but it also showed limitations. When optical flow was applied with a fixed weight to every clip, dialogue-heavy or low-motion clips could become overly sensitive to small movements, camera motion, or subtitles.

### 4. Flow Weight Ablation

To understand the effect of optical flow, two fixed optical-flow weights were tested:

* `flow_weight = 0.25`
* `flow_weight = 0.15`

The higher weight was more useful for some dynamic or screen-transition scenes, but it could introduce over-sensitive frame selection in dialogue or dark scenes. The lower weight produced more conservative outputs, but it sometimes reduced the benefit of optical flow in highly dynamic scenes.

This showed that a single global optical-flow weight is not ideal for all clip types.

### 5. Adaptive Optical-Flow Sampling

The current version uses adaptive optical-flow weighting. Instead of using the same optical-flow weight for every clip, the sampler first estimates a clip-level dynamic score.

Based on this score, each clip is assigned one of three flow weights:

* static or dialogue-heavy clips: lower optical-flow weight
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

This design allows the sampler to reduce optical-flow influence in dialogue-heavy clips while still preserving stronger motion sensitivity for explosion, monster, aircraft, and combat scenes.

## Experiment Versions

The project uses episode-level videos split into short clips. Each clip is processed through different sampling strategies:

| Version | Method                         | Description                                                                       |
| ------- | ------------------------------ | --------------------------------------------------------------------------------- |
| v1      | Fixed / uniform sampling       | Initial fixed-frame sampling approach                                             |
| v2      | Content-aware sampling         | Uses frame difference, scene change, sharpness, brightness, and temporal coverage |
| v3      | Optical-flow sampling          | Adds fixed optical-flow motion scoring                                            |
| v3.1    | Lower optical-flow weight      | Tests a more conservative fixed flow weight                                       |
| v4      | Adaptive optical-flow sampling | Adjusts optical-flow weight based on clip-level dynamic score                     |

## Key Findings

The experiments suggest that keyframe sampling has a meaningful effect on long-video understanding quality.

Fixed sampling is simple, but it can miss short-lived visual events. Content-aware sampling helped reduce reliance on fixed intervals and improved the selection of visually informative frames. Optical flow improved some motion-heavy clips, especially scenes involving explosions, monsters, transformations, and combat. However, fixed optical-flow weighting was not ideal for all scenes.

Adaptive optical-flow sampling provided a more balanced approach. Dialogue and meeting clips were assigned medium flow weight, while stronger action-heavy clips were assigned higher flow weight. This made the sampling strategy more flexible across different scene types.

## Limitations

This project does not fine-tune or retrain VideoThinker-R1-3B. The improvement is focused on inference-time input sampling.

The current evaluation is based on clip-level qualitative comparison. A more complete evaluation would require:

* a larger test set
* human-labeled event annotations
* quantitative scoring for event recall and hallucination rate
* comparison against additional sampling strategies
* better handling of subtitles, credits, dark scenes, and camera motion

The model may still infer actions that are not fully supported by the selected keyframes, such as describing static standing poses as continuous actions like “discussing,” “speaking,” or “looking around.”

## Future Work

Future improvements may include:

* scene-aware dynamic thresholding
* better subtitle and low-information frame filtering
* separating camera motion from object motion more robustly
* automatic clip-type classification
* query-conditioned keyframe selection
* quantitative evaluation of event recall and hallucination reduction
* a larger manually reviewed evaluation table

## Summary

The core contribution of this project is not model retraining, but the design and evaluation of a more effective inference-time sampling pipeline for long-video understanding.

By moving from fixed sampling to content-aware sampling, then to optical flow and adaptive optical flow, this project demonstrates how better visual evidence selection can improve clip-level understanding under limited visual input budgets.

The final goal is to make long-video understanding more stable, more interpretable, and more suitable for lightweight multimodal video reasoning models.

## References

- Jingze Wu, Quan Zhang, Hongfei Suo, Zeqiang Cai, and Hongbo Chen.  
  **Beyond Perceptual Shortcuts: Causal-Inspired Debiasing Optimization for Generalizable Video Reasoning in Lightweight MLLMs**.  
  arXiv:2605.01324 / CVPR 2026.  
  https://arxiv.org/abs/2605.01324

- VideoThinker official repository.  
  **falonss703/VideoThinker**.  
  https://github.com/falonss703/VideoThinker

- Falconss1.  
  **VideoThinker-R1-3B**. Hugging Face model page.  
  https://huggingface.co/Falconss1/VideoThinker-R1-3B

- Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye.  
  **Adaptive Keyframe Sampling for Long Video Understanding**.  
  arXiv:2502.21271, 2025.  
  https://arxiv.org/abs/2502.21271

- Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, and Yang You.  
  **FOCUS: Efficient Keyframe Selection for Long Video Understanding**.  
  arXiv:2510.27280, 2025.  
  https://arxiv.org/abs/2510.27280

- Yifeng Yao, Yike Yun, Jing Wang, Huishuai Zhang, Dongyan Zhao, Ke Tian, Zhihao Wang, Minghui Qiu, and Tao Wang.  
  **K-frames: Scene-Driven Any-k Keyframe Selection for Long Video Understanding**.  
  arXiv:2510.13891, 2025.  
  https://arxiv.org/abs/2510.13891

- Berthold K. P. Horn and Brian G. Schunck.  
  **Determining Optical Flow**.  
  Artificial Intelligence, 1981.  
  https://www.cmor-faculty.rice.edu/~zhang/caam699/opt-flow/horn81.pdf

- Gunnar Farnebäck.  
  **Two-Frame Motion Estimation Based on Polynomial Expansion**.  
  Scandinavian Conference on Image Analysis, 2003.  
  https://link.springer.com/chapter/10.1007/3-540-45103-X_50
