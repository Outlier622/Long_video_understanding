# Long Video Understanding Optimization

## Overview

This project explores inference-time optimization methods for long-video understanding using a lightweight multimodal video reasoning model. The work is based on **VideoThinker-R1-3B** and focuses on improving the video input pipeline rather than retraining or fine-tuning the model.

The core idea is simple: for long videos, model performance depends heavily on which visual frames are selected as evidence. Uniform frame sampling is easy to implement, but it can miss short-lived events, over-sample static scenes, and provide weak visual support for reasoning.

This project improves the sampling pipeline through:

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

Uniform sampling treats all clips in the same way, regardless of whether the clip contains a static dialogue scene, a fast scene transition, an explosion, a monster movement, or a combat sequence.

This project investigates whether better frame selection can improve clip-level understanding without modifying the model weights.


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

The project evolved through four main versions:

| Version | Method                         | Description                                                                       |
| ------- | ------------------------------ | --------------------------------------------------------------------------------- |
| v1      | Fixed / uniform sampling       | Initial fixed-frame sampling approach                                             |
| v2      | Content-aware sampling         | Uses frame difference, scene change, sharpness, brightness, and temporal coverage |
| v3      | Optical-flow sampling          | Adds fixed optical-flow motion scoring                                            |
| v3.1    | Lower optical-flow weight      | Tests a more conservative fixed flow weight                                       |
| v4      | Adaptive optical-flow sampling | Adjusts optical-flow weight based on clip-level dynamic score                     |


## Version 1: Fixed / Uniform Sampling

The initial version used fixed frame sampling. Each clip was represented by a fixed number of frames selected at regular intervals.

This approach was simple and stable, but it had several limitations:

* It could miss short-lived visual events.
* It treated low-information and high-information moments similarly.
* It did not distinguish between static dialogue scenes and motion-heavy scenes.
* It sometimes encouraged the model to infer actions that were not strongly supported by visible frames.

Fixed sampling served as the baseline, but it was not sufficient for more complex long-video understanding tasks.

### Observation

Fixed sampling is content-agnostic. It applies the same sampling interval to all clips, regardless of whether the clip contains dialogue, screen transitions, explosions, monsters, combat, or fast camera movement.

During manual inspection, this limitation was especially visible in clips where important visual events happened briefly between regular sampling positions.

This motivated the next step: selecting frames based not only on time, but also on visual information.


## Version 2: Content-Aware Keyframe Sampling

The second version introduced content-aware keyframe selection. Instead of selecting frames only by time interval, each candidate frame was scored using visual signals.

The scoring signals included:

* frame difference
* histogram / scene change
* brightness
* sharpness
* temporal coverage

The sampler selected a fixed number of keyframes per clip while trying to preserve both temporal coverage and high-information visual moments.

This version improved the pipeline by selecting more meaningful frames instead of sampling only at fixed intervals.

### Full-Episode Processing Result

On `ep02`, the content-aware sampling pipeline successfully processed the full episode into clip-level keyframe inputs.

| Metric                      |     Value |
| --------------------------- | --------: |
| Episode                     |      ep02 |
| Number of clips             |        74 |
| Total video duration        | 24:09.760 |
| Selected keyframes per clip |         8 |
| Total selected keyframes    |       592 |
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


## Experimental Results

This section summarizes the current experimental evidence collected from the episode-level sampling pipeline and selected clip-level ablation studies.

### Full-Episode Processing Statistics

The pipeline was tested on `ep02`, a full episode-level video. The video was split into short clips and processed with the content-aware / optical-flow-based keyframe sampling pipeline.

| Metric                      |     Value |
| --------------------------- | --------: |
| Episode                     |      ep02 |
| Total duration              | 24:09.760 |
| Number of clips             |        74 |
| Keyframes selected per clip |         8 |
| Total selected keyframes    |       592 |
| Sampler errors              |         0 |

The full-episode run shows that the pipeline can process a complete long video into stable clip-level keyframe inputs. Across 73 clips, the sampler produced 584 selected keyframes with no sampler errors.

This result supports the engineering stability of the pipeline, but it does not by itself prove semantic improvement. For that reason, additional clip-level comparisons and ablation observations were used to analyze sampling quality.


## Sampling Strategy Comparison

The project compares four main sampling strategies.

| Version | Method                         | Main Signal Used                                      | Adaptivity | Main Strength                            | Main Weakness                                               |
| ------- | ------------------------------ | ----------------------------------------------------- | ---------- | ---------------------------------------- | ----------------------------------------------------------- |
| v1      | Fixed / uniform sampling       | Time interval                                         | No         | Simple and stable                        | Can miss short-lived events                                 |
| v2      | Content-aware sampling         | Frame difference, scene change, brightness, sharpness | Partial    | Selects more visually informative frames | Does not explicitly model motion                            |
| v3      | Fixed optical-flow sampling    | Content-aware score + optical flow                    | No         | More sensitive to dynamic scenes         | Can over-emphasize small motion                             |
| v4      | Adaptive optical-flow sampling | Content-aware score + dynamic-score-based flow weight | Yes        | Adjusts motion sensitivity by clip type  | Still affected by camera motion, subtitles, and dark scenes |

This comparison shows the main direction of the project: moving from a content-agnostic baseline toward a more adaptive sampling pipeline.


## Fixed Flow Weight Ablation Summary

Two fixed optical-flow weights were compared.

| Flow Weight | Behavior                             | Advantage                                                                    | Limitation                                                                 |
| ----------: | ------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
|        0.25 | Higher motion sensitivity            | Better for explosions, monster movement, screen transitions, and fast action | Can become over-sensitive in dialogue-heavy, dark, or subtitle-heavy clips |
|        0.15 | More conservative motion sensitivity | Reduces over-sensitive frame selection in some static or dialogue scenes     | May weaken the benefit of optical flow in highly dynamic scenes            |

The ablation suggested that neither fixed weight was universally best. This motivated the adaptive optical-flow design, where the flow weight is selected according to the estimated dynamic level of each clip.


## Adaptive Flow Weight Assignment

The adaptive sampler was tested on two manually selected groups of clips: dialogue / meeting-oriented clips and dynamic / action-oriented clips.

### Dialogue / Meeting-Oriented Clips

| Clip Group                        | Number of Clips | Assigned 0.05 | Assigned 0.15 | Assigned 0.25 |
| --------------------------------- | --------------: | ------------: | ------------: | ------------: |
| Dialogue / meeting-oriented clips |               6 |             0 |             6 |             0 |

All six dialogue / meeting-oriented clips were assigned the medium flow weight of `0.15`. None of them received the highest motion-heavy weight of `0.25`.

This indicates that the adaptive sampler avoided treating dialogue-heavy clips as highly motion-heavy scenes.

### Dynamic / Action-Oriented Clips

| Clip Group                      | Number of Clips | Assigned 0.05 | Assigned 0.15 | Assigned 0.25 |
| ------------------------------- | --------------: | ------------: | ------------: | ------------: |
| Dynamic / action-oriented clips |               6 |             0 |             3 |             3 |

Among the six dynamic / action-oriented clips, three were assigned the higher optical-flow weight of `0.25`. These clips corresponded to stronger visual motion, such as aircraft / explosion scenes, monster confrontation, fire, or heavy action.

This shows that the adaptive sampler increased motion sensitivity for some high-motion clips while keeping moderate weights for less dynamic action or transition clips.


## Adaptive Flow Assignment Rate

Across the 12 manually reviewed clips used for adaptive-flow verification:

| Flow Weight | Number of Clips | Percentage |
| ----------: | --------------: | ---------: |
|        0.05 |               0 |       0.0% |
|        0.15 |               9 |      75.0% |
|        0.25 |               3 |      25.0% |

The reviewed subset shows that most clips were assigned the medium flow weight, while only clearly stronger dynamic clips were assigned the highest flow weight.

This is consistent with the design goal of adaptive optical-flow sampling: avoid applying the highest motion sensitivity globally, while still allowing high-motion clips to receive stronger motion weighting.


## Dynamic Score Statistics on Reviewed Clips

The adaptive-flow verification used 12 manually selected clips.

| Clip Group                  | Number of Clips | Average Dynamic Score | Minimum Score | Maximum Score | Most Common Flow Weight |
| --------------------------- | --------------: | --------------------: | ------------: | ------------: | ----------------------: |
| Dialogue / meeting-oriented |               6 |                 0.562 |         0.406 |         0.722 |                    0.15 |
| Dynamic / action-oriented   |               6 |                 0.695 |         0.541 |         0.805 |             0.15 / 0.25 |
| All reviewed clips          |              12 |                 0.629 |         0.406 |         0.805 |                    0.15 |

The dynamic / action-oriented group had a higher average dynamic score than the dialogue / meeting-oriented group. This supports the assumption that the dynamic score captures at least part of the motion difference between clip types.

However, the separation is not perfect. Some dialogue clips still received moderately high dynamic scores, which suggests that the dynamic score may also be affected by camera motion, subtitles, scene changes, lighting changes, or background movement.


## Current Evidence Level

The current evidence supports three conclusions:

1. The pipeline is stable enough to process a full episode-level video.
2. Content-aware and optical-flow-based sampling provide more flexible frame selection than fixed sampling.
3. Adaptive optical-flow weighting is more reasonable than using a single global flow weight for all clip types.

However, the current evidence is still limited. The project does not yet include a large-scale quantitative evaluation of event recall, hallucination rate, or temporal reasoning accuracy.

A stronger evaluation should add manually labeled ground truth for a subset of clips and compare:

| Metric                      | Meaning                                                                  |
| --------------------------- | ------------------------------------------------------------------------ |
| Event recall                | Whether important visible events are captured by selected keyframes      |
| Hallucination rate          | Whether the model describes actions not supported by the selected frames |
| Low-information frame ratio | How many selected frames are visually redundant or uninformative         |
| Motion-event coverage       | Whether motion-heavy events are represented by selected keyframes        |
| Static-scene over-selection | Whether the sampler wastes frames on visually similar dialogue frames    |

These metrics will be added in future evaluation work.


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

The current experiments suggest that sampling strategy has a meaningful impact on the quality and reliability of long-video understanding inputs.

The main findings are:

1. **Fixed sampling is stable but content-agnostic.**
   It is easy to implement, but it does not adapt to different visual content. It can miss short-lived events and over-sample static or low-information moments.

2. **Content-aware sampling improves visual evidence selection.**
   By using frame difference, scene change, brightness, sharpness, and temporal coverage, the sampler can select frames based on visual information density instead of relying only on fixed time intervals.

3. **Optical flow improves sensitivity to motion-heavy scenes.**
   Optical flow is useful for clips involving explosions, monster movement, transformations, aircraft motion, combat, and fast transitions.

4. **A single fixed optical-flow weight is not ideal.**
   Higher flow weights can improve motion sensitivity, but they can also make the sampler over-sensitive to camera motion, subtitles, dark scenes, or minor visual changes.

5. **Adaptive optical-flow weighting provides a more balanced design.**
   In the reviewed subset, dialogue / meeting-oriented clips avoided the highest flow weight, while stronger dynamic clips could receive higher motion sensitivity. This supports the use of clip-level dynamic scores for adaptive sampling.

6. **The current evaluation is promising but not complete.**
   The project currently provides full-episode processing statistics, selected clip-level ablation results, and qualitative observations. A stronger future evaluation should include manually labeled event recall, hallucination rate, and low-information frame ratio.


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
