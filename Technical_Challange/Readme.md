# CPU enabled Streaming ASR Model for Everyday Devices
# Choosen Model: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
# Size : 0.6B

## 1. Where the Idea Started

During a NeurIPS ASR competition, I realised that I did not have a fixed set of speech recognition models that I could confidently use on my own phone, CPU, or other edge devices. I had experimented with different ASR models, but choosing the right one for a real deployment was not as straightforward as comparing their reported scores.

Most ASR benchmarks focus on metrics such as Word Error Rate (WER), Real-Time Factor (RTF), RTFx, and sometimes Time to First Token (TTFT). However, these measurements can be reported under very different hardware conditions, including high-end GPUs such as NVIDIA H100s. A model that performs well on a powerful GPU may not be the best choice for a laptop CPU, a mobile device, or a small edge computer.

This made me think about a simple question:

**If I want to run speech recognition locally on my own device, how do I know which model is actually suitable for me?**

## 2. Why This Matters

Automatic Speech Recognition (ASR) is becoming part of everyday life. People use it for voice typing, accessibility, meeting transcription, personal assistants, and many other tasks. However, running every speech recognition system through a remote server is not always practical.

There are several reasons to consider local inference:

- **Accessibility:** People should be able to use ASR on affordable devices without requiring access to expensive hardware.
- **Privacy:** Local inference can reduce the privacy violation. Less risk of data theft.
- **Connectivity:** Offline speech recognition can remain useful when internet access is unreliable or unavailable.
- **Cost:** Local inference may reduce recurring server costs for suitable workloads.
- **Latency:** A well-designed local streaming model may provide a responsive experience without relying on network communication.

The challenge is that model performance depends on more than recognition accuracy. It also depends on the device, inference runtime, quantization, streaming configuration and the amount of delay a user can tolerate.

## 3. The Research Idea

I want to investigate whether we can build a practical **streaming ASR benchmark that helps people choose a model based on their actual deployment requirements**, rather than relying only on conventional accuracy scores or results obtained on powerful GPUs.

My starting point is the OpenASR leaderboard. I would like to explore whether a complementary section could report streaming performance on CPU and other resource-constrained devices.

The goal is not simply to identify the fastest model. A very fast model may make too many transcription errors, while a highly accurate model may require too much memory or produce its first useful text too late.

Instead, I want the benchmark to show the trade-offs clearly, so that a developer can select a suitable model before investing time in task-specific fine-tuning.

## 4. What Should the Benchmark Measure?

The benchmark would evaluate each model under a consistent experimental setup.

| Metric | Meaning |
|---|---|
| Word Error Rate (WER) | How often the recognised words differ from the reference transcript. |
| Character Error Rate (CER) | How much the recognised text differs at the character level. |
| Time to First Text (TTFT) | How long the system takes to emit its first non-empty text. |
| Real-Time Factor (RTF) | How much computation time is required relative to the duration of the audio. |
| Processing speed | How quickly the system processes audio under the specified test conditions. |
| Streaming stability | Whether partial transcripts are revised, delayed, or frequently changed as more audio arrives. |
| Hardware requirements | Which CPU, memory configuration, runtime, and inference settings were used. |

These metrics need precise definitions. For example, TTFT should specify whether timing begins when audio arrives, when the first audio chunk is available, or when inference starts. Similarly, RTF should distinguish offline processing from real-time audio streaming.

The benchmark should also distinguish **configured streaming latency** from experimentally measured TTFT. They describe different aspects of system behaviour.

## 5. My Initial Experimental Plan

I plan to build the evaluation in stages rather than immediately developing a new model.

###  1 — Load the models

###  2 — Load the dataset
Torgo has been chosen becasue dysarthric ASR can require streaming solutions and it's transcription is still hard for models.
###  3 — Retrieve and prepare audio

###  4 — Verify streaming inference setup

###  5 — Benchmark the model

Measure WER, CER, TTFT, RTF, processing speed etc.

###  6 — Compare different models

Models such as parakeet, Moonshine Streaming, Zipformer .

###  7 — Analyse the trade-offs

Determine which model performs best under different requirements. For example, a developer might prefer lower TTFT for interactive voice typing, lower WER for transcription, or lower memory use for an edge device.


## 6. What I Hope to Achieve

My longer-term goal is to create a useful reference for choosing ASR models under realistic deployment constraints.

Rather than asking only which model achieves the best WER, I want to answer questions such as:

- Which streaming ASR model gives the best accuracy on a single CPU?
- Does quantization improve inference speed enough to justify any loss in recognition accuracy?
- Does a model that performs well on clean English speech remain reliable on difficult speech?
- Which models provide a useful balance between accuracy, first-text latency and computational cost?
- Can the benchmark help developers choose a suitable starting model before fine-tuning it for a particular task?

If the results support it, the benchmark could eventually become very useful  community resource, similar to openASR leaderboard.

## 7. The Next Research Direction

The first milestone is to establish a reliable baseline across a small number of models. Only after that will I decide whether the main contribution should be a benchmark, a new evaluation metric, a deployment-oriented model selection method, or an improved inference approach.
Next steps would be understanding the inference metrices and using them to properly benchamrk according to use cases.

**The central question remains simple: when someone needs speech recognition on an ordinary device, can we help them choose the model that actually works best for their needs?**
