# ASR Edge Benchmark Plan

**User:** Anonymous  
**Created:** 10/19/2026 16:50:43  
**Updated:** 10/20/2026 1:06:54  
**Exported:** 10/20/2026 1:10:41  
**Link:** [https://chatgpt.com/c/6ad5f603-0238-83e8-84c4-919f622445fa](https://chatgpt.com/c/6ad5f603-0238-83e8-84c4-919f622445fa)  

## Prompt:
10/19/2026, 4:50:39 PM

Technical challenge: Blind Spots of Frontier Models&#x20;
For research, it is crucial that you think creatively and directly inspect a model to develop a deeper understanding of its behavior. We will prioritize applications that answer this question creatively and show clear signs of human thought and effort\*  At Fatima Institute, we believe that AI should be built by researchers from around the world with diverse lived experiences and unique perspectives. &#x20;
1\. Drawing directly from your background and lived experience, identify a critical blind spot, bias, or capability gap in current AI models that is rarely captured by standard benchmarking datasets, and that you feel is underexplored and underdiscussed.  2. Systematically evaluate this gap on a frontier open-weight model hosted on Hugging Face that has between 0.6 and 6B parameters (You can find these at [https://huggingface.co/models?num_parameters=min:0.6B,max:6B&sort=trending](https://huggingface.co/models?num_parameters=min:0.6B,max:6B\&sort=trending))   Try to find one that realistically captures the behavior of current frontier models while being small enough that you can load yourself. We will consider your choice of model as part of grading this question.   Next, load the model in a Colab notebook at [https://colab.google.com/](https://colab.google.com/) or using Modal at [https://modal.com/](https://modal.com/). You can also load the model using any other GPU platform you have access to, but both of these services offer free tiers.  Evaluate the model you have chosen to showcase the gap. Do this in a systematic way so that you can tell a coherent story and help explain the specific blind spot of the model.  3. Propose a path forward: what specific data curation strategy, fine-tuning paradigm, or architectural adjustment do you think might close this gap? This can be purely speculative, you do not need to prove this.  Then, create a public Hugging Face Bucket that includes all the artifacts related to your evaluation, e.g. the model, the evaluation dataset you built, as well as a README that includes the answers to questions 1 and 3. Please link to the Hugging Face Bucket here.  .........................get read y for my plan...I will use nemo 0.6B model,,,,ASR,,will note the streaming ttft, for cpu based asr models...as edge devices could use faster infefrence benhcamakr

## Response:
10/19/2026, 4:50:50 PM · gpt-6-thinking

> **Remembering**
>
>
> **Reviewed Fatima plan, NeMo ASR choices, benchmarks, and streaming datasets**
>
> 
>
> Worked for 23s

Your plan has a promising research direction: **investigate whether small, open-weight ASR models can deliver genuinely low-latency streaming transcription on CPU edge devices without sacrificing accuracy on difficult speech.**

The strongest version of this project is not simply a speed benchmark. It is an investigation into a potential blind spot in frontier-model evaluation: **a model can appear efficient in offline transcription benchmarks while performing poorly under real-time streaming constraints, especially when speech is disfluent, acoustically difficult, or produced by speakers with dysarthria.**

Your proposed Nemo 0.6B model can be the starting point, but we need to verify its exact identity, parameter count, and streaming support before committing to it. The evaluation should distinguish model limitations from implementation limitations and should not assume that the model will outperform existing systems.

## I. Recommended research framing

PROPOSED RESEARCH TITLE

# The Hidden Cost of Real-Time Speech Recognition: Evaluating Latency, Accuracy, and Robustness of Small ASR Models on CPU Edge Devices

**Central research question\:** Can a compact streaming ASR model provide fast, accurate, and stable transcription on CPU-only devices when speech is difficult, rather than merely on clean benchmark recordings?

This framing fits the challenge because it identifies a measurable blind spot, supports a reproducible experiment, and leads to a concrete engineering proposal.

## II. Model selection

The model I recommend starting with is NVIDIA's Nemotron Speech Streaming EN 0.6B\.

[![Nemotron Speech ASR Released with Low Latency and High Throughput | Piotr Żelasko posted on the topic | LinkedIn](https://images.openai.com/static-rsc-4/9kQOadOm--Cy9iDoZoVmiexvvkUXlGMdEUuH0uPn7QX_Zk8Y3nb6E1OSuVyt2HbRyUBVO02jFIcD_7rvWBQp3DmHw2Bimcgob1_byf5rSWO6CuFfsVNzpY5V5B16F3e1JuREX0Q0IkPyShdy6pqEfUDmdnT1RULbMrtuz-0MOOadSy_8wkhh64vwuLtWXIAy?purpose=fullsize)](https://www.linkedin.com/posts/piotr-%C5%BCelasko-937b33102_nvidianemotron-speech-streaming-en-06b-activity-7414296628972601344-M-5s)

**[Official Hugging Face model](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b?utm_source=chatgpt.com)**

- Approximately 600 million parameters, within the challenge's required range.
- FastConformer encoder with an RNN-T transducer.
- Designed for cache-aware streaming inference.
- English speech recognition with punctuation and capitalization support. \[[nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b?utm_source=chatgpt.com); [NeMo-Speech.cpp/docs/asr/models.md at main · NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/models.md?utm_source=chatgpt.com); [ASR Model Checkpoints](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_checkpoints.html?utm_source=chatgpt.com)\]

An important distinction: the official checkpoint supports streaming, but its default streaming configuration may wait for future audio before emitting results. We must measure actual observed latency rather than treating the model's nominal chunk duration as its TTFT. \[[transcribe.cpp/docs/models/nemotron-speech-streaming-en-0.6b.md at main · handy-computer/transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/nemotron-speech-streaming-en-0.6b.md?utm_source=chatgpt.com); [Speech Recognition](https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/speech-recognition/?utm_source=chatgpt.com)\]

## III. What exactly should we investigate?

I would build the evaluation around four hypotheses.

| Hypothesis | Test |
|--------|--------|
| H1: Low computational latency does not guarantee low perceived latency. | Compare time to first token (TTFT), first stable text, and finalization delay. |
| H2: Faster streaming configurations may sacrifice recognition accuracy. | Sweep supported lookahead settings and measure WER/CER. |
| H3: Average performance conceals failures on difficult speech. | Report accuracy by speech difficulty, speaker subgroup where documented, and noise condition. |
| H4: Quantization improves practical edge deployment, but its accuracy cost varies by condition. | Compare supported FP32/FP16 and quantized runtimes on the same audio and CPU. |

These are hypotheses to test, not conclusions to assume.

### The distinctive blind spot

**Conventional ASR evaluation often reduces performance to the final transcript, even though a streaming user experiences a sequence of partial transcripts over time.**

Two models can produce identical final transcripts but behave very differently: one may produce useful text quickly, while the other waits, revises its output repeatedly, or only becomes accurate near the end of the utterance.

This becomes especially important for assistive communication, live captioning, and edge devices with limited compute. Your study should measure both *what the model recognizes* and *when that information becomes usable*\.

## IV. Experimental protocol

## Evaluation pipeline

**1\. Select and document datasets**

Clean speech, noisy speech, and dysarthric or otherwise difficult speech

**2\. Run the same model on CPU**

Fixed hardware, software versions, audio format, and decoding configuration

**3\. Capture the complete streaming trace**

Audio arrival, first output, partial revisions, stable output, final transcript

**4\. Calculate accuracy and latency**

WER, CER, TTFT, real-time factor, stable-prefix delay, and memory use

**5\. Identify the failure mechanism**

Determine whether the bottleneck comes from model lookahead, decoding, CPU compute, or speech difficulty

### Metrics that should be reported

- **WER and CER\:** final transcript quality.
- **TTFT\:** time from the start of the defined audio stream to the first non-empty transcript output.
- **Time to first stable token\:** when the first emitted token is no longer revised, using a documented stability rule.
- **Real-time factor (RTF)\:** processing time divided by audio duration; lower is better.
- **Peak RAM\:** practical memory demand on an edge device.
- **Revision rate\:** how often previously emitted partial text changes, if the runtime exposes partial hypotheses.
- **p50 and p95 latency\:** typical and tail behaviour across recordings.

Record model-internal lookahead separately from observed wall-clock latency. Also specify whether TTFT includes audio buffering, feature extraction, model loading, and initialization. Report both cold-start and warmed-up performance if edge deployment is the intended use case.

## V. Dataset strategy

Your prior work on streaming ASR for dysarthric speech gives this project a more distinctive starting point than another generic clean-speech leaderboard. In that work, you evaluated streaming architectures and explored CPU-compatible inference. The new project should reuse relevant lessons without simply repeating the same experiment.

I suggest three evaluation groups\:

1. **Clean reference set\:** LibriSpeech test-clean, for a reproducible baseline.
2. **Challenging speech set\:** a public dysarthric-speech benchmark, with speaker-disjoint evaluation wherever the dataset permits it.
3. **Stress-test set\:** documented noisy or difficult recordings, preferably using a public dataset rather than inventing test conditions after seeing the results.

Keep dataset identities, splits, transcript normalization, and preprocessing fixed across every configuration. Do not pool the scores into one headline number without also reporting each dataset separately.

## VI. Path toward a novel contribution

A benchmark alone may be insufficiently novel. The stronger contribution would be a **latency-robustness evaluation framework for streaming ASR on resource-constrained hardware**, followed by a method to improve the trade-off.

A plausible proposed method is a *difficulty-aware streaming policy*\: use a low-lookahead configuration when speech is straightforward, but allow more context when confidence or acoustic evidence indicates that the current output is unreliable. This is a proposal, not an established result. It must be compared against fixed-lookahead baselines and evaluated for both accuracy and delay.

A further constraint: do not claim that a CPU benchmark establishes performance on every edge device. Report the processor, core count, thread configuration, RAM, power conditions where measurable, runtime, and quantization format.

## VII. Deliverables for the Fatima Institute challenge

Your public Hugging Face repository should contain\:

- `README.md` - the blind spot, motivation, findings, limitations, and proposed remedy.
- `dataset_manifest.csv` - dataset names, splits, licensing information, and sample counts.
- `evaluation_manifest.json` - model revision, runtime version, hardware, and decoding settings.
- `results.csv` - per-recording or per-condition accuracy and latency.
- `streaming_traces/` - timestamped partial hypotheses, subject to dataset licensing and privacy restrictions.
- `notebooks/` - the reproducible Colab notebook.
- `plots/` - accuracy-versus-latency curves and condition-specific results.
- `reproduce.sh` or equivalent instructions - the exact evaluation procedure.

Use a public Hugging Face repository or dataset repository that supports the required artifacts. Check the current Hub upload options and access controls; do not assume that every repository type is called a “Bucket.” The challenge requires a public, accessible link, so verify that the final repository can be opened without special permissions.

**Privacy requirement\:** do not publish identifiable voice recordings, speaker metadata, or derived traces that could expose sensitive information unless the dataset licence and consent terms explicitly permit it. Public availability does not automatically mean unrestricted redistribution.

## VIII. Immediate next step

Start with the official model and verify that the same checkpoint can be run in a genuinely streaming mode on CPU. Then establish one reproducible baseline before running the broader sweep.

One point to settle early: **are you planning to use the English-only `nvidia/nemotron-speech-streaming-en-0.6b` checkpoint, or the newer multilingual `nvidia/nemotron-3.5-asr-streaming-0.6b` model?** Both are in the 0.6B class, but the language coverage and runtime details differ. \[[NeMo-Speech.cpp/docs/asr/models.md at main · NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/models.md?utm_source=chatgpt.com); [ASR Model Checkpoints](https://docs.nvidia.com/nemo/speech/nightly/asr/asr_checkpoints.html?utm_source=chatgpt.com)\]

## Prompt:
10/19/2026, 4:52:52 PM

only en for now...but I dont get the direction...I want to make sure which one is fast,,a comparison,,does that full fill theire demand

## Response:
10/19/2026, 4:53:00 PM · gpt-6-thinking

> **Searched 3 websites**
>
> 
>
> Worked for 18s

## The simple answer: yes, but the comparison must be designed correctly.

The challenge asks you to do three things: identify an underexplored blind spot, systematically evaluate a qualifying open-weight model, and propose a path forward. A comparison of CPU inference speed can be part of that evaluation, but **a speed-only comparison is not enough to answer all three requirements**\.

Your clearest direction is\:

> **Find out which small ASR model is fastest on CPU, how quickly it begins producing usable text, and what accuracy it sacrifices to achieve that speed.**

That gives you a direct comparison and a research question that goes beyond ordinary benchmarking.

## 1\. Which models should you compare?

I checked the official Hugging Face model information. NVIDIA's English-only 0.6B checkpoint is a valid candidate, and the model card reports a 0.6B parameter count. \[[nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b?utm_source=chatgpt.com)\]

I would compare three models, rather than testing only one.

[![Nemotron Speech ASR Released with Low Latency and High Throughput | Piotr Żelasko posted on the topic | LinkedIn](https://images.openai.com/static-rsc-4/9kQOadOm--Cy9iDoZoVmiexvvkUXlGMdEUuH0uPn7QX_Zk8Y3nb6E1OSuVyt2HbRyUBVO02jFIcD_7rvWBQp3DmHw2Bimcgob1_byf5rSWO6CuFfsVNzpY5V5B16F3e1JuREX0Q0IkPyShdy6pqEfUDmdnT1RULbMrtuz-0MOOadSy_8wkhh64vwuLtWXIAy?purpose=fullsize)](https://www.linkedin.com/posts/piotr-%C5%BCelasko-937b33102_nvidianemotron-speech-streaming-en-06b-activity-7414296628972601344-M-5s)

**1\. Nemotron Speech Streaming EN 0.6B - primary model**

(Challenge model)

[huggingface.co](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b?utm_source=chatgpt.com)

Use this model for the main investigation. Test how its latency and accuracy change with supported streaming lookahead settings and CPU quantization. \[[nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b?utm_source=chatgpt.com); [transcribe.cpp/docs/models/nemotron-speech-streaming-en-0.6b.md at main · handy-computer/transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/nemotron-speech-streaming-en-0.6b.md?utm_source=chatgpt.com)\]

---

[![nvidia/nemotron-speech-streaming-en-0.6b · Hugging Face](https://images.openai.com/static-rsc-4/_aqDF_xs_sCmUS6la1M2vgw4dkILO4B3okGWvBvP77Nmi-CizlT0kN7MO9tzST5Bg0sEE2xN5-fl2DAHH7-H2YDvqBYQ-BjaqWyeR1jKbhw7wFmYDcoMgBwuvSkYxFJBGl_Bc9ekwbZWuRJny5utZ6ZWQh-URmMqocX3jOkASrgwbK1F6aAd68vdZP45Thmo?purpose=fullsize)](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)

**2\. NeMo FastConformer CTC - comparison baseline**

[docs.nvidia.com](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html?utm_source=chatgpt.com)

Compare a streaming CTC configuration against the primary RNN-T model. Confirm the checkpoint, streaming implementation, and parameter count before including it in the final comparison.

---

[![Checking your browser - reCAPTCHA](https://images.openai.com/static-rsc-4/8msaEJ_W8nkJ1kMpHc1bBeSWKtmF4NPUm0emBt1pZlIccCFgHcfVvQ8h-nXK_r6qhawJ0lvoSa-itz7dywbHcGyzwW-VQdRrzoE9wDpPXiaUfxxUuInul5IK4-jL8ImuqaGrk-EG4XqUh2uL9wjTwMLyMgxk5oKC5ISGjqqPu-O2GQhelnnu4xNQBYSwwb9D?purpose=fullsize)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12101685/)

**3\. Streaming Zipformer - efficient-model baseline**

[github.com](https://github.com/k2-fsa/icefall?utm_source=chatgpt.com)

This provides a useful reference for streaming ASR efficiency, particularly given your previous work with Zipformer. Fix the checkpoint and runtime before comparing results.

The two comparison systems do not need to satisfy the challenge's parameter constraint if they are only baselines. The model you formally evaluate for the challenge must satisfy it.

## 2\. What exactly will you measure?

Run every model on the **same CPU, same recordings, same audio chunks, and equivalent decoding conditions**\.

| Measurement | What it tells you |
|--------|--------|
| TTFT | How long the user waits for the first text |
| Finalization latency | How long until the final transcript is available |
| RTF | How much computation is needed relative to audio duration |
| WER and CER | How accurate the transcript is |
| Peak RAM | Whether deployment is practical on an edge device |
| Partial-output revisions | Whether early text is subsequently changed |

Do not confuse TTFT with time to first *correct* or stable token. Measure these separately when the output interface allows it.

Your final results might look like this\:

| Model | TTFT ↓ | RTF ↓ | WER ↓ | Peak RAM ↓ |
|--------|--------|--------|--------|--------|
| Nemotron 0.6B | Measure | Measure | Measure | Measure |
| FastConformer CTC | Measure | Measure | Measure | Measure |
| Streaming Zipformer | Measure | Measure | Measure | Measure |

These are placeholders, not results. We should not declare a winner before running the tests.

## 3\. Where is the novelty?

The comparison answers *which model is fastest*\. The research contribution should explain **why the fastest model may not be the best model for real users**\.

For example, Nemotron might emit text quickly on clean speech but require more time or produce more errors on dysarthric speech. Alternatively, a different model may be slower initially but produce more accurate text earlier in the conversation. Both outcomes are worth investigating.

I would formulate the blind spot as\:

**The missing dimension in conventional ASR evaluation is the relationship between early-output usefulness, recognition accuracy, and computational cost under realistic streaming conditions.**

To test it, use two evaluation stages\:

1. **General-purpose baseline\:** clean English speech to establish ordinary latency and accuracy.
2. **Difficult-speech evaluation\:** dysarthric speech or another documented difficult-speech dataset to test whether the latency-accuracy relationship changes.

The second stage is what makes the project more distinctive than a generic speed leaderboard.

## 4\. Does this fulfil the Fatima Institute challenge?

**Question 1 - Identify a blind spot**

Explain why final-transcript accuracy and average inference speed fail to capture the real-time usability of streaming ASR.

**Question 2 - Evaluate an open-weight model**

Run Nemotron 0.6B systematically on a GPU platform for model setup if necessary, but use CPU for the actual edge-device benchmarks. Record results, experimental settings, and failures.

**Question 3 - Propose a path forward**

Develop a specific proposal, such as adaptive lookahead that trades additional context for improved recognition when the audio is difficult. Evaluate it against a fixed-lookahead baseline if time permits.

**Final requirement - Public Hugging Face artifacts**

Publish your dataset manifest, evaluation scripts, measurements, plots, and README. Include audio only where redistribution is permitted. The repository must be publicly accessible.

## 5\. My recommendation

**Proceed with Nemotron 0.6B as the primary model and compare it against FastConformer CTC and streaming Zipformer.** First establish the CPU benchmark, then investigate the latency-accuracy trade-off on difficult speech. Do not start by designing a new architecture or fine-tuning a model; first determine exactly where the failure occurs.

One final distinction matters: the challenge calls for a frontier open-weight model that captures the behaviour of current frontier models. Parameter count alone does not establish that status. Your submission should justify the selected checkpoint using its published capabilities and relevant evidence, while acknowledging that it is a specialised ASR model rather than a general-purpose language model.

That is a coherent and feasible project, and it addresses the challenge substantially better than reporting CPU speed alone.

## Prompt:
10/19/2026, 6:06:45 PM

take it

[Attachment: colaB_neomoV1.pdf]

## Response:
10/19/2026, 6:06:49 PM · gpt-6-thinking

> **Extracting PDF text and page count**
>
>
> **Rendering page 13 for exception inspection**
>
>
> **Extracted PDF text and inspected rendered page 13**
>
> 
>
> Worked for 48s

I reviewed your uploaded Colab notebook, *colaB\_neomoV1.pdf*\. It shows that you have already made substantial progress: the model loads on CPU, you have selected 50 audio clips, and you have begun testing streaming lookahead settings. fileciteturn0file0

## What your notebook has established

- **Model loaded\:** Your code uses `nvidia/nemotron-3.5-asr-streaming-0.6b` on CPU.
- **Hardware recorded\:** Intel Xeon CPU at 2.20 GHz, with one PyTorch thread configured.
- **Dataset\:** You selected 50 clips from an English dysarthric/healthy speech dataset, with a manifest recording speech status, gender, duration, and transcript.
- **Streaming test\:** You tested lookahead values of 0, 1, 3, 6, and 13. The notebook shows that a lookahead of 1 is unsupported and that the model accepts 0, 3, 6, and 13.
- **Initial latency observations\:** Your displayed results include approximately 320 ms, 568 ms, and 1120 ms for some configurations, although the notebook does not establish a complete, validated comparison across the dataset.

The critical problem is at the end of the notebook: the streaming implementation raises an exception. Consequently, **we do not yet have a valid, completed benchmark**, even though the model itself loads successfully. fileciteturn0file0

## What we should do next

1. **Fix the streaming implementation.** The traceback on page 13 is truncated, so the precise failing line cannot yet be established.
2. **Run a smoke test on five clips** using supported lookahead settings: 0, 3, 6, and 13.
3. **Benchmark the 50 selected clips** only after the smoke test passes.
4. **Compare against FastConformer and Zipformer** on the same clips and CPU hardware.
5. **Export the results to CSV** with per-clip TTFT, finalization latency, RTF, WER, and CER.

Please upload the actual `.ipynb` notebook from Colab (**File → Download → Download .ipynb**\), rather than only the PDF screenshots. That will let me inspect and repair the executable code instead of guessing at the truncated error.

One important point: the current notebook uses **Nemotron 3.5 ASR Streaming 0.6B**, not necessarily the earlier Nemotron checkpoint we discussed. We should keep the model already loaded as the primary candidate unless testing reveals a concrete reason to switch.

## Prompt:
10/19/2026, 6:09:54 PM

gimme code

## Response:
10/19/2026, 6:10:06 PM · gpt-6-thinking

> ****
>
> 
>
> **Refining transcription code**
>
> I’m checking the Colab setup and focusing on reliable streaming. I’ll ensure generation errors don’t stall the iterator, measure time to first output, keep CPU execution controlled, and pass audio features correctly.
>
> **Building benchmark cells**
>
> I’m preparing Colab-ready code that handles model loading, audio resampling, streaming exceptions, time-to-first-token, total latency, real-time factor, WER, and CER while using the existing dataset and manifest variables.
>
> **Refined transcription code and built benchmark cells**
>
> 
>
> **Searched 2 websites**
>
> 
>
> Worked for 2m 40s

Below is the corrected Colab code for your existing project. It keeps **Nemotron 3.5 ASR Streaming 0.6B**, English transcription, CPU-only inference, your 50-clip manifest, and lookahead values of `0, 3, 6, 13`\.

The key correction follows the model's documented streaming API: pass the initial chunk inputs and a generator of subsequent audio features to `model.generate()`\. This should address the implementation problem shown at the end of your PDF. \[[transformers/docs/source/en/model\_doc/nemotron3\_5\_asr.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nemotron3_5_asr.md?utm_source=chatgpt.com); [Nemotron 3.5 ASR](https://huggingface.co/docs/transformers/v5.13.0/model_doc/nemotron3_5_asr?utm_source=chatgpt.com)\]

Run the cells in order. The benchmark cells assume that your existing dataset object `ds` and `/content/manifest.csv` are still available in the Colab session.

## Cell 1 - Install dependencies

```python
!pip install -q -U "transformers>=5.13.0" accelerate jiwer librosa soundfile pandas
```

## Cell 2 - Load the model on CPU

```python
import os
import gc
import time
import threading
import traceback
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoProcessor,
    AutoModelForRNNT,
    TextIteratorStreamer,
)

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# CPU-only benchmark
DEVICE = "cpu"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForRNNT.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
).eval()

print("Model:", MODEL_ID)
print("Device:", next(model.parameters()).device)
print("Dtype:", next(model.parameters()).dtype)
print("CPU threads:", torch.get_num_threads())
print("Sample rate:", processor.feature_extractor.sampling_rate)
print("Supported lookahead latencies:",
      processor.supported_streaming_latencies_ms)
```

## Cell 3 - Validate the dataset and manifest

```python
from pathlib import Path

MANIFEST_PATH = "/content/manifest.csv"

assert "ds" in globals(), (
    "Dataset variable 'ds' is missing. Rerun your dataset-loading cell."
)
assert Path(MANIFEST_PATH).exists(), (
    "manifest.csv is missing. Rerun your manifest-creation cell."
)

manifest = pd.read_csv(MANIFEST_PATH)

print("Dataset rows:", len(ds))
print("Manifest rows:", len(manifest))
print("Manifest columns:", manifest.columns.tolist())

required = {"row", "speech_status", "duration"}
missing = required - set(manifest.columns)
assert not missing, f"Missing manifest columns: {missing}"

# Use at most 50 clips for the first benchmark.
manifest = manifest.head(50).copy().reset_index(drop=True)

print("\nSpeech-status counts:")
print(manifest["speech_status"].value_counts(dropna=False))

display(manifest.head())
```

## Cell 4 - Safely retrieve and prepare audio

This supports the audio formats commonly returned by Hugging Face Datasets, including decoded audio dictionaries and newer audio-decoder objects.

```python
import librosa

TARGET_SR = processor.feature_extractor.sampling_rate

def get_audio_from_dataset(dataset_row):
    audio_obj = ds[int(dataset_row)]["audio"]

    # Newer Hugging Face audio decoder objects
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        audio = samples.data
        sr = int(samples.sample_rate)

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32)

        # Convert channel-first multichannel audio to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=0) if audio.shape[0] <= 8 else audio.mean(axis=1)

    # Older Hugging Face Audio dictionaries
    elif isinstance(audio_obj, dict):
        sr = int(audio_obj.get("sampling_rate", TARGET_SR))
        audio = audio_obj.get("array")

        if audio is None and audio_obj.get("path"):
            import soundfile as sf
            audio, sr = sf.read(audio_obj["path"], dtype="float32")

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

    else:
        raise TypeError(
            f"Unsupported audio object: {type(audio_obj)}"
        )

    audio = np.squeeze(audio).astype(np.float32)

    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError(f"Invalid audio shape: {audio.shape}")

    if sr != TARGET_SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR,
        ).astype(np.float32)

    return audio, TARGET_SR

# Test one clip before running the benchmark.
test_audio, test_sr = get_audio_from_dataset(manifest.iloc[0]["row"])

print("Audio samples:", len(test_audio))
print("Sample rate:", test_sr)
print("Duration:", round(len(test_audio) / test_sr, 3), "seconds")
```

If this cell raises an audio-decoding error, stop there and share the traceback. The correct fix depends on the actual audio object returned by your dataset.

## Cell 5 - Correct streaming inference and measure TTFT

This function uses the documented streaming feature generator and captures the first emitted text, total processing time, and output transcript. It also surfaces exceptions from the generation thread instead of silently losing them.

```python
def streaming_transcribe(audio, lookahead):
    processor.set_num_lookahead_tokens(int(lookahead))

    sr = processor.feature_extractor.sampling_rate
    audio = np.asarray(audio, dtype=np.float32)

    # Include first-chunk feature extraction in the measured TTFT.
    t_start = time.perf_counter()

    first_n = processor.num_samples_first_audio_chunk

    first_inputs = processor(
        audio[:first_n],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        language="en-US",
        return_tensors="pt",
    )
    first_inputs = first_inputs.to(
        model.device,
        dtype=model.dtype,
    )

    def input_features_generator():
        # Initial chunk
        yield first_inputs.input_features[
            :, :processor.num_mel_frames_first_audio_chunk, :
        ]

        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft

        start_idx = mel_frame_idx * hop_length - n_fft // 2

        while True:
            end_idx = start_idx + processor.num_samples_per_audio_chunk

            if end_idx >= len(audio):
                break

            chunk_inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                language="en-US",
                return_tensors="pt",
            )
            chunk_inputs = chunk_inputs.to(
                model.device,
                dtype=model.dtype,
            )

            yield chunk_inputs.input_features

            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        timeout=300,
    )

    errors = []

    def worker():
        try:
            with torch.inference_mode():
                model.generate(
                    **first_inputs,
                    input_features=input_features_generator(),
                    streamer=streamer,
                )
        except Exception as exc:
            errors.append((exc, traceback.format_exc()))
            streamer.end()

    thread = threading.Thread(target=worker, daemon=True)

    first_text_time = None
    text_parts = []

    thread.start()

    for piece in streamer:
        now = time.perf_counter()

        if piece and piece.strip() and first_text_time is None:
            first_text_time = now - t_start

        text_parts.append(piece)

    thread.join(timeout=300)

    if thread.is_alive():
        raise TimeoutError("Streaming generation exceeded 300 seconds.")

    if errors:
        raise RuntimeError(errors[0][1]) from errors[0][0]

    total_time = time.perf_counter() - t_start
    transcript = "".join(text_parts).strip()
    audio_duration = len(audio) / sr

    return {
        "prediction": transcript,
        "ttft_s": first_text_time,
        "total_time_s": total_time,
        "audio_duration_s": audio_duration,
        "rtf": total_time / audio_duration if audio_duration else np.nan,
        "lookahead": int(lookahead),
        "configured_latency_ms": processor.streaming_latency_ms,
    }

# Smoke test: one recording, lowest supported lookahead.
smoke = streaming_transcribe(test_audio, lookahead=0)

print("Transcript:", smoke["prediction"])
print("Configured streaming latency:", smoke["configured_latency_ms"], "ms")
print("Measured TTFT:", smoke["ttft_s"], "seconds")
print("Total processing time:", smoke["total_time_s"], "seconds")
print("RTF:", smoke["rtf"])
```

The model's documented configured streaming latencies are 80, 320, 560, and 1,120 ms for lookahead values 0, 3, 6, and 13, respectively. These are configuration latencies, **not measured CPU TTFT**\. \[[transformers/docs/source/en/model\_doc/nemotron3\_5\_asr.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nemotron3_5_asr.md?utm_source=chatgpt.com)\]

## Cell 6 - Run the 50-clip benchmark

This runs all four supported configurations on the same recordings and computes WER/CER against the reference transcripts.

```python
import re
from jiwer import wer, cer
from IPython.display import display

LOOKAHEAD_VALUES = [0, 3, 6, 13]

# Find the reference-transcript column.
gt_candidates = ["transcription", "transcript", "text", "sentence"]
GT_COL = next(
    (c for c in gt_candidates if c in manifest.columns),
    None,
)
assert GT_COL is not None, (
    f"No transcript column found. Available columns: {manifest.columns.tolist()}"
)

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())

# Load and cache the audio once so every configuration uses identical audio.
audio_cache = {}
for _, row in manifest.iterrows():
    idx = int(row["row"])
    try:
        audio_cache[idx] = get_audio_from_dataset(idx)[0]
    except Exception as exc:
        print(f"Audio load failed for row {idx}: {exc}")

print(f"Audio clips ready: {len(audio_cache)} / {len(manifest)}")

# Warm-up: one clip per configuration. Excluded from reported results.
first_idx = next(iter(audio_cache))

for lookahead in LOOKAHEAD_VALUES:
    try:
        _ = streaming_transcribe(
            audio_cache[first_idx],
            lookahead,
        )
        print(f"Warm-up passed: lookahead={lookahead}")
    except Exception as exc:
        print(f"Warm-up FAILED: lookahead={lookahead}: {exc}")
        raise

results = []

for lookahead in LOOKAHEAD_VALUES:
    print(f"\nRunning lookahead={lookahead}")

    for i, row in manifest.iterrows():
        idx = int(row["row"])

        if idx not in audio_cache:
            continue

        reference = str(row[GT_COL])

        try:
            out = streaming_transcribe(
                audio_cache[idx],
                lookahead,
            )

            prediction = out["prediction"]

            ref_norm = normalize_text(reference)
            pred_norm = normalize_text(prediction)

            results.append({
                "row": idx,
                "speech_status": row["speech_status"],
                "gender": row.get("gender", None),
                "duration_s": out["audio_duration_s"],
                "lookahead": lookahead,
                "configured_latency_ms": out["configured_latency_ms"],
                "ttft_ms": (
                    out["ttft_s"] * 1000
                    if out["ttft_s"] is not None
                    else np.nan
                ),
                "total_time_s": out["total_time_s"],
                "rtf": out["rtf"],
                "reference": reference,
                "prediction": prediction,
                "wer": wer(ref_norm, pred_norm),
                "cer": cer(ref_norm, pred_norm),
                "error": None,
            })

        except Exception as exc:
            results.append({
                "row": idx,
                "speech_status": row["speech_status"],
                "gender": row.get("gender", None),
                "duration_s": len(audio_cache[idx]) / TARGET_SR,
                "lookahead": lookahead,
                "configured_latency_ms": processor.streaming_latency_ms,
                "ttft_ms": np.nan,
                "total_time_s": np.nan,
                "rtf": np.nan,
                "reference": reference,
                "prediction": "",
                "wer": np.nan,
                "cer": np.nan,
                "error": repr(exc),
            })

        if (i + 1) % 10 == 0:
            print(f"  Processed manifest row {i + 1}/{len(manifest)}")

results_df = pd.DataFrame(results)

output_path = "/content/nemotron_cpu_streaming_results.csv"
results_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)
print("Result rows:", len(results_df))
print("Failed runs:", results_df["error"].notna().sum())

display(results_df.head(10))
```

## Cell 7 - Compare speed and accuracy

```python
successful = results_df[results_df["error"].isna()].copy()

summary = (
    successful
    .groupby(["lookahead", "speech_status"], dropna=False)
    .agg(
        clips=("row", "nunique"),
        mean_TTFT_ms=("ttft_ms", "mean"),
        median_TTFT_ms=("ttft_ms", "median"),
        p95_TTFT_ms=("ttft_ms", lambda x: x.quantile(0.95)),
        mean_RTF=("rtf", "mean"),
        median_RTF=("rtf", "median"),
        mean_WER=("wer", "mean"),
        mean_CER=("cer", "mean"),
    )
    .reset_index()
)

display(summary.round(4))

summary.to_csv(
    "/content/nemotron_cpu_summary.csv",
    index=False,
)

print("Saved: /content/nemotron_cpu_summary.csv")
```

## Cell 8 - Plot the latency-accuracy trade-off

```python
import matplotlib.pyplot as plt

overall = (
    successful
    .groupby("lookahead")
    .agg(
        TTFT_ms=("ttft_ms", "median"),
        WER=("wer", "mean"),
        CER=("cer", "mean"),
        RTF=("rtf", "median"),
    )
    .reset_index()
)

fig, ax = plt.subplots()
ax.plot(overall["TTFT_ms"], overall["WER"], marker="o")

for _, r in overall.iterrows():
    ax.annotate(
        f"Lookahead {int(r['lookahead'])}",
        (r["TTFT_ms"], r["WER"]),
    )

ax.set_xlabel("Median measured TTFT (ms)")
ax.set_ylabel("Mean WER")
ax.set_title("Nemotron CPU: Latency-Accuracy Trade-off")
ax.grid(True)
plt.show()

display(overall.round(4))
```

## Cell 9 - Download the results

```python
from google.colab import files

files.download("/content/nemotron_cpu_streaming_results.csv")
files.download("/content/nemotron_cpu_summary.csv")
```

### Important experimental notes

- The reported TTFT includes first-chunk feature extraction and time to the first non-empty streamed text, but excludes dataset audio loading and resampling.
- RTF includes streaming feature extraction and model generation. An RTF below 1 means processing completed faster than the duration of the audio; it does not, by itself, prove real-time streaming behaviour.
- WER/CER here use normalized lowercase text with punctuation removed. Apply exactly the same normalization to every comparison model.
- The code reports **mean per-clip WER/CER**, not corpus-level WER/CER. For the final paper, calculate corpus-level metrics as well.
- The notebook currently does not measure peak RAM or stable-token latency. Those should be added before claiming a complete edge-device evaluation.

**Next\:** first run Cells 1-5. If the smoke test passes, run the full benchmark. If it fails, send me the complete traceback rather than the PDF screenshot so I can correct the specific issue.

## Prompt:
10/19/2026, 11:13:39 PM

I started a new note book ---------------------------------------------------------------------------&#x20;
AssertionError                            Traceback (most recent call last)&#x20;
[/tmp/ipykernel_618/3297777944.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \<cell line: 0>()&#x20;**&#x20;     6**     "Dataset variable 'ds' is missing. Rerun your dataset-loading cell."&#x20;**&#x20;     7** ) ----> 8 assert Path(MANIFEST_PATH).exists(), (&#x20;**&#x20;     9**     "manifest.csv is missing. Rerun your manifest-creation cell."&#x20;**&#x20;    10** ) &#x20;
AssertionError: manifest.csv is missing. Rerun your manifest-creation cell.

## Response:
10/19/2026, 11:13:54 PM · gpt-6-thinking

> **Rebuilding notebook state**
>
> I’m recreating the missing setup from scratch: loading `abner/TORGO-dataset`, inspecting its columns, and generating the metadata manifest before running later cells.
>
> **Rebuilt notebook state**
>
> 
>
> ****
>
> 
>
> Worked for 20s

That error is happening because you started a **new Colab notebook**\. The variables `ds` and `/content/manifest.csv` from your previous notebook are not available in the new session.

You do not need to restart the project. Run the following cells first, then continue with the model benchmark.

## Cell 1 - Install dataset dependencies

```python
!pip -q install -U datasets pandas numpy librosa soundfile
```

## Cell 2 - Load the dataset and create the 50-clip manifest

This uses the dataset ID shown in your previous notebook, `abner/TORGO-dataset`, and creates a balanced sample of dysarthric and healthy speech with matched transcripts where possible.

```python
import os
import numpy as np
import pandas as pd
from datasets import load_dataset

# Load the dataset used in your previous notebook.
DATASET_ID = "abner/TORGO-dataset"

ds = load_dataset(DATASET_ID, split="train")

print("Dataset loaded successfully")
print("Number of recordings:", len(ds))
print("Columns:", ds.column_names)

# Check that the expected columns exist.
required = {"audio", "transcription", "speech_status", "gender"}
missing = required - set(ds.column_names)

if missing:
    raise ValueError(
        f"Dataset is missing expected columns: {missing}. "
        f"Available columns: {ds.column_names}"
    )

# Create metadata without copying the audio into a DataFrame.
meta = ds.remove_columns("audio").to_pandas()
meta["row"] = np.arange(len(meta))

meta["transcription"] = (
    meta["transcription"].fillna("").astype(str).str.strip()
)
meta["speech_status"] = (
    meta["speech_status"].fillna("").astype(str).str.lower().str.strip()
)
meta["n_words"] = meta["transcription"].str.split().str.len()

# Get durations from the dataset metadata when available.
if "duration" in meta.columns:
    meta["duration"] = pd.to_numeric(
        meta["duration"], errors="coerce"
    )
else:
    meta["duration"] = np.nan

print("\nSpeech categories:")
print(meta["speech_status"].value_counts())

print("\nDataset duration summary:")
print(meta["duration"].describe())
```

## Cell 3 - Create a reproducible manifest

This selects 25 matched transcripts, with one dysarthric and one healthy recording per transcript when the dataset contains suitable recordings for both groups.

```python
SEED = 42
N_PAIRS = 25

# Basic filtering: usable transcript and reasonable recording duration.
eligible = meta[
    meta["speech_status"].isin(["dysarthria", "healthy"])
    & (meta["n_words"] >= 4)
    & meta["duration"].between(1.0, 30.0)
    & (meta["transcription"] != "")
].copy()

# Match transcript strings case-insensitively.
eligible["match_text"] = (
    eligible["transcription"].str.lower().str.strip()
)

groups = {
    status: set(
        eligible.loc[
            eligible["speech_status"] == status, "match_text"
        ]
    )
    for status in ["dysarthria", "healthy"]
}

shared = sorted(groups["dysarthria"] & groups["healthy"])

print("Eligible recordings:", len(eligible))
print("Shared transcript texts:", len(shared))

if len(shared) < N_PAIRS:
    raise ValueError(
        f"Only {len(shared)} shared transcripts found; "
        f"cannot create {N_PAIRS} matched pairs. "
        "Reduce N_PAIRS or inspect the dataset filters."
    )

rng = np.random.default_rng(SEED)
selected_texts = rng.choice(
    shared, size=N_PAIRS, replace=False
)

selected_rows = []

for text in selected_texts:
    for status in ["dysarthria", "healthy"]:
        candidates = eligible[
            (eligible["match_text"] == text)
            & (eligible["speech_status"] == status)
        ]

        # Choose one recording reproducibly.
        chosen = candidates.iloc[
            int(rng.integers(0, len(candidates)))
        ]

        selected_rows.append(chosen)

manifest = pd.DataFrame(selected_rows).reset_index(drop=True)

# Keep the columns needed for evaluation.
manifest = manifest[
    [
        "row",
        "speech_status",
        "gender",
        "duration",
        "transcription",
    ]
].copy()

os.makedirs("/content", exist_ok=True)

MANIFEST_PATH = "/content/manifest.csv"
manifest.to_csv(MANIFEST_PATH, index=False)

print("\nManifest saved:", MANIFEST_PATH)
print("Total clips:", len(manifest))
print("\nClips per category:")
print(manifest["speech_status"].value_counts())

display(manifest.head(10))
```

## Cell 4 - Verify everything before benchmarking

```python
assert "ds" in globals()
assert os.path.exists("/content/manifest.csv")

manifest = pd.read_csv("/content/manifest.csv")

assert len(manifest) == 50
assert manifest["row"].between(0, len(ds) - 1).all()
assert manifest["transcription"].notna().all()

print("Dataset and manifest are ready.")
print("Dataset size:", len(ds))
print("Manifest size:", len(manifest))
print("Transcript column:", "transcription")
print("Audio column:", "audio")
```

Now return to the previous benchmark code. In its dataset-validation cell, use the `ds` and `manifest` variables created above. You should no longer receive the missing-manifest assertion error.

**Important\:** if Cell 2 reports that the dataset cannot be found or accessed, stop there and share that exact error. We need to verify the dataset identifier before changing anything else.

## Prompt:
10/19/2026, 11:20:41 PM

use !pip install -q datasets
from datasets import load_dataset, Audio

ds = load_dataset("abnerh/TORGO-database", split="train")
print(ds)
print(ds.features)
print(ds[0].keys())

## Response:
10/19/2026, 11:20:50 PM · gpt-6-thinking

> **Preparing dataset setup**
>
> I’ll use the provided dataset package and loading code as the starting point. Since the exact feature schema may vary, I’m structuring the next step to inspect available columns before building the paired manifest.
>
> **Prepared dataset setup**
>
> 
>
> **Searched 4 websites**
>
> 
>
> Worked for 15s

Use your exact dataset ID, `abnerh/TORGO-database`\. The Hugging Face dataset card confirms that it contains 16,552 English recordings with `audio`, `transcription`, `speech_status`, `gender`, and `duration` fields. \[[abnerh/TORGO-database](https://huggingface.co/datasets/abnerh/TORGO-database?utm_source=chatgpt.com)\]

Run these cells in your new Colab notebook.

## Cell 1 - Load your dataset

```python
!pip install -q datasets

from datasets import load_dataset, Audio

ds = load_dataset("abnerh/TORGO-database", split="train")

print(ds)
print(ds.features)
print(ds[0].keys())
```

## Cell 2 - Create your reproducible 50-audio manifest

This selects 25 dysarthric recordings and 25 healthy recordings, matching their transcripts where possible. It uses the exact column names from your dataset.

```python
import pandas as pd
import numpy as np
import os

SEED = 42
N_PAIRS = 25

# Extract metadata without copying the audio into the DataFrame.
meta = ds.remove_columns("audio").to_pandas()
meta["row"] = np.arange(len(meta))

meta["transcription"] = meta["transcription"].fillna("").astype(str).str.strip()
meta["speech_status"] = meta["speech_status"].astype(str).str.lower().str.strip()
meta["n_words"] = meta["transcription"].str.split().str.len()
meta["duration"] = pd.to_numeric(meta["duration"], errors="coerce")
meta["match_text"] = meta["transcription"].str.lower().str.strip()

# Select usable recordings.
eligible = meta[
    meta["speech_status"].isin(["dysarthria", "healthy"])
    & meta["n_words"].ge(4)
    & meta["duration"].between(1, 30)
    & meta["transcription"].ne("")
].copy()

# Find transcripts appearing in both groups.
dysarthric_texts = set(
    eligible.loc[
        eligible["speech_status"] == "dysarthria", "match_text"
    ]
)

healthy_texts = set(
    eligible.loc[
        eligible["speech_status"] == "healthy", "match_text"
    ]
)

shared_texts = sorted(dysarthric_texts & healthy_texts)

print("Eligible recordings:", len(eligible))
print("Shared transcripts:", len(shared_texts))

if len(shared_texts) < N_PAIRS:
    raise ValueError(
        f"Only {len(shared_texts)} matched transcripts found. "
        "Reduce N_PAIRS or adjust the filters."
    )

rng = np.random.default_rng(SEED)
chosen_texts = rng.choice(shared_texts, size=N_PAIRS, replace=False)

selected = []

for text in chosen_texts:
    for status in ["dysarthria", "healthy"]:
        candidates = eligible[
            (eligible["match_text"] == text)
            & (eligible["speech_status"] == status)
        ]

        selected_row = candidates.iloc[
            rng.integers(0, len(candidates))
        ]

        selected.append(selected_row)

manifest = pd.DataFrame(selected).reset_index(drop=True)

manifest = manifest[
    ["row", "speech_status", "gender", "duration", "transcription"]
]

MANIFEST_PATH = "/content/manifest.csv"
manifest.to_csv(MANIFEST_PATH, index=False)

print("\nManifest saved:", MANIFEST_PATH)
print("Total clips:", len(manifest))
print(manifest["speech_status"].value_counts())

display(manifest.head(10))
```

## Cell 3 - Verify before running Nemotron

```python
assert len(manifest) == 50
assert manifest["row"].between(0, len(ds) - 1).all()
assert os.path.exists("/content/manifest.csv")

print("Dataset and manifest are ready.")
print("Audio dataset rows:", len(ds))
print("Benchmark clips:", len(manifest))
print("Transcript column: transcription")
print("Audio column: audio")
```

**Important\:** this 50-clip sample is an initial benchmark, not a definitive scientific test set. The dataset has no official train/test split, so keep the seed and manifest fixed across all models and avoid using these same recordings to tune and report final performance without a separate evaluation set. \[[abnerh/TORGO-database](https://huggingface.co/datasets/abnerh/TORGO-database?utm_source=chatgpt.com)\]

Once these cells run successfully, continue with the Nemotron model-loading and streaming-inference cells. Your benchmark code should use `ds` and `manifest` directly-do not reload the dataset or recreate the manifest in a different format.

## Prompt:
10/19/2026, 11:23:23 PM

[transformers] Keyword argument \`language\` is not a valid argument for this processor and will be ignored.&#x20;

\---------------------------------------------------------------------------&#x20;

TypeError                                 Traceback (most recent call last)&#x20;

[/tmp/ipykernel_618/1183372792.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in worker()&#x20;**&#x20;    71**             with torch.inference_mode(): ---> 72                 model.generate(&#x20;**&#x20;    73**                     \*\*first_inputs, &#x20;

TypeError: transformers.models.nemotron_asr_streaming.generation_nemotron_asr_streaming.NemotronAsrStreamingGenerationMixin.generate() got multiple values for keyword argument 'input_features'&#x20;

The above exception was the direct cause of the following exception: &#x20;

RuntimeError                              Traceback (most recent call last)&#x20;

---

1 frames

---

[/tmp/ipykernel_618/1183372792.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \<cell line: 0>()&#x20;**&#x20;   118** &#x20;**&#x20;   119** # Smoke test: one recording, lowest supported lookahead. --> 120 smoke = streaming_transcribe(test_audio, lookahead=0)&#x20;**&#x20;   121** &#x20;**&#x20;   122** print("Transcript:", smoke["prediction"]) &#x20;

[/tmp/ipykernel_618/1183372792.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in streaming_transcribe(audio, lookahead)&#x20;**&#x20;   100** &#x20;**&#x20;   101**     if errors: --> 102         raise RuntimeError(errors[0][1]) from errors[0][0]&#x20;**&#x20;   103** &#x20;**&#x20;   104**     total_time = time.perf_counter() - t_start &#x20;

RuntimeError: Traceback (most recent call last):   File "/tmp/ipykernel_618/1183372792.py", line 72, in worker     model.generate(     \~\~\~\~\~\~\~\~\~\~\~\~\~\~^         \*\*first_inputs,         ^^^^^^^^^^^^^^^         input_features=input_features_generator(),         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^         streamer=streamer,         ^^^^^^^^^^^^^^^^^^     )     ^ TypeError: transformers.models.nemotron_asr_streaming.generation_nemotron_asr_streaming.NemotronAsrStreamingGenerationMixin.generate() got multiple values for keyword argument 'input_features'

## Response:
10/19/2026, 11:23:32 PM · gpt-6-thinking

The error is clear: `first_inputs` already contains `input_features`, and your code passes `input_features` a second time as a streaming generator. Python rejects the duplicate argument.

**Fix these two parts of your existing code.**

### 1\. Remove `language="en-US"` from processor calls

Change your processor calls to\:

```python
first_inputs = processor(
    audio[:first_n],
    sampling_rate=sr,
    is_streaming=True,
    is_first_audio_chunk=True,
    return_tensors="pt",
)
```

The warning means the processor does not accept that keyword. It is not the cause of the crash.

### 2\. Replace the `model.generate()` call inside `worker()`

Replace\:

```python
model.generate(
    **first_inputs,
    input_features=input_features_generator(),
    streamer=streamer,
)
```

with\:

```python
generation_inputs = {
    key: value
    for key, value in first_inputs.items()
    if key != "input_features"
}

model.generate(
    **generation_inputs,
    input_features=input_features_generator(),
    streamer=streamer,
)
```

This removes the duplicate keyword while preserving the other processor inputs.

**Then rerun the cell defining `streaming_transcribe()` and rerun the smoke test\:**

```python
smoke = streaming_transcribe(test_audio, lookahead=0)

print("Transcript:", smoke["prediction"])
print("TTFT:", smoke["ttft_s"])
print("Total processing time:", smoke["total_time_s"])
print("RTF:", smoke["rtf"])
```

If another error appears, send the complete traceback. We should get one clip working correctly before launching the 50-clip benchmark.

## Prompt:
10/19/2026, 11:27:19 PM

what to run next?

## Response:
10/19/2026, 11:27:24 PM · gpt-6-thinking

Run the following cells in this order in your new Colab notebook. **Do not rerun the installation or dataset-loading cells unless the runtime has restarted.**

## Step 1 - Rerun the corrected function

Run the cell containing your updated `streaming_transcribe()` function after applying the duplicate-argument fix.

## Step 2 - Run the smoke test

```python
smoke = streaming_transcribe(test_audio, lookahead=0)

print("Transcript:", smoke["prediction"])
print("TTFT (seconds):", smoke["ttft_s"])
print("Total processing time:", smoke["total_time_s"])
print("Audio duration:", smoke["audio_duration_s"])
print("RTF:", smoke["rtf"])
```

Check that it produces a transcript without an exception.

## Step 3 - Test all four lookahead settings on one clip

```python
smoke_results = []

for lookahead in [0, 3, 6, 13]:
    print(f"\nTesting lookahead={lookahead}")

    result = streaming_transcribe(test_audio, lookahead)

    smoke_results.append(result)

    print("Transcript:", result["prediction"])
    print("TTFT:", result["ttft_s"])
    print("RTF:", result["rtf"])

smoke_df = pd.DataFrame(smoke_results)
display(smoke_df)
```

This confirms whether each supported setting works before committing to the full benchmark.

## Step 4 - Run the 50-clip benchmark

If Steps 2 and 3 pass, run the previously provided **Cell 6 - Run the 50-clip benchmark**\. It evaluates all four settings against the same recordings and

## Prompt:
10/19/2026, 11:28:00 PM

/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py:1963: UserWarning: Using the model-agnostic default \`max_length\` (=1000000000) to control the generation length. We recommend setting \`max_new_tokens\` to control the maximum length of the generation.   warnings.warn(&#x20;

\---------------------------------------------------------------------------&#x20;

RuntimeError                              Traceback (most recent call last)&#x20;

[/tmp/ipykernel_618/1442377858.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in worker()&#x20;**&#x20;    75**                 } ---> 76                 model.generate(&#x20;**&#x20;    77**                     \*\*generation_inputs, &#x20;

---

12 frames

---

[/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in generate(self, inputs, generation_config, \*\*kwargs)&#x20;**&#x20;   227**             # Parakeet's generate() runs the decoding loop and assembles sequences + per-step durations. --> 228             outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)&#x20;**&#x20;   229**         finally: &#x20;

[/usr/local/lib/python3.13/dist-packages/transformers/models/parakeet/generation_parakeet.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in generate(self, inputs, generation_config, \*\*kwargs)&#x20;**&#x20;   255**  --> 256         outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)&#x20;**&#x20;   257**  &#x20;

[/usr/local/lib/python3.13/dist-packages/torch/utils/\_contextlib.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in decorate_context(\*args, \*\*kwargs)&#x20;**&#x20;   123**         with ctx_factory(): --> 124             return func(\*args, \*\*kwargs)&#x20;**&#x20;   125**  &#x20;

[/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in generate(self, inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, custom_generate, \*\*kwargs)&#x20;**&#x20;  2988**         # 9. Call generation mode -> 2989         result = decoding_method(&#x20;**&#x20;  2990**             self, &#x20;

[/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \_sample(self, input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer, \*\*model_kwargs)&#x20;**&#x20;  3218**                 prefill_consumed = True -> 3219                 model_kwargs = self.\_update_model_kwargs_for_generation(&#x20;**&#x20;  3220**                     outputs, &#x20;

[/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \_update_model_kwargs_for_generation(self, outputs, model_kwargs, \*args, \*\*kwargs)&#x20;**&#x20;    52**             try: ---> 53                 chunk = next(generator)&#x20;**&#x20;    54**             except StopIteration: &#x20;

[/tmp/ipykernel_618/1442377858.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in input_features_generator()&#x20;**&#x20;    41**  ---> 42             chunk_inputs = processor(&#x20;**&#x20;    43**                 audio[start_idx:end_idx], &#x20;

[/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/processing_nemotron_asr_streaming.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \_\_call\_\_(self, audio, text, sampling_rate, is_streaming, is_first_audio_chunk, \*\*kwargs)&#x20;**&#x20;   139**             # \`center=True\` for the first/offline chunk, \`center=False\` for subsequent streaming chunks. --> 140             inputs = self.feature_extractor(audio, center=bool(is_first_audio_chunk), \*\*output_kwargs["audio_kwargs"])&#x20;**&#x20;   141**         if text is not None: &#x20;

[/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \_\_call\_\_(self, raw_speech, truncation, pad_to_multiple_of, return_tensors, return_attention_mask, padding, max_length, sampling_rate, device, return_token_timestamps, center, is_last_audio_chunk, \*\*kwargs)&#x20;**&#x20;   260**  --> 261         input_features = self.\_torch_extract_fbank_features(input_features, device, center=center)&#x20;**&#x20;   262**         if center: &#x20;

[/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \_torch_extract_fbank_features(self, waveform, device, center)&#x20;**&#x20;   105**         window = torch.hann_window(self.win_length, periodic=False, device=device) --> 106         stft = torch.stft(&#x20;**&#x20;   107**             waveform, &#x20;

[/usr/local/lib/python3.13/dist-packages/torch/functional.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex, align_to_window)&#x20;**&#x20;   680**         input = input.view(input.shape[-signal_dim:]) --> 681     return \_VF.stft(  # type: ignore[attr-defined]&#x20;**&#x20;   682**         input, &#x20;

RuntimeError: stft(torch.FloatTensor[1, 0], n_fft=512, hop_length=160, win_length=400, window=torch.FloatTensor{[400]}, normalized=0, onesided=None, return_complex=1, align_to_window=None) : expected 0 < n_fft < 0, but got n_fft=400&#x20;

The above exception was the direct cause of the following exception: &#x20;

RuntimeError                              Traceback (most recent call last)&#x20;

[/tmp/ipykernel_618/1442377858.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \<cell line: 0>()&#x20;**&#x20;   122** &#x20;**&#x20;   123** # Smoke test: one recording, lowest supported lookahead. --> 124 smoke = streaming_transcribe(test_audio, lookahead=0)&#x20;**&#x20;   125** &#x20;**&#x20;   126** print("Transcript:", smoke["prediction"]) &#x20;

[/tmp/ipykernel_618/1442377858.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in streaming_transcribe(audio, lookahead)&#x20;**&#x20;   104** &#x20;**&#x20;   105**     if errors: --> 106         raise RuntimeError(errors[0][1]) from errors[0][0]&#x20;**&#x20;   107** &#x20;**&#x20;   108**     total_time = time.perf_counter() - t_start &#x20;

RuntimeError: Traceback (most recent call last):   File "/tmp/ipykernel_618/1442377858.py", line 76, in worker     model.generate(     \~\~\~\~\~\~\~\~\~\~\~\~\~\~^         \*\*generation_inputs,         ^^^^^^^^^^^^^^^^^^^^         input_features=input_features_generator(),         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^         streamer=streamer,         ^^^^^^^^^^^^^^^^^^     )     ^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 228, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/models/parakeet/generation_parakeet.py", line 256, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/torch/utils/\_contextlib.py", line 124, in decorate_context     return func(\*args, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 2989, in generate     result = decoding_method(         self,     ...<5 lines>...         \*\*model_kwargs,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 3219, in \_sample     model_kwargs = self.\_update_model_kwargs_for_generation(         outputs,         model_kwargs,         is_encoder_decoder=self.config.is_encoder_decoder,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 53, in \_update_model_kwargs_for_generation     chunk = next(generator)   File "/tmp/ipykernel_618/1442377858.py", line 42, in input_features_generator     chunk_inputs = processor(         audio[start_idx:end_idx],     ...<4 lines>...         return_tensors="pt",     )   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/processing_nemotron_asr_streaming.py", line 140, in \_\_call\_\_     inputs = self.feature_extractor(audio, center=bool(is_first_audio_chunk), \*\*output_kwargs["audio_kwargs"])   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py", line 261, in \_\_call\_\_     input_features = self.\_torch_extract_fbank_features(input_features, device, center=center)   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py", line 106, in \_torch_extract_fbank_features     stft = torch.stft(         waveform,     ...<6 lines>...         center=center,     )   File "/usr/local/lib/python3.13/dist-packages/torch/functional.py", line 681, in stft     return \_VF.stft(  # type: ignore[attr-defined]            \~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^         input,         ^^^^^^     ...<7 lines>...         align_to_window,         ^^^^^^^^^^^^^^^^     )     ^ RuntimeError: stft(torch.FloatTensor[1, 0], n_fft=512, hop_length=160, win_length=400, window=torch.FloatTensor{[400]}, normalized=0, onesided=None, return_complex=1, align_to_window=None) : expected 0 < n_fft < 0, but got n_fft=400 ....................fix code&#x20;
```
def streaming_transcribe(audio, lookahead):
    processor.set_num_lookahead_tokens(int(lookahead))

    sr = processor.feature_extractor.sampling_rate
    audio = np.asarray(audio, dtype=np.float32)

    # Include first-chunk feature extraction in the measured TTFT.
    t_start = time.perf_counter()

    first_n = processor.num_samples_first_audio_chunk

    first_inputs = processor(
        audio[:first_n],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )
    first_inputs = first_inputs.to(
        model.device,
        dtype=model.dtype,
    )

    def input_features_generator():
        # Initial chunk
        yield first_inputs.input_features[
            :, :processor.num_mel_frames_first_audio_chunk, :
        ]

        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft

        start_idx = mel_frame_idx * hop_length - n_fft // 2

        while True:
            end_idx = start_idx + processor.num_samples_per_audio_chunk

            if end_idx >= len(audio):
                break

            chunk_inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                language="en-US",
                return_tensors="pt",
            )
            chunk_inputs = chunk_inputs.to(
                model.device,
                dtype=model.dtype,
            )

            yield chunk_inputs.input_features

            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        timeout=300,
    )

    errors = []

    def worker():
        try:
            with torch.inference_mode():
                generation_inputs = {
                    key: value
                    for key, value in first_inputs.items()
                    if key != "input_features"
                }
                model.generate(
                    **generation_inputs,
                    input_features=input_features_generator(),
                    streamer=streamer,
                )
        except Exception as exc:
            errors.append((exc, traceback.format_exc()))
            streamer.end()

    thread = threading.Thread(target=worker, daemon=True)

    first_text_time = None
    text_parts = []

    thread.start()

    for piece in streamer:
        now = time.perf_counter()

        if piece and piece.strip() and first_text_time is None:
            first_text_time = now - t_start

        text_parts.append(piece)

    thread.join(timeout=300)

    if thread.is_alive():
        raise TimeoutError("Streaming generation exceeded 300 seconds.")

    if errors:
        raise RuntimeError(errors[0][1]) from errors[0][0]

    total_time = time.perf_counter() - t_start
    transcript = "".join(text_parts).strip()
    audio_duration = len(audio) / sr

    return {
        "prediction": transcript,
        "ttft_s": first_text_time,
        "total_time_s": total_time,
        "audio_duration_s": audio_duration,
        "rtf": total_time / audio_duration if audio_duration else np.nan,
        "lookahead": int(lookahead),
        "configured_latency_ms": processor.streaming_latency_ms,
    }

# Smoke test: one recording, lowest supported lookahead.
smoke = streaming_transcribe(test_audio, lookahead=0)

print("Transcript:", smoke["prediction"])
print("Configured streaming latency:", smoke["configured_latency_ms"], "ms")
print("Measured TTFT:", smoke["ttft_s"], "seconds")
print("Total processing time:", smoke["total_time_s"], "seconds")
print("RTF:", smoke["rtf"])
```

## Response:
10/19/2026, 11:28:14 PM · gpt-6-thinking

> **Debugging empty chunks**
>
> I’m tracing the streaming boundaries and noticing a contradiction: the loop should stop before an empty slice, yet the processor receives one. I’m checking chunk-size values, negative offsets, and audio-length comparisons, while also removing the language warning.
>
> **Debugged empty chunks**
>
> 
>
> ****
>
> 
>
> **Aligning implementation details**
>
> I’m comparing the generator with the official pattern and focusing on two likely causes: a version mismatch around language handling and boundary calculations that permit invalid slices. I’ll mirror the documented flow and add an explicit nonempty-chunk guard.
>
> **Aligned implementation details**
>
> 
>
> **Searched 1 website**
>
> 
>
> Worked for 38s

The traceback identifies the immediate problem: the streaming generator is passing an **empty audio chunk** to the feature extractor, causing `torch.stft()` to fail. The `max_length` warning is separate and is not the cause of this crash.

Replace your entire `streaming_transcribe()` function with the version below. It removes the unsupported `language` argument, guards against empty chunks, and checks the audio boundaries before feature extraction.

The official Hugging Face example uses a boundary-checked chunk loop and requires the same lookahead setting to reach `model.generate()`\. \[[Nemotron 3.5 ASR](https://huggingface.co/docs/transformers/v5.13.0/model_doc/nemotron3_5_asr?utm_source=chatgpt.com); [README.md · nvidia/nemotron-3.5-asr-streaming-0.6b at be133a9c2559b9f39bad8c7a789832bdc21a3fde](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/be133a9c2559b9f39bad8c7a789832bdc21a3fde/README.md?utm_source=chatgpt.com)\]

## Replace the function with this version

```python
def streaming_transcribe(audio, lookahead):
    processor.set_num_lookahead_tokens(int(lookahead))

    sr = processor.feature_extractor.sampling_rate
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    t_start = time.perf_counter()

    first_n = min(
        processor.num_samples_first_audio_chunk,
        len(audio),
    )

    first_inputs = processor(
        audio[:first_n],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        return_tensors="pt",
    )

    first_inputs = first_inputs.to(
        model.device,
        dtype=model.dtype,
    )

    def input_features_generator():
        # Yield the initial chunk's streaming features.
        yield first_inputs.input_features[
            :, :processor.num_mel_frames_first_audio_chunk, :
        ]

        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        chunk_samples = processor.num_samples_per_audio_chunk

        if chunk_samples <= 0:
            raise ValueError(
                f"Invalid streaming chunk size: {chunk_samples}"
            )

        start_idx = mel_frame_idx * hop_length - n_fft // 2

        # Process only complete, non-empty subsequent chunks.
        while True:
            end_idx = start_idx + chunk_samples

            if start_idx < 0:
                start_idx = 0
                end_idx = start_idx + chunk_samples

            if start_idx >= len(audio) or end_idx >= len(audio):
                break

            chunk = audio[start_idx:end_idx]

            if chunk.size == 0:
                break

            chunk_inputs = processor(
                chunk,
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )

            chunk_inputs = chunk_inputs.to(
                model.device,
                dtype=model.dtype,
            )

            yield chunk_inputs.input_features

            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        timeout=300,
    )

    errors = []

    def worker():
        try:
            generation_inputs = {
                key: value
                for key, value in first_inputs.items()
                if key != "input_features"
            }

            with torch.inference_mode():
                model.generate(
                    **generation_inputs,
                    input_features=input_features_generator(),
                    num_lookahead_tokens=int(lookahead),
                    streamer=streamer,
                    max_new_tokens=512,
                )

        except Exception:
            errors.append(traceback.format_exc())
            streamer.end()

    thread = threading.Thread(target=worker, daemon=True)

    first_text_time = None
    text_parts = []

    thread.start()

    for piece in streamer:
        if piece and piece.strip() and first_text_time is None:
            first_text_time = time.perf_counter() - t_start

        text_parts.append(piece)

    thread.join(timeout=300)

    if thread.is_alive():
        raise TimeoutError("Streaming generation exceeded 300 seconds.")

    if errors:
        raise RuntimeError(errors[0])

    total_time = time.perf_counter() - t_start
    audio_duration = len(audio) / sr

    return {
        "prediction": "".join(text_parts).strip(),
        "ttft_s": first_text_time,
        "total_time_s": total_time,
        "audio_duration_s": audio_duration,
        "rtf": total_time / audio_duration,
        "lookahead": int(lookahead),
        "configured_latency_ms": processor.streaming_latency_ms,
    }
```

## Run the smoke test again

```python
smoke = streaming_transcribe(test_audio, lookahead=0)

print("Transcript:", smoke["prediction"])
print("Configured latency (ms):", smoke["configured_latency_ms"])
print("Measured TTFT (seconds):", smoke["ttft_s"])
print("Total processing time (seconds):", smoke["total_time_s"])
print("Audio duration (seconds):", smoke["audio_duration_s"])
print("RTF:", smoke["rtf"])
```

**Important\:** this is a defensive correction, not yet a verified fix on your runtime. If it fails again, run this diagnostic cell and share the output before attempting the 50-clip benchmark\:

```python
print("Transformers version:", __import__("transformers").__version__)
print("Audio samples:", len(test_audio))
print("First chunk samples:", processor.num_samples_first_audio_chunk)
print("Subsequent chunk samples:", processor.num_samples_per_audio_chunk)
print("First chunk mel frames:", processor.num_mel_frames_first_audio_chunk)
print("Subsequent chunk mel frames:", processor.num_mel_frames_per_audio_chunk)
print("Lookahead settings:", processor.supported_num_lookahead_tokens)
```

This will help establish whether the remaining issue comes from chunk boundaries or a mismatch between your installed Transformers implementation and the model's streaming API.

## Prompt:
10/19/2026, 11:29:21 PM

print("Transformers version:", __import__("transformers").__version__)
print("Audio samples:", len(test_audio))
print("First chunk samples:", processor.num_samples_first_audio_chunk)
print("Subsequent chunk samples:", processor.num_samples_per_audio_chunk)
print("First chunk mel frames:", processor.num_mel_frames_first_audio_chunk)
print("Subsequent chunk mel frames:", processor.num_mel_frames_per_audio_chunk)
print("Lookahead settings:", processor.supported_num_lookahead_tokens)

## Response:
10/19/2026, 11:29:27 PM · gpt-6-thinking

Run this diagnostic cell in Colab and send me the **printed output**, not the code itself. It will help identify why the streaming audio generator is producing an empty chunk.

```python
print("Transformers version:", __import__("transformers").__version__)
print("Audio samples:", len(test_audio))
print("First chunk samples:", processor.num_samples_first_audio_chunk)
print("Subsequent chunk samples:", processor.num_samples_per_audio_chunk)
print("First chunk mel frames:", processor.num_mel_frames_first_audio_chunk)
print("Subsequent chunk mel frames:", processor.num_mel_frames_per_audio_chunk)
print("Lookahead settings:", processor.supported_num_lookahead_tokens)
```

In Colab, click the **▶ Run** button beside the cell. Copy everything printed below it and paste it here.

I'll use those values to correct the streaming implementation before you run the full 50-clip benchmark.

## Prompt:
10/19/2026, 11:29:56 PM

Transformers version: 5.20.1
Audio samples: 75680
First chunk samples: 200
Subsequent chunk samples: 1680
First chunk mel frames: 1
Subsequent chunk mel frames: 8
Lookahead settings: [13, 6, 1, 0]

## Response:
10/19/2026, 11:30:05 PM · gpt-6-thinking

> **Debugging audio slicing**
>
> I’m tracing why the slice is empty. Negative indexing is normalizing the start near the array’s end, so I’m clipping the start to zero while preserving the intended chunk endpoint and overlap.
>
> **Checking model compatibility**
>
> I’m noticing a likely mismatch: the traceback references an older streaming processor, while the intended checkpoint uses a newer processor class. I’d reinitialize both from the exact checkpoint, verify their class names, then apply the clipped-slice fix.
>
> **Debugged audio slicing and checked model compatibility**
>
> 
>
> **Searched 5 websites**
>
> 
>
> Worked for 43s

Your diagnostic output reveals the underlying issue: **the streaming chunk sizes are suspiciously small for the audio sample rate**, and your earlier traceback refers to `nemotron_asr_streaming` rather than the `nemotron3_5_asr` implementation documented for the checkpoint you intend to use. We should verify that the notebook has loaded the correct model and processor before trusting any benchmark results. The official Hugging Face example uses a specific chunk-generation procedure. \[[transformers/docs/source/en/model\_doc/nemotron3\_5\_asr.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nemotron3_5_asr.md?utm_source=chatgpt.com); [Nemotron 3.5 ASR](https://huggingface.co/docs/transformers/v5.13.0/model_doc/nemotron3_5_asr?utm_source=chatgpt.com)\]

## Step 1 - Verify and reload the intended checkpoint

Restart the Colab runtime first (`Runtime → Restart session`\), then run this cell. It deliberately loads the English-capable Nemotron 3.5 0.6B checkpoint from scratch.

```python
!pip install -q -U "transformers>=5.20.1" accelerate

import torch
import transformers
from transformers import AutoProcessor, AutoModelForRNNT

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForRNNT.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
).to("cpu").eval()

torch.set_num_threads(1)

print("Transformers:", transformers.__version__)
print("Model class:", type(model).__module__)
print("Processor class:", type(processor).__module__)
print("Model:", MODEL_ID)
print("Sample rate:", processor.feature_extractor.sampling_rate)
print("First chunk samples:", processor.num_samples_first_audio_chunk)
print("Next chunk samples:", processor.num_samples_per_audio_chunk)
print("First chunk mel frames:", processor.num_mel_frames_first_audio_chunk)
print("Next chunk mel frames:", processor.num_mel_frames_per_audio_chunk)
print("Supported lookahead:", processor.supported_num_lookahead_tokens)
```

## Step 2 - Check the output

Before running the benchmark again, inspect the printed **model class, processor class, and chunk sizes**\.

- If the module names contain `nemotron3_5_asr`, that is consistent with the intended checkpoint's documented implementation.
- If they still contain `nemotron_asr_streaming`, stop and share the output.
- Do not run the 50-clip benchmark until the checkpoint and streaming API are confirmed.

The negative audio-slice boundary in the earlier implementation also needs correcting; simply changing the chunk sizes or suppressing the warning would not establish a valid streaming benchmark.

## Prompt:
10/19/2026, 11:32:46 PM

Loading weights: 100% 655/655 [00:01<00:00, 366.58it/s]Transformers: 5.20.1
Model class: transformers.models.nemotron3_5_asr.modeling_nemotron3_5_asr
Processor class: transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr
Model: nvidia/nemotron-3.5-asr-streaming-0.6b
Sample rate: 16000
First chunk samples: 4040
Next chunk samples: 5520
First chunk mel frames: 25
Next chunk mel frames: 32
Supported lookahead: [3, 0, 6, 13]

## Response:
10/19/2026, 11:32:52 PM · gpt-6-thinking

> **Searched 3 websites**
>
> 
>
> Worked for a couple of seconds

Good. The model and processor now match the intended Nemotron 3.5 implementation, and the chunk sizes are consistent with its documented streaming configuration. \[[Nemotron 3.5 ASR](https://huggingface.co/docs/transformers/v5.13.0/model_doc/nemotron3_5_asr?utm_source=chatgpt.com); [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b?library=transformers&utm_source=chatgpt.com)\]

**Next, replace your `streaming_transcribe()` function with the corrected version below.** The key differences from the earlier code are that it uses the official chunk boundaries, passes `num_lookahead_tokens` through the processor inputs, and does not manually pass a second `input_features` argument alongside the first-chunk inputs.

## Cell 1 - Correct streaming function

```python
import time
import threading
import traceback
import numpy as np
import torch

from transformers import TextIteratorStreamer

def streaming_transcribe(audio, lookahead):
    processor.set_num_lookahead_tokens(int(lookahead))

    sr = processor.feature_extractor.sampling_rate
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    t_start = time.perf_counter()

    # Prepare the first chunk using the documented API.
    first_chunk_inputs = processor(
        audio[:processor.num_samples_first_audio_chunk],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        language="en-US",
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    def input_features_generator():
        yield first_chunk_inputs.input_features[
            :, :processor.num_mel_frames_first_audio_chunk, :
        ]

        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft

        start_idx = mel_frame_idx * hop_length - n_fft // 2

        while True:
            end_idx = start_idx + processor.num_samples_per_audio_chunk

            # Match the official example's boundary condition.
            if end_idx >= len(audio):
                break

            if start_idx < 0 or end_idx <= start_idx:
                raise ValueError(
                    f"Invalid audio chunk boundaries: "
                    f"{start_idx}:{end_idx}"
                )

            chunk_inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                language="en-US",
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)

            yield chunk_inputs.input_features

            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        timeout=300,
    )

    errors = []

    def worker():
        try:
            with torch.inference_mode():
                # The first-chunk inputs already carry the lookahead value.
                generate_kwargs = {
                    **first_chunk_inputs,
                    "input_features": input_features_generator(),
                    "streamer": streamer,
                    "max_new_tokens": 512,
                }
                model.generate(**generate_kwargs)

        except Exception:
            errors.append(traceback.format_exc())
            streamer.end()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    text_parts = []
    first_text_time = None

    for piece in streamer:
        if piece and piece.strip() and first_text_time is None:
            first_text_time = time.perf_counter() - t_start
        text_parts.append(piece)

    thread.join(timeout=300)

    if thread.is_alive():
        raise TimeoutError("Generation exceeded 300 seconds.")

    if errors:
        raise RuntimeError(errors[0])

    total_time = time.perf_counter() - t_start
    audio_duration = len(audio) / sr

    return {
        "prediction": "".join(text_parts).strip(),
        "ttft_s": first_text_time,
        "total_time_s": total_time,
        "audio_duration_s": audio_duration,
        "rtf": total_time / audio_duration,
        "lookahead": int(lookahead),
        "configured_latency_ms": processor.streaming_latency_ms,
    }
```

## Cell 2 - Run one smoke test

```python
smoke = streaming_transcribe(test_audio, lookahead=0)

for key, value in smoke.items():
    print(f"{key}: {value}")
```

If this succeeds, test the remaining configurations\:

```python
smoke_results = []

for lookahead in [0, 3, 6, 13]:
    print(f"\nTesting lookahead={lookahead}")
    result = streaming_transcribe(test_audio, lookahead)
    smoke_results.append(result)

display(pd.DataFrame(smoke_results))
```

Do not launch the 50-clip benchmark until this passes for all four settings. The measured TTFT is an observed software measurement, while `configured_latency_ms` is the model's nominal streaming latency; keep those metrics separate in your report.

## Prompt:
10/19/2026, 11:39:10 PM

what bext

## Response:
10/19/2026, 11:39:15 PM · gpt-6-thinking

Next, we need to **verify the streaming function works, then run the full benchmark**\. Follow these steps in order.

## Step 1 - Check the smoke test

If you have not run it since replacing the function, execute\:

```python
smoke = streaming_transcribe(test_audio, lookahead=0)
print(smoke)
```

If it returns a transcript and timing metrics without an exception, proceed.

## Step 2 - Test all four lookahead settings

```python
smoke_results = []

for lookahead in [0, 3, 6, 13]:
    print(f"Testing lookahead={lookahead}")

    result = streaming_transcribe(test_audio, lookahead)
    smoke_results.append(result)

display(pd.DataFrame(smoke_results))
```

All four settings must complete successfully before continuing.

## Step 3 - Run the 50-clip benchmark

Once the smoke tests pass, run this cell\:

```python
from jiwer import wer, cer
import pandas as pd
import numpy as np

results = []

for lookahead in [0, 3, 6, 13]:
    print(f"\nRunning lookahead={lookahead}")

    for _, row in manifest.iterrows():
        idx = int(row["row"])
        audio, sr = get_audio_from_dataset(idx)

        try:
            out = streaming_transcribe(audio, lookahead)

            reference = str(row["transcription"])
            prediction = out["prediction"]

            results.append({
                "row": idx,
                "speech_status": row["speech_status"],
                "lookahead": lookahead,
                "configured_latency_ms": out["configured_latency_ms"],
                "ttft_ms": (
                    out["ttft_s"] * 1000
                    if out["ttft_s"] is not None else np.nan
                ),
                "audio_duration_s": out["audio_duration_s"],
                "total_time_s": out["total_time_s"],
                "rtf": out["rtf"],
                "wer": wer(reference.lower(), prediction.lower()),
                "cer": cer(reference.lower(), prediction.lower()),
                "reference": reference,
                "prediction": prediction,
                "error": None,
            })

        except Exception as e:
            print(f"Failed row {idx}: {e}")

    pd.DataFrame(results).to_csv(
        "/content/nemotron_cpu_results.csv",
        index=False,
    )

print("\nBenchmark finished.")
```

This produces up to 200 model runs: 50 recordings × 4 lookahead settings.

## Step 4 - Summarize the results

```python
results_df = pd.read_csv("/content/nemotron_cpu_results.csv")

summary = (
    results_df
    .groupby(["lookahead", "speech_status"])
    .agg(
        clips=("row", "nunique"),
        median_TTFT_ms=("ttft_ms", "median"),
        mean_RTF=("rtf", "mean"),
        mean_WER=("wer", "mean"),
        mean_CER=("cer", "mean"),
    )
    .reset_index()
)

display(summary.round(4))
summary.to_csv("/content/nemotron_cpu_summary.csv", index=False)
```

**Do not interpret the benchmark as a model comparison yet.** This stage establishes Nemotron's performance under different streaming configurations. Once we have valid results, we can run FastConformer and Zipformer on the same recordings and hardware to identify which model offers the best latency-accuracy trade-off.

If the smoke test fails, send me the complete traceback before running the full benchmark.

## Prompt:
10/19/2026, 11:40:56 PM

\---------------------------------------------------------------------------&#x20;
RuntimeError                              Traceback (most recent call last)&#x20;
[/tmp/ipykernel_618/2519907435.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \<cell line: 0>() ----> 1 smoke = streaming_transcribe(test_audio, lookahead=0)&#x20;**&#x20;     2** print(smoke) &#x20;
[/tmp/ipykernel_618/423208144.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in streaming_transcribe(audio, lookahead)&#x20;**&#x20;   107** &#x20;**&#x20;   108**     if errors: --> 109         raise RuntimeError(errors[0])&#x20;**&#x20;   110** &#x20;**&#x20;   111**     total_time = time.perf_counter() - t_start &#x20;
RuntimeError: Traceback (most recent call last):   File "/tmp/ipykernel_618/423208144.py", line 86, in worker     model.generate(\*\*generate_kwargs)     \~\~\~\~\~\~\~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron3_5_asr/generation_nemotron3_5_asr.py", line 35, in generate     return super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)            \~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 228, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/models/parakeet/generation_parakeet.py", line 256, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/torch/utils/\_contextlib.py", line 124, in decorate_context     return func(\*args, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 2989, in generate     result = decoding_method(         self,     ...<5 lines>...         \*\*model_kwargs,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 3219, in \_sample     model_kwargs = self.\_update_model_kwargs_for_generation(         outputs,         model_kwargs,         is_encoder_decoder=self.config.is_encoder_decoder,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 53, in \_update_model_kwargs_for_generation     chunk = next(generator)   File "/tmp/ipykernel_618/423208144.py", line 49, in input_features_generator     raise ValueError(     ...<2 lines>...     ) ValueError: Invalid audio chunk boundaries: -96:1584

## Response:
10/19/2026, 11:41:09 PM · gpt-6-thinking

> ****
>
> 
>
> **Diagnosing stale state**
>
> The error shows older chunk sizes are still being used, even though the diagnostics report current values. I’m inferring that the generator or worker is holding stale definitions, so rerunning the setup and function from a clean state should resolve it.
>
> **Searched 1 website**
>
> 
>
> Worked for 30s

The error `-96:1584` proves that the running function is still using the **old chunk configuration** \(1 mel frame and 1,680 samples), rather than your newly loaded model's configuration (25 mel frames and 5,520 samples). The official model example uses the processor's chunk settings dynamically. \[[README.md](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blame/main/README.md?utm_source=chatgpt.com); [Nemotron 3.5 ASR](https://huggingface.co/docs/transformers/v5.13.0/model_doc/nemotron3_5_asr?utm_source=chatgpt.com)\]

Let's replace the function with a clean version that follows the official implementation, without the custom boundary checks that caused the exception.

## 1\. Run this cell to replace the function

```python
import time
import threading
import traceback
import numpy as np
import torch
from transformers import TextIteratorStreamer

def streaming_transcribe(audio, lookahead):
    # Capture the currently loaded processor and model.
    proc = processor
    net = model

    proc.set_num_lookahead_tokens(int(lookahead))

    sr = proc.feature_extractor.sampling_rate
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    # Confirm the configuration actually used by this call.
    first_samples = proc.num_samples_first_audio_chunk
    next_samples = proc.num_samples_per_audio_chunk
    first_frames = proc.num_mel_frames_first_audio_chunk

    print(
        f"Using chunks: first={first_samples}, "
        f"next={next_samples}, first_frames={first_frames}"
    )

    t_start = time.perf_counter()

    first_inputs = proc(
        audio[:first_samples],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        language="en-US",
        return_tensors="pt",
    ).to(net.device, dtype=net.dtype)

    def input_features_generator():
        yield first_inputs.input_features[:, :first_frames, :]

        mel_frame_idx = first_frames
        hop_length = proc.feature_extractor.hop_length
        n_fft = proc.feature_extractor.n_fft

        start_idx = mel_frame_idx * hop_length - n_fft // 2

        while True:
            end_idx = start_idx + next_samples

            if end_idx >= len(audio):
                break

            # Use the official chunk boundaries without modifying them.
            chunk_inputs = proc(
                audio[start_idx:end_idx],
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                language="en-US",
                return_tensors="pt",
            ).to(net.device, dtype=net.dtype)

            yield chunk_inputs.input_features

            mel_frame_idx += proc.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2

    streamer = TextIteratorStreamer(
        proc.tokenizer,
        skip_special_tokens=True,
        timeout=300,
    )

    errors = []

    # Use a dictionary so input_features is replaced rather than duplicated.
    generate_kwargs = {
        **first_inputs,
        "input_features": input_features_generator(),
        "streamer": streamer,
        "max_new_tokens": 512,
    }

    def worker():
        try:
            with torch.inference_mode():
                net.generate(**generate_kwargs)
        except Exception:
            errors.append(traceback.format_exc())
            streamer.end()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    text_parts = []
    first_text_time = None

    for piece in streamer:
        if piece and piece.strip() and first_text_time is None:
            first_text_time = time.perf_counter() - t_start
        text_parts.append(piece)

    thread.join(timeout=300)

    if thread.is_alive():
        raise TimeoutError("Streaming generation exceeded 300 seconds.")

    if errors:
        raise RuntimeError(errors[0])

    total_time = time.perf_counter() - t_start
    duration = len(audio) / sr

    return {
        "prediction": "".join(text_parts).strip(),
        "ttft_s": first_text_time,
        "total_time_s": total_time,
        "audio_duration_s": duration,
        "rtf": total_time / duration,
        "lookahead": int(lookahead),
        "configured_latency_ms": proc.streaming_latency_ms,
    }
```

## 2\. Run the smoke test

```python
smoke = streaming_transcribe(test_audio, lookahead=0)
print(smoke)
```

Check the first line of output. It should say\:

```text
Using chunks: first=4040, next=5520, first_frames=25
```

If it prints those values and returns a transcript, the stale-configuration problem is resolved. If it prints the old values again, restart the runtime and rerun the model-loading cell followed by this function cell.

Do not start the full benchmark until the smoke test succeeds.

## Prompt:
10/19/2026, 11:42:41 PM

Using chunks: first=200, next=1680, first_frames=1&#x20;
\---------------------------------------------------------------------------&#x20;
RuntimeError                              Traceback (most recent call last)&#x20;
[/tmp/ipykernel_618/2519907435.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \<cell line: 0>() ----> 1 smoke = streaming_transcribe(test_audio, lookahead=0)&#x20;**&#x20;     2** print(smoke) &#x20;
[/tmp/ipykernel_618/1771398588.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in streaming_transcribe(audio, lookahead)&#x20;**&#x20;   111** &#x20;**&#x20;   112**     if errors: --> 113         raise RuntimeError(errors[0])&#x20;**&#x20;   114** &#x20;**&#x20;   115**     total_time = time.perf_counter() - t_start &#x20;
RuntimeError: Traceback (most recent call last):   File "/tmp/ipykernel_618/1771398588.py", line 91, in worker     net.generate(\*\*generate_kwargs)     \~\~\~\~\~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron3_5_asr/generation_nemotron3_5_asr.py", line 35, in generate     return super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)            \~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 228, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/models/parakeet/generation_parakeet.py", line 256, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/torch/utils/\_contextlib.py", line 124, in decorate_context     return func(\*args, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 2989, in generate     result = decoding_method(         self,     ...<5 lines>...         \*\*model_kwargs,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 3219, in \_sample     model_kwargs = self.\_update_model_kwargs_for_generation(         outputs,         model_kwargs,         is_encoder_decoder=self.config.is_encoder_decoder,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 53, in \_update_model_kwargs_for_generation     chunk = next(generator)   File "/tmp/ipykernel_618/1771398588.py", line 58, in input_features_generator     chunk_inputs = proc(                    \~\~\~\~^         audio[start_idx:end_idx],         ^^^^^^^^^^^^^^^^^^^^^^^^^     ...<4 lines>...         return_tensors="pt",         ^^^^^^^^^^^^^^^^^^^^     ).to(net.device, dtype=net.dtype)     ^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron3_5_asr/processing_nemotron3_5_asr.py", line 295, in \_\_call\_\_     inputs = self.feature_extractor(audio, center=bool(is_first_audio_chunk), \*\*output_kwargs["audio_kwargs"])   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py", line 261, in \_\_call\_\_     input_features = self.\_torch_extract_fbank_features(input_features, device, center=center)   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py", line 106, in \_torch_extract_fbank_features     stft = torch.stft(         waveform,     ...<6 lines>...         center=center,     )   File "/usr/local/lib/python3.13/dist-packages/torch/functional.py", line 681, in stft     return \_VF.stft(  # type: ignore[attr-defined]            \~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^         input,         ^^^^^^     ...<7 lines>...         align_to_window,         ^^^^^^^^^^^^^^^^     )     ^ RuntimeError: stft(torch.FloatTensor[1, 0], n_fft=512, hop_length=160, win_length=400, window=torch.FloatTensor{[400]}, normalized=0, onesided=None, return_complex=1, align_to_window=None) : expected 0 < n_fft < 0, but got n_fft=400

## Response:
10/19/2026, 11:42:52 PM · gpt-6-thinking

> ****
>
> 
>
> Worked for 7s

The error is now clear: your current `processor` is reverting to the old chunk configuration (`200` samples and `1` mel frame), even though the correctly loaded Nemotron 3.5 model previously reported `4040` samples and `25` mel frames. The empty-chunk error is a consequence of that mismatch.

**Do not run the benchmark yet.** First, run this single cell to reload the processor and check its configuration.

## Cell 1 - Reload and verify the processor

```python
from transformers import AutoProcessor, AutoModelForRNNT
import transformers
import torch

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# Reload the processor explicitly from the same checkpoint.
processor = AutoProcessor.from_pretrained(
    MODEL_ID
)

# Confirm the processor class and chunk configuration.
print("Transformers:", transformers.__version__)
print("Processor class:", type(processor).__module__)
print("Model class:", type(model).__module__)
print("Sample rate:", processor.feature_extractor.sampling_rate)
print("First chunk samples:", processor.num_samples_first_audio_chunk)
print("Next chunk samples:", processor.num_samples_per_audio_chunk)
print("First chunk mel frames:", processor.num_mel_frames_first_audio_chunk)
print("Next chunk mel frames:", processor.num_mel_frames_per_audio_chunk)
print("Supported lookahead:", processor.supported_num_lookahead_tokens)
```

## Cell 2 - Check the result

The expected values from your earlier successful model-loading cell are\:

```text
Processor class: transformers.models.nemotron3_5_asr.processing_nemotron3_5_asr
First chunk samples: 4040
Next chunk samples: 5520
First chunk mel frames: 25
Next chunk mel frames: 32
```

If these values appear, rerun the smoke test using your existing function\:

```python
smoke = streaming_transcribe(test_audio, lookahead=0)
print(smoke)
```

If the processor still reports `200` and `1680`, stop and share the output. That would indicate a checkpoint or processor-configuration mismatch that needs to be resolved before modifying the streaming algorithm further.

## Prompt:
10/19/2026, 11:44:32 PM

Using chunks: first=200, next=1680, first_frames=1&#x20;
\---------------------------------------------------------------------------&#x20;
RuntimeError                              Traceback (most recent call last)&#x20;
[/tmp/ipykernel_618/2519907435.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \<cell line: 0>() ----> 1 smoke = streaming_transcribe(test_audio, lookahead=0)&#x20;**&#x20;     2** print(smoke) &#x20;
[/tmp/ipykernel_618/1771398588.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in streaming_transcribe(audio, lookahead)&#x20;**&#x20;   111** &#x20;**&#x20;   112**     if errors: --> 113         raise RuntimeError(errors[0])&#x20;**&#x20;   114** &#x20;**&#x20;   115**     total_time = time.perf_counter() - t_start &#x20;
RuntimeError: Traceback (most recent call last):   File "/tmp/ipykernel_618/1771398588.py", line 91, in worker     net.generate(\*\*generate_kwargs)     \~\~\~\~\~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron3_5_asr/generation_nemotron3_5_asr.py", line 35, in generate     return super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)            \~\~\~\~\~\~\~\~\~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 228, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/models/parakeet/generation_parakeet.py", line 256, in generate     outputs = super().generate(inputs=inputs, generation_config=generation_config, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/torch/utils/\_contextlib.py", line 124, in decorate_context     return func(\*args, \*\*kwargs)   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 2989, in generate     result = decoding_method(         self,     ...<5 lines>...         \*\*model_kwargs,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/generation/utils.py", line 3219, in \_sample     model_kwargs = self.\_update_model_kwargs_for_generation(         outputs,         model_kwargs,         is_encoder_decoder=self.config.is_encoder_decoder,     )   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py", line 53, in \_update_model_kwargs_for_generation     chunk = next(generator)   File "/tmp/ipykernel_618/1771398588.py", line 58, in input_features_generator     chunk_inputs = proc(                    \~\~\~\~^         audio[start_idx:end_idx],         ^^^^^^^^^^^^^^^^^^^^^^^^^     ...<4 lines>...         return_tensors="pt",         ^^^^^^^^^^^^^^^^^^^^     ).to(net.device, dtype=net.dtype)     ^   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron3_5_asr/processing_nemotron3_5_asr.py", line 295, in \_\_call\_\_     inputs = self.feature_extractor(audio, center=bool(is_first_audio_chunk), \*\*output_kwargs["audio_kwargs"])   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py", line 261, in \_\_call\_\_     input_features = self.\_torch_extract_fbank_features(input_features, device, center=center)   File "/usr/local/lib/python3.13/dist-packages/transformers/models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py", line 106, in \_torch_extract_fbank_features     stft = torch.stft(         waveform,     ...<6 lines>...         center=center,     )   File "/usr/local/lib/python3.13/dist-packages/torch/functional.py", line 681, in stft     return \_VF.stft(  # type: ignore[attr-defined]            \~\~\~\~\~\~\~\~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^         input,         ^^^^^^     ...<7 lines>...         align_to_window,         ^^^^^^^^^^^^^^^^     )     ^ RuntimeError: stft(torch.FloatTensor[1, 0], n_fft=512, hop_length=160, win_length=400, window=torch.FloatTensor{[400]}, normalized=0, onesided=None, return_complex=1, align_to_window=None) : expected 0 < n_fft < 0, but got n_fft=400

## Response:
10/19/2026, 11:44:46 PM · gpt-6-thinking

> **Diagnosing processor state**
>
> I’m comparing the processor settings before and after applying lookahead values, because the configuration appears to revert unexpectedly. I’ll verify whether that method is mutating chunk parameters, then reset both model and processor cleanly.
>
> **Diagnosed processor state**
>
> 
>
> ****
>
> 
>
> **Correcting chunk expectations**
>
> I’ve identified the mismatch: setting lookahead to zero legitimately produces a 200-sample first chunk, not the larger default. I’m checking whether zero-lookahead requires special boundary handling, while larger settings should follow the documented slicing logic.
>
> **Corrected chunk expectations**
>
> 
>
> **Searched 2 websites**
>
> 
>
> Worked for 39s

The traceback confirms that the failure is still occurring inside the streaming feature generator. The function is printing `first=200, next=1680, first_frames=1`, so it is **not using the expected chunk configuration of 4040, 5520, and 25**\.

Let's stop rerunning the smoke test and isolate why those values change. Run the diagnostic cell below first. It checks the processor both before and after setting the lookahead value.

I found the actual cause. **The `200`\-sample chunk is expected when you select lookahead `0`; it is not evidence that the old processor was reloaded.** The processor recalculates chunk sizes whenever you change lookahead. The problem is that your generator starts at sample `-96`, which produces an empty Python audio slice.

The official model documentation confirms that lookahead `3` uses a 4,040-sample first chunk and a 5,520-sample subsequent chunk. Let's first get that supported configuration working, then we can handle zero-lookahead separately. \[[transformers/docs/source/en/model\_doc/nemotron3\_5\_asr.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/nemotron3_5_asr.md?utm_source=chatgpt.com); [README.md](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blame/main/README.md?utm_source=chatgpt.com)\]

## Replace your function with this version

This version restricts the initial smoke test to lookahead values `3`, `6`, and `13`, where the chunk boundaries remain positive.

```python
import time
import threading
import traceback
import numpy as np
import torch
from transformers import TextIteratorStreamer

def streaming_transcribe(audio, lookahead):
    if lookahead not in [3, 6, 13]:
        raise ValueError(
            "For this initial version, use lookahead 3, 6, or 13."
        )

    processor.set_num_lookahead_tokens(lookahead)

    sr = processor.feature_extractor.sampling_rate
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    t0 = time.perf_counter()

    first_n = processor.num_samples_first_audio_chunk
    chunk_n = processor.num_samples_per_audio_chunk
    first_frames = processor.num_mel_frames_first_audio_chunk

    print(
        f"Lookahead={lookahead}; "
        f"first={first_n}; next={chunk_n}; "
        f"first_frames={first_frames}"
    )

    first_inputs = processor(
        audio[:first_n],
        sampling_rate=sr,
        is_streaming=True,
        is_first_audio_chunk=True,
        language="en-US",
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)

    def feature_generator():
        yield first_inputs.input_features[:, :first_frames, :]

        mel_idx = first_frames
        hop = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft

        start = mel_idx * hop - n_fft // 2

        while start >= 0:
            end = start + chunk_n

            if end >= len(audio):
                break

            chunk = audio[start:end]
            if len(chunk) == 0:
                break

            chunk_inputs = processor(
                chunk,
                sampling_rate=sr,
                is_streaming=True,
                is_first_audio_chunk=False,
                language="en-US",
                return_tensors="pt",
            ).to(model.device, dtype=model.dtype)

            yield chunk_inputs.input_features

            mel_idx += processor.num_mel_frames_per_audio_chunk
            start = mel_idx * hop - n_fft // 2

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        timeout=300,
    )

    errors = []

    kwargs = {
        **first_inputs,
        "input_features": feature_generator(),
        "streamer": streamer,
        "max_new_tokens": 512,
    }

    def worker():
        try:
            with torch.inference_mode():
                model.generate(**kwargs)
        except Exception:
            errors.append(traceback.format_exc())
            streamer.end()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    pieces = []
    ttft = None

    for piece in streamer:
        if piece and piece.strip() and ttft is None:
            ttft = time.perf_counter() - t0
        pieces.append(piece)

    thread.join(timeout=300)

    if thread.is_alive():
        raise TimeoutError("Generation exceeded 300 seconds.")

    if errors:
        raise RuntimeError(errors[0])

    elapsed = time.perf_counter() - t0
    duration = len(audio) / sr

    return {
        "prediction": "".join(pieces).strip(),
        "ttft_s": ttft,
        "total_time_s": elapsed,
        "audio_duration_s": duration,
        "rtf": elapsed / duration,
        "lookahead": lookahead,
        "configured_latency_ms": processor.streaming_latency_ms,
    }
```

## Run this smoke test

```python
smoke = streaming_transcribe(test_audio, lookahead=3)
print(smoke)
```

If it succeeds, test all three configurations\:

```python
smoke_results = []

for lookahead in [3, 6, 13]:
    result = streaming_transcribe(test_audio, lookahead)
    smoke_results.append(result)

display(pd.DataFrame(smoke_results))
```

Expected configured latencies are 320 ms, 560 ms, and 1,120 ms, respectively. These are model-configured streaming latencies, not measured TTFT.

Once these tests pass, proceed to the 50-clip benchmark using `[3, 6, 13]`\. We should not include lookahead `0` until its chunk-boundary handling has been corrected and tested independently.

## Prompt:
10/19/2026, 11:51:15 PM

done

## Response:
10/19/2026, 11:51:20 PM · gpt-6-thinking

Good. Now we move to the **50-clip benchmark** for Nemotron 3.5 ASR Streaming 0.6B on CPU.

## Step 1 - Run the benchmark

This tests lookahead values `3`, `6`, and `13` on the same 50 recordings. It saves results after each recording, so partial results are retained if the runtime stops.

```python
import pandas as pd
import numpy as np
from jiwer import wer, cer
from pathlib import Path

LOOKAHEADS = [3, 6, 13]
RESULT_PATH = "/content/nemotron_cpu_results.csv"

# Resume safely if a previous run already produced results.
if Path(RESULT_PATH).exists():
    results = pd.read_csv(RESULT_PATH).to_dict("records")
else:
    results = []

completed = {
    (int(r["row"]), int(r["lookahead"]))
    for r in results
    if not r.get("error")
}

for lookahead in LOOKAHEADS:
    print(f"\n=== Lookahead {lookahead} ===")

    for _, row in manifest.iterrows():
        idx = int(row["row"])

        if (idx, lookahead) in completed:
            continue

        try:
            audio, sr = get_audio_from_dataset(idx)

            out = streaming_transcribe(audio, lookahead)

            reference = str(row["transcription"])
            prediction = out["prediction"]

            result = {
                "row": idx,
                "speech_status": row["speech_status"],
                "lookahead": lookahead,
                "configured_latency_ms": out["configured_latency_ms"],
                "ttft_ms": (
                    out["ttft_s"] * 1000
                    if out["ttft_s"] is not None else np.nan
                ),
                "audio_duration_s": out["audio_duration_s"],
                "total_time_s": out["total_time_s"],
                "rtf": out["rtf"],
                "wer": wer(reference.lower(), prediction.lower()),
                "cer": cer(reference.lower(), prediction.lower()),
                "reference": reference,
                "prediction": prediction,
                "error": "",
            }

        except Exception as exc:
            result = {
                "row": idx,
                "speech_status": row["speech_status"],
                "lookahead": lookahead,
                "error": repr(exc),
            }
            print(f"Failed row {idx}: {exc}")

        results.append(result)

        # Save after every recording.
        pd.DataFrame(results).to_csv(RESULT_PATH, index=False)

    print(f"Saved results: {RESULT_PATH}")

print("\nBenchmark loop finished.")
print("Total result rows:", len(results))
```

## Step 2 - Generate the summary

Run this after the benchmark finishes.

```python
results_df = pd.read_csv("/content/nemotron_cpu_results.csv")

valid = results_df[
    results_df["error"].fillna("").eq("")
].copy()

summary = (
    valid.groupby(["lookahead", "speech_status"])
    .agg(
        clips=("row", "nunique"),
        median_TTFT_ms=("ttft_ms", "median"),
        p95_TTFT_ms=("ttft_ms", lambda x: x.quantile(0.95)),
        mean_RTF=("rtf", "mean"),
        mean_WER=("wer", "mean"),
        mean_CER=("cer", "mean"),
    )
    .reset_index()
)

display(summary.round(4))

summary.to_csv(
    "/content/nemotron_cpu_summary.csv",
    index=False,
)

print("Summary saved.")
```

## Step 3 - What we do with the results

Once you have the summary, we will\:

1. Identify the fastest supported configuration.
2. Check whether lower latency comes with worse WER or CER.
3. Compare healthy and dysarthric speech separately.
4. Run FastConformer and Zipformer on the same clips and hardware.
5. Prepare reproducible results and plots for your Fatima Institute submission.

**Important\:** these first measurements establish a baseline, not the final research finding. Keep the CPU hardware and thread count fixed across all models, and do not report failed recordings as successful runs.

## Prompt:
10/19/2026, 11:53:07 PM

Yes.. Maybe try moonshine tiny straming too

## Response:
10/19/2026, 11:53:14 PM · gpt-6-thinking

> **Clarifying model setup**
>
> I’m checking the correct architecture value and download workflow, then I’ll show how to locate the downloaded model and initialize the transcriber without relying on uncertain API details.
>
> **Designing streaming benchmarks**
>
> I’m outlining callback-based transcription with chunked audio, measuring time to first text and completion. I’ll start with a smoke test, then decide whether real-time pacing or rapid feeding better reflects the intended benchmark.
>
> **Clarified model setup and designed streaming benchmarks**
>
> 
>
> **Searched 2 websites**
>
> 
>
> Worked for 54s

Yes. **Moonshine Tiny Streaming is a useful additional baseline** because it is specifically designed for low-latency, on-device speech recognition. Its implementation uses a quantized ONNX Runtime model, so it provides a useful comparison against Nemotron running through PyTorch on CPU. However, the runtime difference must be documented: this is a comparison of deployable systems, not just model architectures. \[[moonshine/docs/models/accuracy.md at main · moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/accuracy.md?utm_source=chatgpt.com); [Transcription](https://moonshine-voice.readthedocs.io/en/latest/using/transcription/?utm_source=chatgpt.com); [Classes](https://moonshine-voice.readthedocs.io/en/latest/api/classes/?utm_source=chatgpt.com)\]

We'll run Moonshine on the same 50 TORGO recordings and calculate WER, CER, real-time factor, and streaming latency.

## Cell 1 - Install Moonshine Voice

```python
!pip -q install -U moonshine-voice jiwer soundfile
```

## Cell 2 - Download the Tiny Streaming model

The official downloader supports selecting a model architecture. Tiny Streaming is architecture `2`\. \[[Downloading Models](https://moonshine-voice.readthedocs.io/en/latest/using/downloading-models/?utm_source=chatgpt.com); [moonshine/python/src/moonshine\_voice/moonshine\_api.py at main · moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine/blob/main/python/src/moonshine_voice/moonshine_api.py?utm_source=chatgpt.com)\]

```python
import subprocess
import re
from pathlib import Path

cmd = [
    "moonshine-voice", "download",
    "--stt", "--language", "en",
    "--model-arch", "2",
]

download = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    check=True,
)

print(download.stdout)
print(download.stderr[-2000:] if download.stderr else "")

output = download.stdout + "\n" + download.stderr

match = re.search(
    r"Downloaded model path:\s*(.+)",
    output,
)

if not match:
    raise RuntimeError(
        "Could not identify the model directory. "
        "Read the downloader output above and set MODEL_PATH manually."
    )

MODEL_PATH = match.group(1).strip()
print("Moonshine model directory:", MODEL_PATH)
print("Directory exists:", Path(MODEL_PATH).is_dir())
```

If the downloader does not print a path in this format, use the directory it prints at the end and assign it to `MODEL_PATH` manually.

## Cell 3 - Create a streaming transcription function

This feeds recorded audio in 100 ms chunks, as if it were arriving from a live microphone. The listener captures the first non-empty partial transcript and the final transcript.

```python
import time
import numpy as np
import threading

from moonshine_voice import Transcriber, ModelArch
from moonshine_voice import TranscriptEventListener

class MoonshineListener(TranscriptEventListener):
    def __init__(self):
        self.first_text_time = None
        self.start_time = None
        self.completed_lines = []
        self.latest_text = ""

    def on_line_text_changed(self, event):
        text = event.line.text or ""

        if text.strip():
            self.latest_text = text

            if self.first_text_time is None:
                self.first_text_time = (
                    time.perf_counter() - self.start_time
                )

    def on_line_completed(self, event):
        text = event.line.text or ""

        if text.strip():
            self.completed_lines.append(text)

def moonshine_transcribe(audio, sample_rate=16000,
                         chunk_ms=100, simulate_realtime=True):

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    listener = MoonshineListener()

    transcriber = Transcriber(
        model_path=MODEL_PATH,
        model_arch=ModelArch.TINY_STREAMING,
        update_interval=0.1,
    )

    transcriber.add_listener(listener)

    try:
        transcriber.start()

        start = time.perf_counter()
        listener.start_time = start

        chunk_size = int(sample_rate * chunk_ms / 1000)

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]

            transcriber.add_audio(chunk, sample_rate)

            if simulate_realtime:
                # Simulate audio arriving at its natural rate.
                time.sleep(len(chunk) / sample_rate)

        transcriber.stop()

        elapsed = time.perf_counter() - start
        duration = len(audio) / sample_rate

        return {
            "prediction": " ".join(listener.completed_lines).strip(),
            "partial_text": listener.latest_text,
            "ttft_s": listener.first_text_time,
            "total_time_s": elapsed,
            "audio_duration_s": duration,
            "rtf": elapsed / duration,
        }

    finally:
        transcriber.close()
```

## Cell 4 - Test one recording first

```python
moonshine_smoke = moonshine_transcribe(
    test_audio,
    sample_rate=16000,
    chunk_ms=100,
    simulate_realtime=True,
)

for key, value in moonshine_smoke.items():
    print(f"{key}: {value}")
```

Check that this produces a final transcript and a TTFT value before running the full dataset.

## Cell 5 - Benchmark the same 50 recordings

This uses your existing `manifest` and `get_audio_from_dataset()` function. It saves results after each recording.

```python
import pandas as pd
import numpy as np
from jiwer import wer, cer

MOONSHINE_RESULTS = "/content/moonshine_tiny_streaming_results.csv"

moonshine_results = []

for _, row in manifest.iterrows():
    idx = int(row["row"])

    try:
        audio, sr = get_audio_from_dataset(idx)

        result = moonshine_transcribe(
            audio,
            sample_rate=sr,
            chunk_ms=100,
            simulate_realtime=True,
        )

        reference = str(row["transcription"])
        prediction = result["prediction"]

        record = {
            "row": idx,
            "speech_status": row["speech_status"],
            "audio_duration_s": result["audio_duration_s"],
            "ttft_ms": (
                result["ttft_s"] * 1000
                if result["ttft_s"] is not None else np.nan
            ),
            "total_time_s": result["total_time_s"],
            "rtf": result["rtf"],
            "wer": wer(reference.lower(), prediction.lower()),
            "cer": cer(reference.lower(), prediction.lower()),
            "reference": reference,
            "prediction": prediction,
            "error": "",
        }

    except Exception as exc:
        record = {
            "row": idx,
            "speech_status": row["speech_status"],
            "error": repr(exc),
        }
        print(f"Failed row {idx}: {exc}")

    moonshine_results.append(record)

    pd.DataFrame(moonshine_results).to_csv(
        MOONSHINE_RESULTS,
        index=False,
    )

print("Saved:", MOONSHINE_RESULTS)
print("Completed recordings:", len(moonshine_results))
```

## Cell 6 - Summarize Moonshine results

```python
moonshine_df = pd.read_csv(MOONSHINE_RESULTS)

valid = moonshine_df[
    moonshine_df["error"].fillna("").eq("")
].copy()

moonshine_summary = (
    valid.groupby("speech_status")
    .agg(
        clips=("row", "nunique"),
        median_TTFT_ms=("ttft_ms", "median"),
        p95_TTFT_ms=("ttft_ms", lambda x: x.quantile(0.95)),
        mean_RTF=("rtf", "mean"),
        mean_WER=("wer", "mean"),
        mean_CER=("cer", "mean"),
    )
    .reset_index()
)

display(moonshine_summary.round(4))

moonshine_summary.to_csv(
    "/content/moonshine_tiny_summary.csv",
    index=False,
)
```

### Important for a fair comparison

Moonshine's current test simulates live audio arrival by sleeping between chunks. Your earlier Nemotron timing measured a different execution path, so do not directly rank their TTFT figures as if the protocols were identical. For the final comparison, document the distinction and, ideally, measure both under the same audio-arrival schedule.

Also, Moonshine's published English streaming model is approximately 34 million parameters, far below the challenge's minimum of 0.6 billion. It is a useful **efficiency baseline**, but it cannot replace Nemotron as the qualifying challenge model. \[[moonshine/docs/models/accuracy.md at main · moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine/blob/main/docs/models/accuracy.md?utm_source=chatgpt.com); [GitHub](https://github.com/moonshine-ai/moonshine-v2?utm_source=chatgpt.com)\]

## Prompt:
10/20/2026, 12:15:38 AM

lets go to some other model like zipformer and parakeet

## Response:
10/20/2026, 12:15:45 AM · gpt-6-thinking

> **Preparing benchmark code**
>
> I’m organizing two Colab-ready sections: Zipformer streaming with chunked decoding and TTFT measurement, followed by offline Parakeet CTC transcription. I’ll keep audio loading consistent and avoid mixing streaming and batch APIs.
>
> **Prepared benchmark code**
>
> 
>
> **Searched 5 websites**
>
> 
>
> Worked for 1m 7s

## Which model should we add first?

[![Zipformer: A faster and better encoder for automatic speech recognition](https://images.openai.com/static-rsc-4/KbqoDeFVlOIbRHmHimYNrrV5nVir9lMfl7rnkjVX3G3IlgJczdB_kwURPZzh6pHYoR4W7nH7LjG6gVGukEsfZsonXxSrNi3LcsSE_3fjRLgwCf6SEGZr0BxEUpwdDe3V_nvduDfR-UAL_DnYbJr3JFvv7V4RevPIlk8uU9qppOV_tZGJBjKLw1Fv2_8k8ohC?purpose=fullsize)](https://www.catalyzex.com/paper/zipformer-a-faster-and-better-encoder-for)

**1\. Streaming Zipformer**

(Best next step)

Use a pretrained streaming Zipformer through [github.com](https://github.com/k2-fsa/icefall?utm_source=chatgpt.com) and its documented deployment path through [k2-fsa.github.io](https://k2-fsa.github.io/sherpa/?utm_source=chatgpt.com)\. Zipformer is designed for streaming ASR and gives us a useful CPU inference baseline. \[[Zipformer Transducer](https://icefall.readthedocs.io/en/stable/recipes/Streaming-ASR/librispeech/zipformer_transducer.html?utm_source=chatgpt.com); [icefall/docs/source/recipes/Streaming-ASR/librispeech/zipformer\_transducer.rst at master · k2-fsa/icefall](https://github.com/k2-fsa/icefall/blob/master/docs/source/recipes/Streaming-ASR/librispeech/zipformer_transducer.rst?utm_source=chatgpt.com)\]

---

[![NVIDIA Open Sources Parakeet TDT 0.6B: Fastest ASR Model | Asif Razzaq posted on the topic | LinkedIn](https://images.openai.com/static-rsc-4/nfYH5RKHjCvk1a3PDaekC1hYjzEWLI7Sfc3Xy8ebuSJ89DIH8OqAPynjZm2sPMjSOesO5QrvFEiAQnmvgO5DWB9FrGUFtMCE2QAiKQEs5FfYVQHLjNbbz9rdTJAAqhzuC56uYB6GA5VT2VZNC4uuBTcaxjve6xzar33787ISH3V1qXyrhqinFb7gLF-FesNf?purpose=fullsize)](https://www.linkedin.com/posts/asifrazzaq_nvidia-open-sources-parakeet-tdt-06b-achieving-activity-7325396472420540416-EdQO)

**2\. Parakeet CTC 0.6B**

[huggingface.co](https://huggingface.co/nvidia/parakeet-ctc-0.6b?utm_source=chatgpt.com)

This model is approximately 600M parameters and therefore meets the challenge's stated size range. However, the standard Parakeet CTC checkpoint is an offline ASR baseline, not automatically a genuine streaming model. \[[nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b?library=transformers&utm_source=chatgpt.com); [transformers/docs/source/en/model\_doc/parakeet.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/parakeet.md?utm_source=chatgpt.com)\]

We should benchmark Zipformer as the streaming comparison and Parakeet as a separate offline reference. That distinction matters: offline inference time and streaming time-to-first-text are not directly equivalent.

Let's implement **Zipformer first**, using Sherpa-ONNX's published English streaming checkpoint. It has an INT8 ONNX version and is designed for online recognition, making it a suitable CPU baseline. \[[sherpa/docs/source/onnx/pretrained\_models/online-transducer/zipformer-transducer-models.rst at master · k2-fsa/sherpa](https://github.com/k2-fsa/sherpa/blob/master/docs/source/onnx/pretrained_models/online-transducer/zipformer-transducer-models.rst?utm_source=chatgpt.com); [Zipformer-transducer-based Models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html?utm_source=chatgpt.com)\]

## A. Zipformer - CPU streaming benchmark

### Cell 1 - Install and download the model

Run this in your existing Colab notebook. It creates a separate model directory and does not overwrite your Nemotron results.

```python
!pip -q install -U sherpa-onnx jiwer soundfile

import os
import glob
import tarfile
import subprocess
from pathlib import Path

ZIPFORMER_DIR = "/content/zipformer_streaming"
os.makedirs(ZIPFORMER_DIR, exist_ok=True)

archive = "/content/zipformer_streaming.tar.bz2"
url = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
)

if not glob.glob(ZIPFORMER_DIR + "/**/encoder-*.onnx", recursive=True):
    subprocess.run(
        ["wget", "-q", "-O", archive, url],
        check=True,
    )

    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(ZIPFORMER_DIR, filter="data")

    os.remove(archive)

print("Downloaded model files:")
for p in glob.glob(ZIPFORMER_DIR + "/**/*", recursive=True):
    if os.path.isfile(p) and (
        p.endswith(".onnx") or p.endswith("tokens.txt")
    ):
        print(p)
```

### Cell 2 - Load the streaming recognizer

```python
import sherpa_onnx

model_root = next(
    Path(ZIPFORMER_DIR).rglob("tokens.txt")
).parent

def find_model(pattern):
    matches = list(model_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Cannot find {pattern} in {model_root}"
        )
    # Prefer the INT8 version for CPU inference.
    int8 = [p for p in matches if ".int8.onnx" in p.name]
    return str(int8[0] if int8 else matches[0])

encoder = find_model("encoder-*.onnx")
decoder = find_model("decoder-*.onnx")
joiner = find_model("joiner-*.onnx")
tokens = str(model_root / "tokens.txt")

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=tokens,
    encoder=encoder,
    decoder=decoder,
    joiner=joiner,
    num_threads=1,
    sample_rate=16000,
    feature_dim=80,
    decoding_method="greedy_search",
)

print("Zipformer streaming recognizer loaded.")
print("Encoder:", encoder)
print("Decoder:", decoder)
print("Joiner:", joiner)
```

### Cell 3 - Measure streaming TTFT and final accuracy

This feeds the recording in 100 ms chunks without sleeping. It measures computational latency under chunked input, not simulated real-time arrival latency. We can run a separate real-time simulation afterward.

```python
import time
import numpy as np
import pandas as pd
from jiwer import wer, cer

def zipformer_transcribe(audio, sr=16000, chunk_ms=100):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if sr != 16000:
        raise ValueError("Resample audio to 16 kHz before inference.")
    if len(audio) == 0:
        raise ValueError("Empty audio.")

    stream = recognizer.create_stream()
    chunk_size = int(sr * chunk_ms / 1000)

    t0 = time.perf_counter()
    ttft = None
    previous_text = ""

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        stream.accept_waveform(sr, chunk)

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        current_text = recognizer.get_result(stream).text.strip()

        if current_text and ttft is None:
            ttft = time.perf_counter() - t0

        previous_text = current_text

    stream.input_finished()

    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    final_text = recognizer.get_result(stream).text.strip()
    elapsed = time.perf_counter() - t0
    duration = len(audio) / sr

    return {
        "prediction": final_text,
        "ttft_s": ttft,
        "total_time_s": elapsed,
        "audio_duration_s": duration,
        "rtf": elapsed / duration,
    }

# Test one of your existing recordings.
zip_smoke = zipformer_transcribe(test_audio)

print(zip_smoke)
```

If the smoke test passes, run the 50-clip evaluation.

### Cell 4 - Evaluate the same manifest

```python
ZIP_RESULTS = "/content/zipformer_cpu_results.csv"
zip_rows = []

for _, row in manifest.iterrows():
    idx = int(row["row"])

    try:
        audio, sr = get_audio_from_dataset(idx)

        if sr != 16000:
            import librosa
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=16000
            ).astype(np.float32)

        out = zipformer_transcribe(audio, sr=16000)
        reference = str(row["transcription"])
        prediction = out["prediction"]

        record = {
            "row": idx,
            "speech_status": row["speech_status"],
            "ttft_ms": (
                out["ttft_s"] * 1000
                if out["ttft_s"] is not None else np.nan
            ),
            "audio_duration_s": out["audio_duration_s"],
            "total_time_s": out["total_time_s"],
            "rtf": out["rtf"],
            "wer": wer(reference.lower(), prediction.lower()),
            "cer": cer(reference.lower(), prediction.lower()),
            "reference": reference,
            "prediction": prediction,
            "error": "",
        }

    except Exception as exc:
        record = {
            "row": idx,
            "speech_status": row["speech_status"],
            "error": repr(exc),
        }
        print(f"Failed row {idx}: {exc}")

    zip_rows.append(record)
    pd.DataFrame(zip_rows).to_csv(ZIP_RESULTS, index=False)

print("Saved:", ZIP_RESULTS)
display(pd.DataFrame(zip_rows).head())
```

## B. Parakeet CTC 0.6B - offline CPU baseline

Use [huggingface.co](https://huggingface.co/nvidia/parakeet-ctc-0.6b?utm_source=chatgpt.com)\. It is an approximately 600M-parameter English ASR model. The standard checkpoint performs offline transcription, so we can compare its final accuracy and processing speed, but not treat its first output as streaming TTFT. \[[nvidia/parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b?library=transformers&utm_source=chatgpt.com); [transformers/docs/source/en/model\_doc/parakeet.md at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/parakeet.md?utm_source=chatgpt.com)\]

Run this **after saving the Zipformer results**, because loading another large model can consume substantial RAM.

### Cell 5 - Load Parakeet

```python
import gc
import torch
import time
import numpy as np
import pandas as pd

from transformers import AutoProcessor, AutoModelForCTC

# Release Nemotron if it is no longer needed in memory.
if "model" in globals():
    del model
if "processor" in globals():
    del processor

gc.collect()

PARAKEET_ID = "nvidia/parakeet-ctc-0.6b"

parakeet_processor = AutoProcessor.from_pretrained(PARAKEET_ID)
parakeet_model = AutoModelForCTC.from_pretrained(
    PARAKEET_ID,
    torch_dtype=torch.float32,
).to("cpu").eval()

torch.set_num_threads(1)

print("Model:", PARAKEET_ID)
print("Model class:", type(parakeet_model).__name__)
print("Sample rate:", parakeet_processor.feature_extractor.sampling_rate)
```

### Cell 6 - Transcribe and measure CPU processing time

```python
def parakeet_transcribe(audio, sr=16000):
    import librosa

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000,
        ).astype(np.float32)

    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    start = time.perf_counter()

    inputs = parakeet_processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = parakeet_model(**inputs)
        predicted_ids = outputs.logits.argmax(dim=-1)
        prediction = parakeet_processor.batch_decode(
            predicted_ids
        )[0]

    elapsed = time.perf_counter() - start
    duration = len(audio) / 16000

    return {
        "prediction": prediction.strip(),
        "total_time_s": elapsed,
        "audio_duration_s": duration,
        "rtf": elapsed / duration,
    }

parakeet_smoke = parakeet_transcribe(test_audio, sr=16000)
print(parakeet_smoke)
```

### Cell 7 - Evaluate Parakeet on the same manifest

```python
from jiwer import wer, cer

PARAKEET_RESULTS = "/content/parakeet_ctc_cpu_results.csv"
parakeet_rows = []

for _, row in manifest.iterrows():
    idx = int(row["row"])

    try:
        audio, sr = get_audio_from_dataset(idx)
        out = parakeet_transcribe(audio, sr=sr)

        reference = str(row["transcription"])
        prediction = out["prediction"]

        record = {
            "row": idx,
            "speech_status": row["speech_status"],
            "total_time_s": out["total_time_s"],
            "audio_duration_s": out["audio_duration_s"],
            "rtf": out["rtf"],
            "wer": wer(reference.lower(), prediction.lower()),
            "cer": cer(reference.lower(), prediction.lower()),
            "reference": reference,
            "prediction": prediction,
            "error": "",
        }

    except Exception as exc:
        record = {
            "row": idx,
            "speech_status": row["speech_status"],
            "error": repr(exc),
        }
        print(f"Failed row {idx}: {exc}")

    parakeet_rows.append(record)
    pd.DataFrame(parakeet_rows).to_csv(
        PARAKEET_RESULTS, index=False
    )

print("Saved:", PARAKEET_RESULTS)
```

## C. What to compare

Once all runs finish, you will have three useful results files\:

- `nemotron_cpu_results.csv`
- `zipformer_cpu_results.csv`
- `parakeet_ctc_cpu_results.csv`

Compare them using a common table with WER, CER, RTF, CPU thread count, and peak RAM. Report TTFT only for systems whose streaming output has actually been measured.

One methodological caution: Zipformer's 100 ms chunks above are processed as fast as possible, while Nemotron's generation and Parakeet's offline pass use different execution paths. Therefore, treat these as preliminary system-level measurements. For a defensible final comparison, standardize audio arrival, warm-up, preprocessing, and timing boundaries, and record each model's supported streaming behavior.

## Prompt:
10/20/2026, 12:26:07 AM

lets make a mardown wuth the story I did plant...During an ASR competition of Neurips I realised I dont have a fixed model directory That I can use ( Suitable for my phone or cpu or edge devices ). Most of the ASR measured RTF, RTFx ,WER, TTFT measured on H100 or high end devices. 

ASR seems a kind of task everybody might need everyday. GPUs or server side inference cant be always possible or secured.

To solve this issue we require a benchmark for streaming inferences score. That'd precisely tell us what model we will need.

This will be like the OpenASR leaderboard, I wish they just included this section.

Therefore we can select the best model for inference and do task specific finetuning.

This is me brainstorming the possible things we wanna do. Obviously I will vibe code and some of the stuffs I look up from documentation everytime.

1. Load model
2. Load Dataset
3. Retrieve Audio
4. Streaming Check
5. Benchmark (WER, TTFT, RTF, Speed etc)

## Response:
10/20/2026, 12:26:16 AM · gpt-6-thinking

Here is a Markdown draft that preserves your original motivation and brainstorming, while making the research direction clearer. It is written as an early-stage research proposal, not as if the experiments have already been completed.

**document: **

# Beyond GPU Benchmarks: Finding the Right Streaming ASR Model for Everyday Devices

## 1. Where the Idea Started

During a NeurIPS ASR competition, I realised that I did not have a fixed set of speech recognition models that I could confidently use on my own phone, CPU, or other edge devices. I had experimented with different ASR models, but choosing the right one for a real deployment was not as straightforward as comparing their reported scores.

Most ASR benchmarks focus on metrics such as Word Error Rate (WER), Real-Time Factor (RTF), RTFx, and sometimes Time to First Token (TTFT). However, these measurements can be reported under very different hardware conditions, including high-end GPUs such as NVIDIA H100s. A model that performs well on a powerful GPU may not be the best choice for a laptop CPU, a mobile device, or a small edge computer.

This made me think about a simple question:

**If I want to run speech recognition locally on my own device, how do I know which model is actually suitable for me?**

## 2. Why This Matters

Automatic Speech Recognition (ASR) is becoming part of everyday life. People use it for voice typing, accessibility, meeting transcription, personal assistants, and many other tasks. However, running every speech recognition system through a remote server is not always practical.

There are several reasons to consider local inference:

- **Accessibility:** People should be able to use speech recognition on affordable devices without requiring access to expensive hardware.
- **Privacy:** Local inference can reduce the need to send sensitive audio to a remote server, although privacy also depends on how the application handles and stores data.
- **Connectivity:** Offline speech recognition can remain useful when internet access is unreliable or unavailable.
- **Cost:** Local inference may reduce recurring server costs for suitable workloads.
- **Latency:** A well-designed local streaming model may provide a responsive experience without relying on network communication.

The challenge is that model performance depends on more than recognition accuracy. It also depends on the device, inference runtime, quantization, streaming configuration, memory requirements, and the amount of delay a user can tolerate.

## 3. The Research Idea

I want to investigate whether we can build a practical **streaming ASR benchmark that helps people choose a model based on their actual deployment requirements**, rather than relying only on conventional accuracy scores or results obtained on powerful GPUs.

My starting point is the OpenASR leaderboard. I would like to explore whether a complementary section could report streaming performance on CPU and other resource-constrained devices.

The goal is not simply to identify the fastest model. A very fast model may make too many transcription errors, while a highly accurate model may require too much memory or produce its first useful text too late.

Instead, I want the benchmark to show the trade-offs clearly, so that a developer can select a suitable model before investing time in task-specific fine-tuning.

## 4. What Should the Benchmark Measure?

The benchmark would evaluate each model under a consistent experimental setup.

| Metric | What it tells us |
|---|---|
| Word Error Rate (WER) | How often the recognised words differ from the reference transcript. |
| Character Error Rate (CER) | How much the recognised text differs at the character level. |
| Time to First Text (TTFT) | How long the system takes to emit its first non-empty text. |
| Real-Time Factor (RTF) | How much computation time is required relative to the duration of the audio. |
| Processing speed | How quickly the system processes audio under the specified test conditions. |
| Peak RAM | How much system memory the model requires during inference. |
| Streaming stability | Whether partial transcripts are revised, delayed, or frequently changed as more audio arrives. |
| Hardware requirements | Which CPU, memory configuration, runtime, and inference settings were used. |

These metrics need precise definitions. For example, TTFT should specify whether timing begins when audio arrives, when the first audio chunk is available, or when inference starts. Similarly, RTF should distinguish offline processing from real-time audio streaming.

The benchmark should also distinguish **configured streaming latency** from experimentally measured TTFT. They describe different aspects of system behaviour.

## 5. My Initial Experimental Plan

I plan to build the evaluation in stages rather than immediately developing a new model.

### Step 1 - Load the model

Start with a small, open-weight ASR model that can run within the available memory and is suitable for the intended deployment.

### Step 2 - Load the dataset

Use a public speech dataset with reference transcripts. Include both ordinary English speech and challenging speech, such as dysarthric speech, where the dataset permits a meaningful evaluation.

### Step 3 - Retrieve and prepare audio

Load the recordings, check their sample rates and durations, and apply consistent preprocessing. Keep a reproducible manifest so that every model is evaluated on the same recordings.

### Step 4 - Verify genuine streaming inference

Determine whether the model supports incremental audio input and partial transcription. Test its documented streaming interface before running a large benchmark.

### Step 5 - Benchmark the model

Measure WER, CER, TTFT, RTF, processing speed, and memory use. Record failures and configuration details rather than silently excluding unsuccessful runs.

### Step 6 - Compare different models

Evaluate streaming models such as Nemotron and Zipformer, alongside suitable additional baselines. An offline model such as Parakeet CTC can provide an accuracy and processing-speed reference, but it should not be treated as a streaming system unless a genuine streaming implementation is used.

### Step 7 - Analyse the trade-offs

Determine which model performs best under different requirements. For example, a developer might prefer lower TTFT for interactive voice typing, lower WER for transcription, or lower memory use for an edge device.

### Step 8 - Publish reproducible results

Release the evaluation scripts, dataset manifest, model configurations, measured results, and documentation so that others can reproduce the experiments and contribute additional models.

## 6. What I Hope to Achieve

My longer-term goal is to create a useful reference for choosing ASR models under realistic deployment constraints.

Rather than asking only which model achieves the best WER, I want to answer questions such as:

- Which streaming ASR model gives the best accuracy on a single-threaded CPU?
- How much latency is introduced when a model is configured for more contextual lookahead?
- Does quantization improve inference speed enough to justify any loss in recognition accuracy?
- Does a model that performs well on clean English speech remain reliable on difficult speech?
- Which models provide a useful balance between accuracy, first-text latency, memory use, and computational cost?
- Can the benchmark help developers choose a suitable starting model before fine-tuning it for a particular task?

If the results support it, the benchmark could eventually become a community resource, similar in spirit to OpenASR but with a stronger focus on streaming and deployment constraints.

## 7. The Next Research Direction

I am treating this as an evolving research idea. At this stage, I am exploring existing models, reading their documentation, testing their streaming interfaces, and learning how to design fair experiments. I will use coding assistants and technical documentation where helpful, but the evaluation protocol, experimental decisions, interpretation of the results, and final conclusions need to be checked carefully.

The first milestone is to establish a reliable baseline across a small number of models. Only after that will I decide whether the main contribution should be a benchmark, a new evaluation metric, a deployment-oriented model selection method, or an improved inference approach.

**The central question remains simple: when someone needs speech recognition on an ordinary device, can we help them choose the model that actually works best for their needs?**

One important research distinction: your current plan is a strong starting point for a **benchmarking study**, but the benchmark itself is not yet a demonstrated research contribution. Its novelty will depend on what existing leaderboards omit, how rigorously you standardise the tests, and whether your evaluation reveals meaningful trade-offs that current benchmarks do not capture.

## Prompt:
10/20/2026, 12:33:49 AM

redo zipformer ---------------------------------------------------------------------------&#x20;
AttributeError                            Traceback (most recent call last)&#x20;
[/tmp/ipykernel_618/3609022373.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in \<cell line: 0>()&#x20;**&#x20;    51** &#x20;**&#x20;    52** # Test one of your existing recordings. ---> 53 zip_smoke = zipformer_transcribe(test_audio)&#x20;**&#x20;    54** &#x20;**&#x20;    55** print(zip_smoke) &#x20;
[/tmp/ipykernel_618/3609022373.py](https://colab.research.google.com/drive/1jvgBA-LNKtCXPPc6ZCXLqLLBTFvRTies?authuser=0#) in zipformer_transcribe(audio, sr, chunk_ms)&#x20;**&#x20;    26**             recognizer.decode_stream(stream)&#x20;**&#x20;    27**  ---> 28         current_text = recognizer.get_result(stream).text.strip()&#x20;**&#x20;    29** &#x20;**&#x20;    30**         if current_text and ttft is None: &#x20;
AttributeError: 'str' object has no attribute 'text'

## Response:
10/20/2026, 12:33:57 AM · gpt-6-thinking

The error is straightforward: `recognizer.get_result(stream)` returns a **string**, not an object with a `.text` attribute.

You only need to change one line in your Zipformer code.

### Fix

Replace\:

```python
current_text = recognizer.get_result(stream).text.strip()
```

with\:

```python
current_text = recognizer.get_result(stream).strip()
```

Also check the final transcript line. Replace\:

```python
final_text = recognizer.get_result(stream).text.strip()
```

with\:

```python
final_text = recognizer.get_result(stream).strip()
```

### Then rerun the smoke test

```python
zip_smoke = zipformer_transcribe(test_audio)

print("Transcript:", zip_smoke["prediction"])
print("TTFT (seconds):", zip_smoke["ttft_s"])
print("Total processing time:", zip_smoke["total_time_s"])
print("Audio duration:", zip_smoke["audio_duration_s"])
print("RTF:", zip_smoke["rtf"])
```

If it succeeds, proceed to the 50-clip Zipformer benchmark. Do not reload the model or redownload the checkpoint.

## Prompt:
10/20/2026, 12:39:44 AM

okay all run...lets get a comparison table

## Response:
10/20/2026, 12:39:56 AM · gpt-6-thinking

> **Building comparison table**
>
> I’m preparing robust code that reads each results CSV, standardizes metric names, separates configurations, aggregates WER/CER and latency, and clearly marks unavailable or incomparable measurements.
>
> **Built comparison table**
>
> 
>
> ****
>
> 
>
> Worked for 17s

Let's build one consolidated comparison table from the CSV files produced by your Colab experiments.

We'll compare **Nemotron 0.6B, Zipformer Streaming, Parakeet CTC 0.6B, and Moonshine Tiny Streaming** where results are available. The table will include accuracy, latency, processing speed, number of successful recordings, and errors.

One important distinction: Parakeet CTC is an offline baseline, whereas the other systems support streaming. We should not treat their TTFT measurements as equivalent unless they were measured under the same audio-arrival and timing protocol.

## Run this Colab cell

It reads your existing result files, recalculates corpus-level WER/CER where transcripts are available, and produces a consolidated table without rerunning the models.

```python
import os
import numpy as np
import pandas as pd
from jiwer import wer, cer
from IPython.display import display

FILES = {
    "Nemotron 3.5 0.6B": "/content/nemotron_cpu_results.csv",
    "Zipformer Streaming": "/content/zipformer_cpu_results.csv",
    "Parakeet CTC 0.6B": "/content/parakeet_ctc_cpu_results.csv",
    "Moonshine Tiny Streaming": "/content/moonshine_tiny_streaming_results.csv",
}

def normalize_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())

def safe_mean(series):
    return pd.to_numeric(series, errors="coerce").mean()

def safe_median(series):
    return pd.to_numeric(series, errors="coerce").median()

summary_rows = []
all_results = []

for model_name, filepath in FILES.items():
    if not os.path.exists(filepath):
        print(f"Not found (skipped): {filepath}")
        continue

    df = pd.read_csv(filepath)
    df["model"] = model_name

    if "error" not in df.columns:
        df["error"] = ""

    df["error"] = df["error"].fillna("").astype(str)
    valid = df[df["error"].str.strip().eq("")].copy()

    if valid.empty:
        print(f"No successful recordings: {model_name}")
        continue

    # Convert timing columns to numeric.
    for col in [
        "ttft_ms", "total_time_s", "audio_duration_s",
        "duration_s", "rtf", "wer", "cer"
    ]:
        if col in valid.columns:
            valid[col] = pd.to_numeric(valid[col], errors="coerce")

    # Standardize audio duration column.
    if "audio_duration_s" not in valid.columns:
        if "duration_s" in valid.columns:
            valid["audio_duration_s"] = valid["duration_s"]
        else:
            valid["audio_duration_s"] = np.nan

    # Derive RTF if missing.
    if "rtf" not in valid.columns:
        valid["rtf"] = (
            valid["total_time_s"] / valid["audio_duration_s"]
        )

    # Recalculate text metrics consistently when transcripts exist.
    if {"reference", "prediction"}.issubset(valid.columns):
        valid["ref_norm"] = valid["reference"].map(normalize_text)
        valid["pred_norm"] = valid["prediction"].map(normalize_text)

        valid["utterance_wer"] = [
            wer(r, p)
            for r, p in zip(valid["ref_norm"], valid["pred_norm"])
        ]
        valid["utterance_cer"] = [
            cer(r, p)
            for r, p in zip(valid["ref_norm"], valid["pred_norm"])
        ]

        corpus_wer = wer(
            valid["ref_norm"].tolist(),
            valid["pred_norm"].tolist()
        )
        corpus_cer = cer(
            valid["ref_norm"].tolist(),
            valid["pred_norm"].tolist()
        )
    else:
        corpus_wer = safe_mean(valid["wer"]) if "wer" in valid else np.nan
        corpus_cer = safe_mean(valid["cer"]) if "cer" in valid else np.nan

    # Keep Nemotron configurations separate.
    if "lookahead" in valid.columns:
        groups = valid.groupby("lookahead", dropna=False)
    else:
        groups = [(None, valid)]

    for lookahead, group in groups:
        duration = group["audio_duration_s"].sum()
        elapsed = group["total_time_s"].sum() if "total_time_s" in group else np.nan

        if {"reference", "prediction"}.issubset(group.columns):
            refs = group["reference"].map(normalize_text).tolist()
            preds = group["prediction"].map(normalize_text).tolist()
            group_wer = wer(refs, preds)
            group_cer = cer(refs, preds)
        else:
            group_wer = safe_mean(group["wer"]) if "wer" in group else np.nan
            group_cer = safe_mean(group["cer"]) if "cer" in group else np.nan

        total_rtf = elapsed / duration if duration > 0 and pd.notna(elapsed) else np.nan

        summary_rows.append({
            "Model": model_name,
            "Lookahead": lookahead if lookahead is not None else "N/A",
            "Successful clips": group["row"].nunique() if "row" in group else len(group),
            "Failed runs": int(df["error"].str.strip().ne("").sum()),
            "Corpus WER (%)": group_wer * 100,
            "Corpus CER (%)": group_cer * 100,
            "Median TTFT (ms)": (
                safe_median(group["ttft_ms"])
                if "ttft_ms" in group else np.nan
            ),
            "Median RTF": safe_median(group["rtf"]),
            "Aggregate RTF": total_rtf,
            "Aggregate RTFx": 1 / total_rtf if pd.notna(total_rtf) and total_rtf > 0 else np.nan,
        })

    all_results.append(df)

comparison = pd.DataFrame(summary_rows)

if comparison.empty:
    print("No usable result files were found. Check the filenames above.")
else:
    comparison = comparison.sort_values(
        ["Model", "Lookahead"], key=lambda s: s.astype(str)
    ).reset_index(drop=True)

    display(comparison.round(3))

    comparison.to_csv(
        "/content/asr_model_comparison.csv",
        index=False
    )
    print("Saved: /content/asr_model_comparison.csv")
```

## How to interpret the table

| Metric | Preferred direction | Meaning |
|--------|--------|--------|
| Corpus WER | Lower | Fewer word recognition errors |
| Corpus CER | Lower | Fewer character recognition errors |
| Median TTFT | Lower | Faster first text, subject to the timing protocol |
| Aggregate RTF | Lower | Less processing time per second of audio |
| Aggregate RTFx | Higher | More audio processed per unit of computation |
| Successful clips | Higher | More recordings completed successfully |

**Before making a final model ranking**, check that every model used the same 50 recordings, the same transcript normalization, the same CPU thread count, and comparable preprocessing. The code recalculates corpus WER/CER only over successful recordings, so compare the successful-clip counts alongside the scores.

If you want to see whether the systems behave differently on healthy versus dysarthric speech, we can generate a second table grouped by `speech_status` from the same files.

## Prompt:
10/20/2026, 12:43:18 AM

![image(20261019-184258).png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000b59c81fb84b24fd3a4b04365&fn=image%2820261019-184258%29.png&cd=attachment&ts=497899&p=fs&cid=1&sig=1d9b2a1f4dc2c9985f13d5070483828b384fe2f9f34ad20ac69162d5d845bbca&v=0)

here,,,can we get a plot as pareto,,with model sixe as radius?

## Response:
10/20/2026, 12:43:23 AM · gpt-5-4-thinking

> ****
>
> 
>
> **Plotting Pareto trade-offs**
>
> I’m using WER and RTF as the two objectives, with smaller values being better. Bubble areas represent model size; approximate sizes are placeholders you can replace with exact checkpoint counts.
>
> **Plotted Pareto trade-offs**
>
> 
>
> Worked for 8s

Yes — a **Pareto-style bubble plot** is a very good idea.

The cleanest main plot from your table is:

- **x-axis:** `Aggregate RTF` → lower is better  
- **y-axis:** `Corpus WER (%)` → lower is better  
- **bubble radius / size:** model size  
- **point labels:** model name + lookahead

This gives you an **accuracy vs speed trade-off** view, and the **Pareto frontier** is the set of points where no other point is better on both WER and RTF.

Because **Parakeet is offline**, it is still okay in this plot if you use **RTF**, but not ideal for a TTFT-based comparison.  
So I suggest:

1. **Main Pareto bubble plot:** `Aggregate RTF` vs `Corpus WER`
2. **Optional streaming-only plot:** `Median TTFT` vs `Corpus WER`

---

## Colab code for the Pareto bubble plot

Run this after you already created `/content/asr_model_comparison.csv`.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your comparison table
df = pd.read_csv("/content/asr_model_comparison.csv")

# -----------------------------
# Add approximate model sizes
# -----------------------------
# Edit these if you want more exact values for your specific checkpoints
size_map_million = {
    "Moonshine Tiny Streaming": 34,     # ~34M
    "Nemotron 3.5 0.6B": 600,           # 0.6B
    "Parakeet CTC 0.6B": 600,           # 0.6B
    "Zipformer Streaming": 20,          # approximate; edit if you know exact
}

df["Model Size (M)"] = df["Model"].map(size_map_million)

# Create a display label
df["Label"] = df["Model"].astype(str)
df.loc[df["Lookahead"].astype(str) != "N/A", "Label"] = (
    df["Model"] + " (L=" + df["Lookahead"].astype(str) + ")"
)

# Keep valid rows only
plot_df = df.dropna(subset=["Aggregate RTF", "Corpus WER (%)", "Model Size (M)"]).copy()

# -----------------------------
# Pareto frontier calculation
# For this plot: lower RTF and lower WER are both better
# -----------------------------
def is_pareto_efficient_min(points):
    """
    points: array of shape (n_points, 2)
    lower is better for both columns
    returns boolean mask
    """
    n = points.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue

        # A point is dominated if another point is <= in all dims
        # and < in at least one dim
        dominates_i = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        dominates_i[i] = False

        if dominates_i.any():
            is_efficient[i] = False

    return is_efficient

points = plot_df[["Aggregate RTF", "Corpus WER (%)"]].to_numpy()
plot_df["Pareto"] = is_pareto_efficient_min(points)

pareto_df = plot_df[plot_df["Pareto"]].sort_values("Aggregate RTF")

# -----------------------------
# Bubble size scaling
# -----------------------------
# matplotlib bubble size is area, not radius
# This scaling makes small models visible but keeps 600M larger
plot_df["bubble_size"] = 80 + 8 * np.sqrt(plot_df["Model Size (M)"])

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12, 8))

# Plot all points
plt.scatter(
    plot_df["Aggregate RTF"],
    plot_df["Corpus WER (%)"],
    s=plot_df["bubble_size"] ** 2 / 10,
    alpha=0.6,
)

# Highlight Pareto points
plt.scatter(
    pareto_df["Aggregate RTF"],
    pareto_df["Corpus WER (%)"],
    s=(80 + 8 * np.sqrt(pareto_df["Model Size (M)"])) ** 2 / 10,
    alpha=0.9,
    edgecolors="black",
    linewidths=1.5,
)

# Connect Pareto frontier
plt.plot(
    pareto_df["Aggregate RTF"],
    pareto_df["Corpus WER (%)"],
    linestyle="--",
    marker="o",
)

# Labels
for _, row in plot_df.iterrows():
    plt.annotate(
        row["Label"],
        (row["Aggregate RTF"], row["Corpus WER (%)"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

plt.xlabel("Aggregate RTF (lower is better)")
plt.ylabel("Corpus WER (%) (lower is better)")
plt.title("ASR Pareto Bubble Plot: Accuracy vs Speed")
plt.grid(True)

# Optional: invert both axes visually so "better" moves toward top-right
# Comment these out if you prefer the normal direction
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
```

---

## Optional: streaming-only Pareto plot using TTFT

This is also very useful, because it focuses on **user-perceived responsiveness**.

- **x-axis:** `Median TTFT (ms)` → lower is better  
- **y-axis:** `Corpus WER (%)` → lower is better  
- **bubble size:** model size  
- **exclude Parakeet** because TTFT is not meaningful there in your setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/content/asr_model_comparison.csv")

size_map_million = {
    "Moonshine Tiny Streaming": 34,
    "Nemotron 3.5 0.6B": 600,
    "Parakeet CTC 0.6B": 600,
    "Zipformer Streaming": 20,
}

df["Model Size (M)"] = df["Model"].map(size_map_million)
df["Label"] = df["Model"].astype(str)
df.loc[df["Lookahead"].astype(str) != "N/A", "Label"] = (
    df["Model"] + " (L=" + df["Lookahead"].astype(str) + ")"
)

plot_df = df[
    df["Median TTFT (ms)"].notna()
    & df["Corpus WER (%)"].notna()
    & df["Model Size (M)"].notna()
].copy()

# Optional: exclude offline model
plot_df = plot_df[plot_df["Model"] != "Parakeet CTC 0.6B"].copy()

def is_pareto_efficient_min(points):
    n = points.shape[0]
    is_efficient = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates_i = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        dominates_i[i] = False
        if dominates_i.any():
            is_efficient[i] = False

    return is_efficient

points = plot_df[["Median TTFT (ms)", "Corpus WER (%)"]].to_numpy()
plot_df["Pareto"] = is_pareto_efficient_min(points)

pareto_df = plot_df[plot_df["Pareto"]].sort_values("Median TTFT (ms)")
plot_df["bubble_size"] = 80 + 8 * np.sqrt(plot_df["Model Size (M)"])

plt.figure(figsize=(12, 8))

plt.scatter(
    plot_df["Median TTFT (ms)"],
    plot_df["Corpus WER (%)"],
    s=plot_df["bubble_size"] ** 2 / 10,
    alpha=0.6,
)

plt.scatter(
    pareto_df["Median TTFT (ms)"],
    pareto_df["Corpus WER (%)"],
    s=(80 + 8 * np.sqrt(pareto_df["Model Size (M)"])) ** 2 / 10,
    alpha=0.9,
    edgecolors="black",
    linewidths=1.5,
)

plt.plot(
    pareto_df["Median TTFT (ms)"],
    pareto_df["Corpus WER (%)"],
    linestyle="--",
    marker="o",
)

for _, row in plot_df.iterrows():
    plt.annotate(
        row["Label"],
        (row["Median TTFT (ms)"], row["Corpus WER (%)"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

plt.xlabel("Median TTFT (ms) (lower is better)")
plt.ylabel("Corpus WER (%) (lower is better)")
plt.title("Streaming ASR Pareto Bubble Plot: Accuracy vs First-Text Latency")
plt.grid(True)

plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
```

---

## What this will likely show from your current numbers

From the table you shared, the likely story is:

- **Parakeet CTC 0.6B**: best WER among tested systems, also strong RTF, but **offline**
- **Zipformer Streaming**: by far the fastest streaming option, very low TTFT and RTF, but worse WER than Parakeet and Nemotron-L3
- **Nemotron 3.5 0.6B (L=13)**: strongest Nemotron speed/accuracy balance in your current results
- **Moonshine Tiny Streaming**: small model, practical edge baseline, but not the strongest on WER

So the Pareto interpretation may be something like:

- **Overall speed frontier:** Zipformer
- **Overall accuracy frontier:** Parakeet
- **Streaming trade-off frontier:** Zipformer vs Nemotron-L13
- **Edge-size frontier:** Moonshine

---

## My suggestion for the paper/report

Use **two plots**, not just one:

1. **RTF vs WER bubble Pareto**
   - includes all models
   - best “system efficiency” view

2. **TTFT vs WER bubble Pareto**
   - only streaming models
   - best “real-time usability” view

If you want, I can now give you **one polished final plotting cell** that:
- makes both plots,
- uses nice labels,
- adds a legend for bubble sizes,
- and saves the figures as PNG for your report.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)