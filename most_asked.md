My most challenging recent project was adapting Moonshine-base (61.5M parameters, English-only) into an edge-efficient Bengali ASR model, presented as a poster at the ICML 2026 MuslimML Workshop.

During a Bangla ASR competition, we found that Moonshine ran smoothly on CPU, unlike models that needed a GPU to be fast. But fine-tuning it, with both LoRA and full fine-tuning, left WER near 100%.

The breakthrough came from inspecting the tokenizer. It was built for English and fragmented Bengali into about 9 tokens per word. Instead of training a new one, I found a Bengali BERT model and transplanted its vocabulary. After refinetuning, WER fell from ~100% to 21.82%.

The main obstacle was compute and dataset: a lab PC with an RTX 4070, available only outside lab hours, at about 3 hours per epoch. I had to learn robust checkpoint resuming to train around that. Also we built an 800 hour dataset (Lipighor 882 sstt, accepted to SALMA workshop 2026) in the competition that has been used for model finetuning.

Trade-offs: I prioritized efficiency over the lowest WER. Whisper Medium is marginally better but far larger, while mine runs on CPU. I reused an existing vocabulary rather than building one, and I shipped a demo instead of finishing Android deployment.

Workload: For about 6 weeks, I trained model most nights, setting aside other works , roughly 10-15 hours per week.
