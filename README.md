# LM_Dec_AttnFree_Mega

Mega - AttnFree family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: LM_2022_Mega-pytorch
- Upstream / license: lucidrains/Mega-pytorch (MIT)

## Differences from the source ([Unified] protocol)

- depth 6 (upstream 8)
- fp32 (its FFT does not support bf16) and micro-batch 4 with gradient accumulation (same update): about 0.75 GB per 512-token sample in fp32; the first LM-05 run spilled past 8 GB and stalled

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 12.39 M
- Causality check: pass (largest change of earlier outputs 6.6e-06)
- Overfit check (4 x 256 bytes): 5.687 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.686 / test 1.692 bits per byte (time-capped)
- Peak GPU memory at length 2048 (batch 1): train 3033 MB, inference 709 MB
- Peak GPU memory at length 8192 (batch 1): train 16888 MB, inference 2926 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.01 (keys removed 0.01); 4 pairs @ 2048: 0.02 (keys removed 0.02); 16 pairs @ 512: 0.02 (keys removed 0.02); 16 pairs @ 2048: 0.02 (keys removed 0.02)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check mega
python run.py lm02 mega      # enwik8 is downloaded to data/ on first use
```
