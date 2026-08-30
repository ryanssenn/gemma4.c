# gemma4.c

Gemma 4 E2B CPU inference in 700 lines of pure C.

An educational project made to understand how LLM inference works. The full inference path is implemented in one file without external libraries.

<img width="800" height="339" alt="demo" src="https://github.com/user-attachments/assets/2de47c35-ee34-473e-9164-635c82377267" />

## Benchmark

Measured on an AMD Ryzen 7 7700 using the default native build.

### Prefill (tok/s)

| Prompt tokens | gemma4.c (int8) | llama.cpp (Q8_0) |
| ------------: | --------------: | ----------------: |
| 512 | 636.85 | 275.44 |
| 2,048 | 495.65 | 258.15 |
| 8,192 | 253.85 | 220.97 |
| 16,384 | 106.61 | 188.85 |

### Decode (tok/s)

| Starting context | gemma4.c (int8) | llama.cpp (Q8_0) |
| ---------------: | --------------: | ----------------: |
| 512 | 25.04 | 22.71 |
| 2,048 | 23.96 | 21.28 |
| 8,192 | 20.28 | 17.80 |
| 16,384 | 16.78 | 14.23 |

Prefill measures the time to process the stated number of prompt tokens. Decode first fills the KV cache to the stated depth, then measures 128 single-token steps. Results are the mean of three timed runs after one discarded warmup. Both implementations use FP32 KV caches, 512-token batches, eight CPU threads, and native builds.

The benchmark command takes the prefill length followed by the number of decode steps:

```bash
./run -m ./gemma4-E2B-int8.bin --bench 512 128
```

## Quick start

You need a CPU with AVX2, an OpenMP-capable C compiler, and `make`. The model takes about 5.0 GB of disk space, and 8 GB of RAM is recommended.

Clone the repository and download the ready-to-run model:

```bash
git clone https://github.com/ryanssenn/gemma4.c
cd gemma4.c
python3 -m pip install -U huggingface_hub
hf download QmogAI/gemma4-e2b-int8 gemma4-E2B-int8.bin --local-dir .
```

On Linux:

```bash
make
./run -t 0 -n 256 "Why is the sky blue?"
```

On Windows, use a MinGW-w64 environment that provides `gcc`, OpenMP, and `make`:

```powershell
make win64 WINCC=gcc
.\run.exe -t 0 -n 256 "Why is the sky blue?"
```

## Options

- `-m` sets the model path. The default is `gemma4-E2B-int8.bin`.
- `-t` sets the temperature. The default is `1.0`. Use `0` for greedy decoding.
- `-n` sets the maximum number of tokens to generate. The default is `1,024`.
- `--bench` measures prefill and decode throughput.
- `--dump-logits` writes prompt logits as float32 binary data.

## Model

The C runtime cannot read the original checkpoint directly. `exporter.py` takes the tokenizer and language-model weights from the Hugging Face checkpoint and writes them in the exact layout used by `gemma4.c`.

Matrix weights are stored as int8 with FP16 scales. Inputs to linear layers are dynamically quantized to int8 while the rest of the activations remain float32. The resulting file is about 5.0 GB (4.7 GiB).

To create it yourself instead:

```bash
python3 -m pip install -r requirements.txt
python3 exporter.py /path/to/gemma-4-E2B-it-qat-q4_0-unquantized -o ./gemma4-E2B-int8.bin
```

Python is only needed to export the model or run numerical validation. Once the `.bin` file exists, inference runs entirely through the C program.

## Numerical validation

The implementation is validated against the Hugging Face Transformers reference implementation running Google's unquantized [Gemma 4 E2B QAT checkpoint](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-unquantized) in BF16. Both implementations process the complete Éva Gauthier article from the WikiText-103 validation split under teacher forcing. The passage contains 2,387 text tokens, and the added BOS token brings the comparison to 2,388 model positions.

| Metric | Result |
| ------ | -----: |
| Top-1 agreement | 2,305 / 2,388 (96.5%) |
| Mean KL divergence | 0.005207 |

An exact match is not expected because gemma4.c uses int8 matrix weights and linear inputs while the reference runs the unquantized checkpoint in BF16. The output distributions nevertheless remain closely aligned.

Build the C runtime first (`make` or `make win64`). The complete validation peaks at about 10 GiB of RAM:

```bash
python3 validation.py
```

Pass a smaller token count for a quicker check:

```bash
python3 validation.py 64
```

`validation.py` uses the runtime's `--dump-logits` flag to collect float32 logits after each prompt position. The flag can also be used directly when comparing gemma4.c with another implementation:

```bash
./run -m ./gemma4-E2B-int8.bin --dump-logits "Why is the sky blue?" > logits.bin
```

## Repository contents

- `gemma4.c` contains the tokenizer, model definitions, kernels, transformer, KV cache, and generation loop.
- `exporter.py` converts the original checkpoint into the binary layout read by the C runtime.
- `validation.py` compares the runtime's logits with Hugging Face Transformers.
- `validation.txt` contains the WikiText-103 passage used for numerical validation.
- `win.c` and `win.h` provide the small Windows memory-mapping compatibility layer.
- `Makefile` builds the Linux or Windows executable.
