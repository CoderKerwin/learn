---
title: "Why a 27B Model Runs at 5 tok/s on CPU: llama.cpp, vLLM, and GPUs"
categories: [AI, Inference]
tags: [llama.cpp, vllm, qwen, gguf, cpu, gpu, inference]
---

I recently replaced two old llama.cpp services on a CPU-only server with a single Qwen3.8-27B service. The new service worked on the first real request, but generation speed was only about 5 tokens per second. That raised three questions:

- Is 5 tok/s a deployment mistake, or is it normal for a CPU?
- Can AVX-512 or another CPU trick make it much faster?
- Would llama.cpp or vLLM on a GPU reach tens of tokens per second?

This post answers those questions with measurements from the deployment and the published hardware limits.

## The Test Setup

The CPU server has two Intel Xeon Platinum 8358 processors: 64 physical cores and 128 logical CPUs in two NUMA nodes. Intel lists the 8358 with 8 DDR4-3200 memory channels, AVX-512 with two FMA units, and support for two-socket systems.

The model is [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), using the `Qwen3.8-27B-UD-Q4_K_M.gguf` file. Hugging Face reports that file as 16,464,440,224 bytes, roughly 16.5 GB. The model card describes Qwen3.8-27B as a dense 27B model with thinking mode enabled by default.

I built the latest llama.cpp with Intel oneAPI and MKL, following the Intel oneMKL instructions from the llama.cpp build guide:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -B build-intel -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Intel10_64lp \
  -DGGML_OPENMP=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
cmake --build build-intel --target llama-server -j "$(nproc)"
```

The current llama.cpp source failed with the distribution's CMake 3.16 because its UI asset script requires CMake 3.18 or newer. A user-local CMake 3.31 solved that without changing the system package.

The service ran with these arguments:

```bash
llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 11435 \
  --ctx-size 32768 \
  --threads 128 \
  --threads-batch 128 \
  --parallel 4 \
  --jinja
```

With `--ctx-size 32768` and `--parallel 4`, the server log showed four slots with 8,192 context tokens each.

## What the CPU Delivered

Two short OpenAI-compatible chat requests produced these server-side timings:

| Request | Prompt processing | Generation |
|---|---:|---:|
| Cold 57-token prompt | 7.6 tok/s | 4.9 tok/s |
| 56-token prompt with 42 cached tokens | 16.1 tok/s | 4.8 tok/s |

The prompts were too short to measure prefill throughput meaningfully. The stable number is generation: about **4.8 tok/s** for a single request.

Because thinking was enabled, a small `max_tokens` value could be consumed entirely by `reasoning_content`, leaving `content` empty. That is expected behavior, not a server failure. llama.cpp exposes `--reasoning`, `--reasoning-budget`, and `--reasoning-effort` if you need to control this at the server level.

## Why CPU Decode Is Slow

LLM inference has two phases. Prefill processes the prompt in parallel. Decode then generates output tokens one at a time. NVIDIA's inference optimization guide describes decode as a matrix-vector-like operation whose latency is dominated by moving weights, keys, values, and activations from memory, rather than by arithmetic. In other words, decode is memory-bound.

For a dense model, generating one token requires reading approximately all model weights. That gives a simple upper-bound estimate:

```text
maximum tokens/s ≈ usable memory bandwidth / bytes read per token
```

The following numbers are theoretical ceilings, not benchmarks:

| Hardware | Published or calculated peak bandwidth | Ceiling for 16.46 GB of weights |
|---|---:|---:|
| One Xeon 8358, 8 channels of DDR4-3200 | 204.8 GB/s | about 12 tok/s |
| Two Xeon 8358 sockets | 409.6 GB/s | about 25 tok/s |
| NVIDIA RTX A6000 | 768 GB/s | about 47 tok/s |

The Xeon figures assume all eight channels on each socket are populated and run at DDR4-3200: `8 channels × 3,200 MT/s × 8 bytes`. The two-socket ceiling also assumes a perfect split across both NUMA nodes. Real inference cannot reach these peaks because of KV cache reads, activations, dequantization, thread synchronization, and cross-socket traffic.

The observed 4.8 tok/s implies roughly 79 GB/s of effective weight traffic. That is well below the theoretical peak, but it is not surprising for a two-socket CPU run. The server is doing real work; the bottleneck is that every generated token must stream a 16.5 GB model through the memory system.

## What AVX-512 and MKL Can and Cannot Do

AVX-512 is already part of the picture. The llama.cpp README lists AVX, AVX2, AVX512, and AMX support for x86, and the Intel oneMKL build example enables `GGML_NATIVE`. A native build targets the machine's own instruction set.

MKL has a narrower role than many people expect. The llama.cpp build guide says BLAS can improve prompt processing with batch sizes above 32, but **does not affect generation performance**. So MKL may help long prompts, but it will not turn 5 tok/s decode into 50 tok/s.

Wider SIMD increases compute throughput. It does not remove the need to read the weights for each token. When decode is memory-bound, faster arithmetic has limited impact.

## CPU Tuning That Is Still Worth Trying

These options may improve throughput or latency, but they do not change the memory-bandwidth ceiling:

- **Tune NUMA placement.** `llama-server --numa` supports `distribute`, `isolate`, and `numactl`. On a two-socket server, test these modes and thread counts instead of assuming that using all 128 logical CPUs is fastest.
- **Use a smaller quantization.** Q3 or Q2 files reduce bytes per token and may raise the ceiling, at the cost of model quality.
- **Limit reasoning.** A reasoning budget does not increase tok/s, but it can reduce the number of tokens generated before the final answer.
- **Try speculative decoding.** `llama-server` supports speculative decoding types including `draft-mtp`, and the Qwen3.8 GGUF repository includes an MTP file. The gain depends on how often draft tokens are accepted, so benchmark it on your workload.
- **Right-size context and slots.** Context and parallel slots mainly affect KV cache memory and concurrency. They are not a direct fix for single-request decode speed.

## llama.cpp or vLLM on the GPU?

The GPU server has one NVIDIA RTX A6000. NVIDIA specifies 48 GB of GDDR6 ECC memory and 768 GB/s of memory bandwidth. Using the same estimate, the 16.46 GB Q4_K_M model has a ceiling of about 47 tok/s. Real throughput should be lower than that ceiling, but the GPU raises the limit by about an order of magnitude compared with the measured CPU result.

### llama.cpp

llama.cpp supports NVIDIA GPUs through its CUDA backend and can store model layers in VRAM with `--n-gpu-layers`. It also supports CPU+GPU hybrid inference when a model does not fit in VRAM. A 16.5 GB Q4_K_M model fits on a 48 GB card and leaves room for KV cache.

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j "$(nproc)"

./build/bin/llama-server \
  --model /path/to/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --n-gpu-layers all \
  --ctx-size 32768 \
  --parallel 4 \
  --host 0.0.0.0 \
  --port 11435
```

This is the fastest path when you already have the GGUF file. `llama-server` also supports parallel decoding, multi-user operation, and continuous batching.

### vLLM

vLLM is designed for high-throughput serving. Its documentation highlights PagedAttention, continuous batching, prefix caching, CUDA graphs, and a wide range of quantization formats.

The BF16 checkpoint does not fit on this card. Hugging Face lists 27,320,697,856 parameters for the model, so BF16 weights alone need about 54.6 GB before KV cache is allocated. On a single 48 GB A6000, use a quantized format supported by vLLM instead.

vLLM can load GGUF files through the `vllm-gguf-plugin`, but its documentation describes GGUF support as highly experimental and under-optimized. For a production vLLM deployment, prefer a format such as AWQ, GPTQ, or FP8 that vLLM supports natively, provided a compatible checkpoint exists for this model.

vLLM also supports x86 CPUs. It does not remove the CPU memory-bandwidth limit for single-request decode.

### Choosing Between Them

| Goal | Better starting point |
|---|---|
| Reuse the existing GGUF file quickly | llama.cpp |
| Single-user or small-team API | llama.cpp |
| CPU-only or CPU+GPU hybrid inference | llama.cpp |
| High-concurrency serving on NVIDIA GPUs | vLLM |
| Native AWQ, GPTQ, or FP8 serving | vLLM |

For one chat stream, both engines are constrained by the same GPU memory bandwidth. vLLM's larger advantage is usually aggregate throughput under concurrent load, not a dramatic speedup for a single conversation.

## Summary

The 5 tok/s CPU result is consistent with the hardware. Decode is memory-bound, a dense 27B Q4_K_M model reads about 16.5 GB of weights per generated token, and AVX-512 or MKL cannot eliminate that traffic. CPU tuning can help at the margins, but a GPU is the direct path to tens of tokens per second. On one RTX A6000, llama.cpp is the pragmatic choice for this GGUF model, while vLLM becomes more attractive for high-concurrency serving with a natively supported quantized checkpoint.

## References

- [llama.cpp repository](https://github.com/ggml-org/llama.cpp) - Lists AVX, AVX2, AVX512, and AMX support on x86, integer quantization, CUDA kernels, and CPU+GPU hybrid inference.
- [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) - Documents the Intel oneMKL and CUDA build options, and states that BLAS helps prompt processing but not generation performance.
- [llama.cpp HTTP server README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md) - Documents `--ctx-size`, `--parallel`, `--n-gpu-layers`, `--numa`, reasoning controls, speculative decoding types, and continuous batching.
- [NVIDIA: Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - Explains the prefill and decode phases and why decode is memory-bound.
- [Intel Xeon Platinum 8358 specifications](https://www.intel.com/content/www/us/en/products/sku/212282/intel-xeon-platinum-8358-processor-48m-cache-2-60-ghz/specifications.html) - Provides the core count, DDR4-3200 support, eight memory channels, AVX-512 FMA units, and two-socket scalability.
- [NVIDIA RTX A6000 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a6000-datasheet.pdf) - Specifies the RTX A6000's 48 GB GDDR6 memory and 768 GB/s memory bandwidth.
- [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) - Describes Qwen3.8-27B as a dense model with thinking enabled by default and hosts the Q4_K_M and MTP GGUF files.
- [Hugging Face model API for unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF?blobs=true) - Reports the Q4_K_M file size and the model parameter count used in the bandwidth and BF16 calculations.
- [vLLM documentation](https://docs.vllm.ai/en/latest/) - Lists PagedAttention, continuous batching, supported quantization formats, and x86 CPU support.
- [vLLM GGUF documentation](https://docs.vllm.ai/en/latest/features/quantization/gguf/) - Describes GGUF support as highly experimental and requiring `vllm-gguf-plugin`.
