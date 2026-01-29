# SIMD-Optimized Neural Network Inference Engine (C++)

A **from-scratch neural network inference engine** written in **C++ with AVX2 SIMD intrinsics**, designed for **low-latency, single-position inference** in chess-engine–style workloads.

The project emphasizes **practical inference speed and systems-level control** rather than framework abstractions. All layers, quantization steps, and matrix multiplications are implemented manually.

---

## Key Features

- **AVX2-optimized int8 inference pipeline**
  - Float → int8 quantization
  - int8 × int8 → int32 accumulation
  - int32 → float rescaling
- **Custom SIMD matrix multiplication**
  - Manual vectorization using `_mm256_*` intrinsics
  - Cache-aware blocking
- **Latency-oriented design**
  - Batch size = 1
  - Optimized for real-time decision systems (e.g. chess engines)
- **End-to-end inference**
  - Input embedding
  - Residual evaluation blocks
  - Value head
  - Optional policy head over legal moves

---

## Architecture Overview

The network follows a lightweight MLP-style architecture:

- Input projection layer
- One or more residual evaluation blocks
- Value head for scalar evaluation
- Policy head that scores only legal moves (batched)

The policy head avoids evaluating the full action space by processing only legal moves, improving inference efficiency in structured domains such as chess.

---

## Performance

Benchmarks were run on a single CPU core using AVX2, with batch size = 1.

| Configuration | FLOPs | Latency |
|--------------|------:|--------:|
| Full Network (~900k FLOPs) | ~23 µs |
| Peak Throughput | — | ~40 GFLOPs/s | Single threaded, consumer CPU

**Peak throughput (~40 GFLOPs/s)** was observed at:
- `input_size = 1024`
- `hidden_dim = 256`

Larger configurations become memory- and cache-bound, which is expected for single-position inference workloads.

---

## Design Notes

- Optimized for **latency**, not throughput or batching
- No external ML libraries (no Eigen, BLAS, or frameworks)
- Emphasis on predictable runtime and memory access patterns
- Performance limited primarily by cache behavior at large hidden sizes

---

## Build & Run

```bash
g++ -O3 -mavx2 -march=native -std=c++17 main.cpp -o simd_nn
./simd_nn
