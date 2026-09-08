# Computational Benchmark Report: Phase 5 Feature Extraction

## Benchmark Setup
- **Evaluated Records**: 8 representative records (4 Challenge 2015, 4 VFDB; 4 VT and 4 VF).
- **Total Windows Benchmarked**: 768.

## Feature-by-Feature Timing Profile
| Feature Component | Mean (ms) | Median (ms) | P90 (ms) | Max (ms) |
|:---|---:|---:|---:|---:|
| `lz_original_N1800` | 606.90 | 181.88 | 1616.18 | 9960.48 |
| `lz_optimized_N450` | 6.36 | 2.60 | 16.03 | 86.98 |
| `sampen_N225` | 5.93 | 5.59 | 6.98 | 10.66 |
| `sampen_N150` | 2.71 | 2.54 | 3.26 | 5.38 |
| `spectral_welch` | 1.42 | 1.32 | 1.85 | 3.15 |
| `autocorrelation` | 0.48 | 0.45 | 0.62 | 1.39 |
| `permutation_entropy` | 0.30 | 0.28 | 0.41 | 1.18 |
| `hjorth` | 0.13 | 0.12 | 0.18 | 0.62 |

## Primary Computational Bottleneck
- **Identified Bottleneck**: `lz_original_N1800` (606.90 ms/window, accounting for 98.7% of per-window calculation time in unoptimized configuration).
- **Secondary Factor**: One-time archive decompression of Challenge 2015 `training.zip` takes approximately 65.0 seconds.

## Full Dataset Runtime Projections (14,618 Windows)
- **Strategy A (Unoptimized LZ N=1800 + SampEn N=225)**: `615.17 ms/win` -> **150.96 minutes** (9057.5 s).
- **Strategy B (Optimized LZ N=450 + SampEn N=225)**: `14.63 ms/win` -> **4.65 minutes** (278.8 s).
- **Strategy C (Fast LZ N=450 + SampEn N=150)**: `11.40 ms/win` -> **3.86 minutes** (231.7 s).

## Recommendation for Full Phase 5 Extraction
Implement **Strategy B**:
1. Downsample LZ complexity to 90 Hz (N=450), which preserves all cardiac frequency components (0.5–30 Hz) while reducing LZ execution time significantly.
2. Vectorize Sample Entropy with N=225 (45 Hz, $m=2, r=0.2\sigma$) using float32 arrays.
3. Pre-extract `training.zip` once into a temporary folder to eliminate archive seek delays.
4. Provide per-record streaming progress reporting with `flush=True`, elapsed time, and ETA.
