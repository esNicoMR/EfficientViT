# EfficientViT-B1 CPU Statistics

Processed images: 25 / 25; status: completed

MACs: 12,980,364,800

Potential zero-product skips: 274,525,955 (2.115%)

Padding-only products: 76,990,800

Potential skips excluding padding: 197,535,155

## MAC Opportunities by Operation

| Operation | MACs | Share of model MACs | Potential skips within operation | Skips excluding padding within operation |
|---|---:|---:|---:|---:|
| Convolutions | 12,665,676,800 | 97.58% | 1.17% | 0.56% |
| Linear layers | 101,440,000 | 0.78% | 9.55% | 9.55% |
| Attention: `Vpad × Kᵀ` | 106,624,000 | 0.82% | 60.43% | 60.43% |
| Attention: `VK × Q` | 106,624,000 | 0.82% | 49.43% | 49.43% |
| **Total** | **12,980,364,800** | **100.00%** | **2.11%** | **1.52%** |

Source: `macs_by_layer.csv`, grouped by `operation` across all processed images. Share = operation MACs / model MACs × 100. Potential skips = operation skippable MACs / operation MACs × 100. The final column subtracts padding MACs before division. These are ratios of summed counts, not averages of layer percentages; the Total row is weighted by MAC count. Percentages are rounded.

See macs_by_layer.csv for weighted layer totals, images.csv for image variation, tensors_by_layer.csv for aggregate sparsity, tensors.jsonl for per-image tensor/structured sparsity, macs.jsonl for per-image layer MAC counts, and weights.csv for weights.

Counts use exact zeros, FP32 CPU, batch size 1, eval and inference mode.
MACs cover Conv2d, Linear and both LiteMLA matrix products, including its padded
denominator channel. Biases and non-MAC arithmetic are NOT counted as MAC savings.
Either-zero products are counted once; both-zero is the overlap. Padding MACs are
a subset of skippable MACs; skippable minus padding excludes zero-padding opportunities.
Conv left operand is weight; linear left operand is activation; attention operand
names give their order. Zero tensor entries alone do not imply skippable work.
Structured channels/positions refer to NCHW tensors. Blocks are contiguous flattened
groups, not a particular hardware layout; incomplete tails are excluded from blocks.
Activation rows overlap across boundaries and must not be summed as unique model data.
The instrumentation recomputes attention intermediates for observation and uses
tiled convolution masks. Timings include this overhead and are NOT inference benchmarks.
Potential savings are upper bounds on zero-product arithmetic, not latency/energy savings.
Dense kernels may already avoid padding; zero skipping needs implementation support.
Reordering sparse reductions may affect floating-point rounding. No accuracy is measured.
Image labels are unnecessary. A few personal images are a smoke test, not representative evidence.
