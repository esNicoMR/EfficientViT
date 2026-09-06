# EfficientViT-B1 CPU Statistics

Processed images: 2 / 2; status: completed

MACs: 1,038,429,184

Potential zero-product skips: 21,243,064 (2.046%)

Padding-only products: 6,159,264

Potential skips excluding padding: 15,083,800

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
