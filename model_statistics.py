"""CPU, exact-zero statistics for EfficientViT-B1. No dataset download required.

Examples (run in the efficientvit Conda environment):
  python model_statistics.py --num-images 2
  python model_statistics.py --image-dir "D:/datasets/imagenette2-320/val" --recursive --num-images 25

Default input: image files beside this script (a smoke test, not a dataset).
For representative measurements use a diverse dataset, e.g. Imagenette validation.
All MAC opportunities are analytical counts, NOT measured speedups.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.models.nn.ops import LiteMLA, ResidualBlock

DEFAULT_NUM_IMAGES = 25
ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def percent(a, b):
    return 100.0 * a / b if b else 0.0


def operation_table(rows):
    """Weighted operation-family percentages from summed per-layer counts."""
    groups = defaultdict(lambda: dict(macs=0, skippable_macs=0, padding_macs=0))
    for row in rows:
        for key in ('macs', 'skippable_macs', 'padding_macs'):
            groups[row['operation']][key] += int(row[key])
    total = sum(group['macs'] for group in groups.values())
    labels = {'conv': 'Convolutions', 'linear': 'Linear layers',
              'Vpad_Kt': 'Attention: `Vpad × Kᵀ`', 'VK_Q': 'Attention: `VK × Q`',
              'Kt_Q': 'Attention: `Kᵀ × Q`', 'V_attention': 'Attention: `V × attention`'}
    lines = ['## MAC Opportunities by Operation', '',
             '| Operation | MACs | Share of model MACs | Potential skips within operation | Skips excluding padding within operation |',
             '|---|---:|---:|---:|---:|']
    for operation in list(labels) + sorted(set(groups) - set(labels)):
        if operation not in groups:
            continue
        counts = groups[operation]
        macs, skips, padding = (counts[key] for key in ('macs', 'skippable_macs', 'padding_macs'))
        lines.append(f"| {labels.get(operation, operation)} | {macs:,} | {percent(macs, total):.2f}% | "
                     f"{percent(skips, macs):.2f}% | {percent(skips - padding, macs):.2f}% |")
    skips = sum(group['skippable_macs'] for group in groups.values())
    padding = sum(group['padding_macs'] for group in groups.values())
    lines.append(f"| **Total** | **{total:,}** | **{percent(total, total):.2f}%** | "
                 f"**{percent(skips, total):.2f}%** | **{percent(skips - padding, total):.2f}%** |")
    lines.extend(['', 'Source: `macs_by_layer.csv`, grouped by `operation` across all processed images. '
                  'Share = operation MACs / model MACs × 100. Potential skips = operation skippable MACs / '
                  'operation MACs × 100. The final column subtracts padding MACs before division. '
                  'These are ratios of summed counts, not averages of layer percentages; '
                  'the Total row is weighted by MAC count. Percentages are rounded.', ''])
    return '\n'.join(lines) + '\n'


def tensor_stats(x, block_size=16):
    """Structured zeros use NCHW channels/positions and contiguous flattened blocks."""
    if not torch.isfinite(x).all():
        raise ValueError('Non-finite tensor encountered; zero-product estimates require finite operands.')
    zero = x == 0
    flat = zero.reshape(-1)
    blocks = flat.numel() // block_size
    out = dict(elements=x.numel(), zeros=int(zero.sum()),
               blocks=blocks, zero_blocks=int(flat[:blocks * block_size].reshape(-1, block_size).all(1).sum()),
               block_tail_elements=x.numel() % block_size)
    out['zero_percent'] = percent(out['zeros'], out['elements'])
    if x.ndim == 4:
        out.update(channels=x.shape[0] * x.shape[1],
                   zero_channels=int(zero.flatten(2).all(2).sum()),
                   positions=x.shape[0] * x.shape[2] * x.shape[3],
                   zero_positions=int(zero.all(1).sum()))
    return out


def product_counts(a, b):
    """Exact counts for (..., M, K) @ (..., K, N), including broadcasting.

    Count nonzero pairs by contracting-axis marginals, avoiding M*K*N tensors.
    'left' and 'right' refer to matrix operand positions, not weights/activations.
    """
    if a.shape[-1] != b.shape[-2]:
        raise ValueError('Incompatible matrix product')
    shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    az = (a != 0).sum(-2, dtype=torch.int64).expand(*shape, a.shape[-1])
    bz = (b != 0).sum(-1, dtype=torch.int64).expand(*shape, b.shape[-2])
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    batches = az.numel() // k
    total = batches * m * k * n
    left_zero = total - int(az.sum()) * n
    right_zero = total - int(bz.sum()) * m
    active = int((az * bz).sum())
    skip = total - active
    return dict(macs=total, left_zero_macs=left_zero, right_zero_macs=right_zero,
                both_zero_macs=left_zero + right_zero - skip,
                skippable_macs=skip, nonzero_product_macs=active, padding_macs=0)


def conv_counts(module, x, y, tile_rows=4):
    """Exact grouped-convolution counts; unfold only a few output rows at once.

    Left = weight, right = activation. Padding-only products are separated.
    Supports numeric symmetric zero padding used by this EfficientViT model.
    """
    if module.padding_mode != 'zeros' or isinstance(module.padding, str):
        raise ValueError('Convolution statistics require numeric zero padding.')
    kh, kw = module.kernel_size
    sh, sw = module.stride
    dh, dw = module.dilation
    ph, pw = module.padding
    groups = module.groups
    cin_g, cout_g = module.in_channels // groups, module.out_channels // groups
    k = cin_g * kh * kw
    weight = (module.weight != 0).reshape(groups, cout_g, k)
    weight_nz = weight.sum(1, dtype=torch.int64)
    oh, ow = y.shape[-2:]
    total = x.shape[0] * module.out_channels * oh * ow * k
    counts = dict(macs=total, left_zero_macs=total - int(weight_nz.sum()) * x.shape[0] * oh * ow,
                  right_zero_macs=0, both_zero_macs=0, skippable_macs=0,
                  nonzero_product_macs=0, padding_macs=0)
    for sample in x:
        padded = F.pad(sample.unsqueeze(0), (pw, pw, ph, ph))
        valid = F.pad(torch.ones(1, 1, *sample.shape[-2:]), (pw, pw, ph, ph))
        for row in range(0, oh, tile_rows):
            rows = min(tile_rows, oh - row)
            begin = row * sh
            end = begin + (rows - 1) * sh + dh * (kh - 1) + 1
            patches = F.unfold(padded[:, :, begin:end], (kh, kw), dilation=(dh, dw), stride=(sh, sw))
            nz = (patches != 0).reshape(groups, k, -1).sum(2, dtype=torch.int64)
            counts['nonzero_product_macs'] += int((nz * weight_nz).sum())
            counts['right_zero_macs'] += groups * k * rows * ow * cout_g - int(nz.sum()) * cout_g
            valid_patches = F.unfold(valid[:, :, begin:end], (kh, kw), dilation=(dh, dw), stride=(sh, sw))
            counts['padding_macs'] += int((valid_patches == 0).sum()) * cin_g * module.out_channels
    counts['skippable_macs'] = total - counts['nonzero_product_macs']
    counts['both_zero_macs'] = counts['left_zero_macs'] + counts['right_zero_macs'] - counts['skippable_macs']
    return counts


class Collector:
    def __init__(self, model, tile_rows, block_size):
        self.model, self.tile_rows, self.block_size = model, tile_rows, block_size
        self.handles, self.methods = [], []
        self.tensor_rows, self.mac_rows = [], []
        for name, module in model.named_modules():
            if not list(module.children()) or isinstance(module, (LiteMLA, ResidualBlock)):
                self.handles.append(module.register_forward_hook(self.hook(name)))
            if isinstance(module, LiteMLA):
                for method in ('relu_linear_att', 'relu_quadratic_att'):
                    original = getattr(module, method)
                    self.methods.append((module, method, original))
                    setattr(module, method, self.attention(name, module, method, original))

    def record_tensor(self, name, role, x):
        if isinstance(x, torch.Tensor):
            stats = tensor_stats(x, self.block_size)
            if role not in ('input', 'output'):
                # Attention internals are B x heads x matrix rows x columns, not NCHW.
                for key in ('channels', 'zero_channels', 'positions', 'zero_positions'):
                    stats.pop(key, None)
            self.tensor_rows.append(dict(module=name, role=role, shape=str(list(x.shape)), **stats))

    def hook(self, name):
        def hook(module, inputs, output):
            self.record_tensor(name, 'input', inputs[0])
            self.record_tensor(name, 'output', output)
            if isinstance(module, torch.nn.Conv2d):
                counts = conv_counts(module, inputs[0], output, self.tile_rows)
                self.mac_rows.append(dict(module=name, operation='conv', **counts))
            elif isinstance(module, torch.nn.Linear):
                counts = product_counts(inputs[0].reshape(-1, module.in_features), module.weight.T)
                self.mac_rows.append(dict(module=name, operation='linear', **counts))
        return hook

    def attention(self, name, module, method, original):
        def wrapper(qkv):
            # Let the unmodified method determine the actual model output.
            output = original(qkv)
            b, _, h, w = qkv.shape
            packed = qkv.reshape(b, -1, 3 * module.dim, h * w)
            q, k, v = packed.split(module.dim, dim=2)
            q, k = F.relu(q), F.relu(k)
            for role, tensor in [('Q_after_ReLU', q), ('K_after_ReLU', k), ('V', v)]:
                self.record_tensor(name, role, tensor)
            if method == 'relu_linear_att':
                v = F.pad(v, (0, 0, 0, 1), value=1)
                kt = k.transpose(-1, -2)
                intermediate = torch.matmul(v, kt)
                self.record_tensor(name, 'VK_padded', intermediate)
                products = [('Vpad_Kt', v, kt), ('VK_Q', intermediate, q)]
            else:
                kt = k.transpose(-1, -2)
                attention = torch.matmul(kt, q)
                attention = attention / (attention.sum(2, keepdim=True) + module.eps)
                self.record_tensor(name, 'normalized_attention', attention)
                products = [('Kt_Q', kt, q), ('V_attention', v, attention)]
            for label, left, right in products:
                self.mac_rows.append(dict(module=name, operation=label, **product_counts(left, right)))
            return output
        return wrapper

    def reset(self):
        self.tensor_rows.clear()
        self.mac_rows.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, method, original in self.methods:
            setattr(module, method, original)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


NOTES = """Counts use exact zeros, FP32 CPU, batch size 1, eval and inference mode.
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
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-images', type=int, default=DEFAULT_NUM_IMAGES, help='Maximum successfully processed images (default: 25).')
    parser.add_argument('--image-dir', type=Path, default=ROOT)
    parser.add_argument('--recursive', action='store_true', help='Search dataset class subfolders.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Default: model_statistics/run_TIMESTAMP beside script.')
    parser.add_argument('--weights', type=Path, default=ROOT / 'assets/checkpoints/efficientvit_cls/efficientvit_b1_r224.pt')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tile-rows', type=int, default=4, help='Output rows per convolution counting tile.')
    parser.add_argument('--block-size', type=int, default=16)
    args = parser.parse_args()
    if min(args.num_images, args.threads, args.tile_rows, args.block_size) < 1:
        parser.error('Image count, threads, tile rows and block size must be positive.')
    if not args.image_dir.is_dir():
        parser.error(f'Image directory does not exist: {args.image_dir}')
    files = sorted(p.resolve() for p in (args.image_dir.rglob('*') if args.recursive else args.image_dir.iterdir())
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        parser.error('No images found. Use --image-dir PATH --recursive for an extracted dataset.')
    if not args.weights.is_file():
        parser.error(f'Pretrained weights not found: {args.weights}. Supply --weights PATH; no random-weight fallback.')
    random.Random(args.seed).shuffle(files)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = args.output_dir or ROOT / 'model_statistics' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    print(f'Found {len(files)} images; will process up to {args.num_images}. CPU threads={args.threads}', flush=True)
    if len(files) < args.num_images:
        print('Fewer images available than requested; each available image is used at most once.', flush=True)
    model = create_efficientvit_cls_model('efficientvit-b1-r224', pretrained=True, weight_url=str(args.weights.resolve())).cpu().float().eval()
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    weights = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            stats = tensor_stats(module.weight, args.block_size)
            for key in ('channels', 'zero_channels', 'positions', 'zero_positions'):
                stats.pop(key, None)
            stats.update(output_filters=module.weight.shape[0],
                         zero_output_filters=int((module.weight == 0).flatten(1).all(1).sum()))
            weights.append(dict(module=name, **stats))
    write_csv(out / 'weights.csv', weights)
    metadata = dict(model='efficientvit-b1-r224', requested_images=args.num_images, available_images=len(files),
                    seed=args.seed, threads=args.threads, dtype='float32', device='cpu', torch_version=torch.__version__,
                    weights=str(args.weights.resolve()), weights_sha256=digest(args.weights), script_sha256=digest(Path(__file__)),
                    preprocessing='EXIF transpose, RGB, resize shortest side 256 bilinear, center crop 224, ImageNet mean/std',
                    block_size=args.block_size, tile_rows=args.tile_rows, notes=NOTES)
    images, failures = [], []
    aggregate = defaultdict(lambda: defaultdict(int))
    tensor_aggregate = defaultdict(lambda: defaultdict(int))
    collector = Collector(model, args.tile_rows, args.block_size)
    status = 'completed'
    def save():
        rows = []
        for (name, operation), counts in aggregate.items():
            rows.append(dict(module=name, operation=operation, **counts,
                             skippable_percent=percent(counts['skippable_macs'], counts['macs']),
                             skippable_excluding_padding=counts['skippable_macs'] - counts['padding_macs']))
        write_csv(out / 'macs_by_layer.csv', rows)
        tensor_summary = []
        for (name, role), counts in tensor_aggregate.items():
            tensor_summary.append(dict(module=name, role=role, **counts,
                                       zero_percent=percent(counts['zeros'], counts['elements']),
                                       zero_block_percent=percent(counts['zero_blocks'], counts['blocks'])))
        write_csv(out / 'tensors_by_layer.csv', tensor_summary)
        write_csv(out / 'images.csv', images)
        totals = {key:sum(row[key] for row in rows) for key in
                  ['macs','skippable_macs','padding_macs','left_zero_macs','right_zero_macs','both_zero_macs','nonzero_product_macs']}
        summary = dict(metadata, status=status, processed_images=len(images), failures=failures, totals=totals,
                       skippable_percent=percent(totals['skippable_macs'], totals['macs']))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (out / 'report.md').write_text(
            '# EfficientViT-B1 CPU Statistics\n\n'
            f"Processed images: {len(images)} / {args.num_images}; status: {status}\n\n"
            f"MACs: {totals['macs']:,}\n\nPotential zero-product skips: {totals['skippable_macs']:,} "
            f"({summary['skippable_percent']:.3f}%)\n\nPadding-only products: {totals['padding_macs']:,}\n\n"
            f"Potential skips excluding padding: {totals['skippable_macs']-totals['padding_macs']:,}\n\n"
            + operation_table(rows) +
            'See macs_by_layer.csv for weighted layer totals, images.csv for image variation, '
            'tensors_by_layer.csv for aggregate sparsity, tensors.jsonl for per-image tensor/structured '
            'sparsity, macs.jsonl for per-image layer MAC counts, and weights.csv for weights.\n\n' + NOTES,
            encoding='utf-8')
    try:
        with (out / 'tensors.jsonl').open('w', encoding='utf-8') as tensor_file, (out / 'macs.jsonl').open('w', encoding='utf-8') as mac_file:
            for path in files:
                if len(images) >= args.num_images:
                    break
                try:
                    with Image.open(path) as image:
                        x = preprocess(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
                except (OSError, ValueError) as error:
                    failures.append(dict(path=str(path), error=str(error)))
                    continue
                collector.reset()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model(x)
                if not torch.isfinite(output).all():
                    raise ValueError('Non-finite logits')
                for row in collector.mac_rows:
                    for key,value in row.items():
                        if isinstance(value,int):
                            aggregate[(row['module'],row['operation'])][key] += value
                    mac_file.write(json.dumps(dict(image=str(path), **row)) + '\n')
                for row in collector.tensor_rows:
                    tensor_file.write(json.dumps(dict(image=str(path), **row)) + '\n')
                    for key, value in row.items():
                        if isinstance(value, int):
                            tensor_aggregate[(row['module'], row['role'])][key] += value
                mac_file.flush()
                tensor_file.flush()
                total = sum(r['macs'] for r in collector.mac_rows)
                skip = sum(r['skippable_macs'] for r in collector.mac_rows)
                elapsed = time.perf_counter() - started
                images.append(dict(path=str(path), sha256=digest(path), instrumented_seconds=elapsed,
                                   predicted_class=int(output.argmax()), macs=total, skippable_macs=skip,
                                   skippable_percent=percent(skip,total)))
                status = 'running'
                save()
                print(f"[{len(images)}/{min(args.num_images,len(files))}] {path.name}: {elapsed:.2f}s; potential skips {percent(skip,total):.2f}%", flush=True)
        status = 'completed' if images else 'no_valid_images'
    except KeyboardInterrupt:
        status = 'interrupted'
        print('Interrupted; preserving completed images.', flush=True)
    except Exception:
        status = 'failed'
        raise
    finally:
        collector.close()
        save()
    print(f'Reports saved to {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
