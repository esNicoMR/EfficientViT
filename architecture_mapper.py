import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from efficientvit.cls_model_zoo import create_efficientvit_cls_model
from efficientvit.models.nn.ops import LiteMLA, ResidualBlock


def estimate_non_mac_ops(module, inputs, output):
    """Logical, unfused scalar operations; reductions are a subset of additions.

    Counts are algorithmic estimates, not CPU/GPU instruction or latency counts.
    None marks unsupported leaf arithmetic; composite children count separately.
    """
    x, y = first_tensor(inputs), first_tensor(output)
    if x is None or y is None:
        return None
    n = y.numel()
    ops = {}

    def put(**counts):
        ops.update({key: int(value) for key, value in counts.items() if value})

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        put(add=n if module.bias is not None else 0)
    elif isinstance(module, ResidualBlock):
        put(add=n if module.main is not None and module.shortcut is not None else 0)
    elif isinstance(module, LiteMLA):
        b, _, h, w = x.shape
        tokens, dim = h * w, module.dim
        heads = module.qkv.conv.out_channels // (3 * dim) * (1 + len(module.aggreg))
        rows = b * heads * tokens
        if tokens > dim:
            put(add=rows, div=rows * dim)
        else:
            reduction = rows * (tokens - 1)
            put(add=reduction + rows, div=rows * tokens, reduction_add=reduction)
        # Q/K ReLUs are counted by their own (twice-executed) child hook.
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.training or not module.track_running_stats:
            return None
        channels = x.shape[1]
        put(add=n + channels + (n if module.affine else 0),
            mul=n if module.affine else 0, div=n, sqrt=channels)
    elif isinstance(module, nn.LayerNorm):
        width = math.prod(module.normalized_shape)
        groups = n // width
        reduction = 2 * groups * (width - 1)
        put(add=reduction + n + groups + (n if module.bias is not None else 0),
            mul=n + (n if module.weight is not None else 0),
            div=2 * groups + n, sqrt=groups, reduction_add=reduction)
    elif isinstance(module, nn.ReLU6):
        put(compare=2 * n)
    elif isinstance(module, nn.ReLU):
        put(compare=n)
    elif isinstance(module, nn.Hardswish):
        put(add=n, compare=2 * n, mul=n, div=n)
    elif isinstance(module, nn.SiLU):
        put(neg=n, exp=n, add=n, div=n, mul=n)
    elif isinstance(module, nn.GELU):
        if module.approximate == 'tanh':
            put(mul=6 * n, add=2 * n, tanh=n)
        else:
            put(mul=2 * n, div=n, erf=n, add=n)
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        ih, iw = x.shape[-2:]
        oh, ow = y.shape[-2:]
        # PyTorch adaptive bins may overlap for non-divisible resolutions.
        hs = sum(((i + 1) * ih + oh - 1) // oh - i * ih // oh for i in range(oh))
        ws = sum(((i + 1) * iw + ow - 1) // ow - i * iw // ow for i in range(ow))
        reduction = (y.numel() // (oh * ow)) * (hs * ws - oh * ow)
        put(add=reduction, div=n, reduction_add=reduction)
    elif not is_leaf(module) or isinstance(module, (nn.Identity, nn.Dropout)) or module.__class__.__name__ == 'IdentityLayer':
        pass
    else:
        return None
    return ops


def non_mac_total(ops):
    return sum(value for key, value in (ops or {}).items() if key != 'reduction_add')


NON_MAC_NOTES = (
    'Non-MAC counts are logical unfused scalar operations for evaluation, over the entire batch. '
    'Subtractions count as additions; comparisons and special functions each count as one operation, '
    'not one FLOP or equal hardware cost. Reduction additions are already included in additions. '
    'Biases, residual sums, LiteMLA denominator arithmetic, supported activations, eval BatchNorm, '
    'LayerNorm and adaptive average pooling are included. BatchNorm uses subtract/divide/affine '
    'arithmetic plus per-channel epsilon and square root; LayerNorm uses a two-pass mean/variance '
    'estimate. Actual kernels may fuse operations or use reciprocals. BatchNorm may fold into '
    'convolutions; these counts do not assert that all operations can be eliminated. Reshapes, '
    'concatenation, padding, casts, memory traffic and indexing are excluded. Unsupported leaf '
    'modules are listed explicitly; unrecognized functional arithmetic inside composites is not traced. '
    '--leaf-only retains LiteMLA and residual rows to preserve totals.'
)


def human_num(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}G"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.3f}K"
    return str(n)


def shape_str(x):
    if isinstance(x, torch.Tensor):
        return str(list(x.shape))
    if isinstance(x, (list, tuple)):
        parts = []
        for item in x:
            if isinstance(item, torch.Tensor):
                parts.append(str(list(item.shape)))
            else:
                parts.append(type(item).__name__)
        return "[" + ", ".join(parts) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}:{shape_str(v)}" for k, v in x.items()) + "}"
    return type(x).__name__


def first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for item in x.values():
            t = first_tensor(item)
            if t is not None:
                return t
    return None


def classify_module(module: nn.Module) -> str:
    cls = module.__class__.__name__.lower()

    if isinstance(module, nn.Conv2d):
        if module.groups == module.in_channels == module.out_channels:
            return "Depthwise Conv2d"
        if module.kernel_size == (1, 1):
            return "Pointwise Conv2d"
        return "Conv2d"

    if isinstance(module, nn.Linear):
        return "Linear"
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        return "Normalization"
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Hardswish)):
        return "Activation"
    if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        return "Pooling"

    if isinstance(module, LiteMLA) or "attention" in cls or "attn" in cls:
        return "Attention block"
    if "mbconv" in cls:
        return "MBConv block"
    if "ffn" in cls or "feedforward" in cls:
        return "FFN block"
    if "residual" in cls or "resblock" in cls:
        return "Residual block"

    return module.__class__.__name__


def estimate_macs(module: nn.Module, inputs, output):
    """
    Count Conv2d/Linear MACs and the two LiteMLA attention matrix products.
    LiteMLA returns only its matrix-product MACs; its convolution children
    are counted separately. One multiply-accumulate is one MAC.
    """
    out = first_tensor(output)
    inp = first_tensor(inputs)
    if out is None or inp is None:
        return None

    if isinstance(module, nn.Conv2d):
        # Output shape normally [N, Cout, Hout, Wout]
        if out.ndim != 4:
            return None
        batch, cout, hout, wout = out.shape
        kh, kw = module.kernel_size
        cin_per_group = module.in_channels // module.groups
        macs_per_output = cin_per_group * kh * kw
        return int(batch * cout * hout * wout * macs_per_output)

    if isinstance(module, LiteMLA):
        batch, _, height, width = inp.shape
        tokens = height * width
        dim = module.dim
        heads = module.qkv.conv.out_channels // (3 * dim)
        scale_heads = heads * (1 + len(module.aggreg))
        if tokens > dim:
            # V is padded to dim + 1 to compute the denominator too:
            # (dim+1, tokens) @ (tokens, dim), then (dim+1, dim) @ (dim, tokens).
            return int(2 * batch * scale_heads * tokens * dim * (dim + 1))
        # (tokens, dim) @ (dim, tokens), then (dim, tokens) @ (tokens, tokens).
        return int(2 * batch * scale_heads * tokens * tokens * dim)

    if isinstance(module, nn.Linear):
        # Every output element computes in_features MACs.
        return int(out.numel() * module.in_features)

    return None


def is_leaf(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def should_record(name: str, module: nn.Module, leaf_only: bool) -> bool:
    if name == "":
        return False

    if leaf_only:
        return is_leaf(module) or isinstance(module, (LiteMLA, ResidualBlock))

    # Keep all computationally meaningful blocks + leaf modules.
    category = classify_module(module)
    important = {
        "Conv2d",
        "Depthwise Conv2d",
        "Pointwise Conv2d",
        "Linear",
        "Normalization",
        "Activation",
        "Pooling",
        "Attention block",
        "MBConv block",
        "FFN block",
        "Residual block",
    }
    return is_leaf(module) or category in important


def main():
    parser = argparse.ArgumentParser(
        description="Map EfficientViT-B1 architecture, tensor shapes, parameters, and approximate MACs."
    )
    parser.add_argument(
        "--model",
        default="efficientvit-b1-r224",
        help="EfficientViT classification model name.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Square input resolution.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Build architecture without loading pretrained weights.",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Report leaves plus LiteMLA and residual rows to retain operation totals.",
    )
    parser.add_argument(
        "--output-dir",
        default="architecture_report",
        help="Directory for generated reports.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} ...")
    model = create_efficientvit_cls_model(
        name=args.model,
        pretrained=not args.no_pretrained,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    records = []
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            direct_params = sum(p.numel() for p in module.parameters(recurse=False))
            direct_trainable = sum(
                p.numel() for p in module.parameters(recurse=False) if p.requires_grad
            )

            rec = {
                "name": name,
                "class": module.__class__.__name__,
                "category": classify_module(module),
                "input_shape": shape_str(inputs),
                "output_shape": shape_str(output),
                "direct_params": direct_params,
                "direct_trainable_params": direct_trainable,
                "approx_macs": estimate_macs(module, inputs, output),
            }
            non_mac = estimate_non_mac_ops(module, inputs, output)
            rec["approx_non_mac_ops"] = non_mac_total(non_mac) if non_mac is not None else None
            rec["non_mac_breakdown"] = json.dumps(non_mac, sort_keys=True)

            if isinstance(module, nn.Conv2d):
                rec["details"] = (
                    f"in={module.in_channels}, out={module.out_channels}, "
                    f"k={module.kernel_size}, stride={module.stride}, "
                    f"groups={module.groups}, bias={module.bias is not None}"
                )
            elif isinstance(module, nn.Linear):
                rec["details"] = (
                    f"in={module.in_features}, out={module.out_features}, "
                    f"bias={module.bias is not None}"
                )
            elif isinstance(module, LiteMLA):
                tokens = first_tensor(inputs).shape[-2] * first_tensor(inputs).shape[-1]
                branch = "linear" if tokens > module.dim else "quadratic"
                rec["details"] = (
                    f"{branch} attention matmuls only; tokens={tokens}, dim={module.dim}, "
                    f"scales={1 + len(module.aggreg)}; convolution MACs counted in child rows"
                )
            else:
                rec["details"] = ""

            records.append(rec)

        return hook

    for name, module in model.named_modules():
        if should_record(name, module, args.leaf_only):
            hooks.append(module.register_forward_hook(make_hook(name)))

    x = torch.randn(
        args.batch_size,
        3,
        args.resolution,
        args.resolution,
    )

    print(f"Running dummy input {list(x.shape)} ...")
    with torch.no_grad():
        y = model(x)

    for h in hooks:
        h.remove()

    total_macs = sum(r["approx_macs"] or 0 for r in records)
    attention_macs = sum(
        r["approx_macs"] or 0 for r in records if r["category"] == "Attention block"
    )
    conv_linear_macs = total_macs - attention_macs
    total_non_mac = sum(r["approx_non_mac_ops"] or 0 for r in records)
    non_mac_totals = {}
    for r in records:
        for kind, count in (json.loads(r["non_mac_breakdown"]) or {}).items():
            non_mac_totals[kind] = non_mac_totals.get(kind, 0) + count
    unsupported = sorted({r["class"] for r in records if r["approx_non_mac_ops"] is None})

    # Preserve the original MAC-only leaf view while retaining residual arithmetic separately.
    nonleaf_residuals = {
        name for name, module in model.named_modules()
        if isinstance(module, ResidualBlock) and not is_leaf(module)
    }
    mac_records = [r for r in records if not args.leaf_only or r["name"] not in nonleaf_residuals]

    # CSV report
    csv_path = output_dir / "architecture_map.csv"
    fieldnames = [
        "name",
        "class",
        "category",
        "input_shape",
        "output_shape",
        "direct_params",
        "direct_trainable_params",
        "approx_macs",
        "details",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fieldnames} for r in mac_records)

    non_mac_csv_path = output_dir / "non_mac_operations.csv"
    with non_mac_csv_path.open("w", newline="", encoding="utf-8") as f:
        non_mac_fields = ["name", "class", "category", "approx_non_mac_ops", "non_mac_breakdown"]
        writer = csv.DictWriter(f, fieldnames=non_mac_fields)
        writer.writeheader()
        writer.writerows({key: r[key] for key in non_mac_fields} for r in records)

    # Text hierarchy from model.__str__()
    txt_path = output_dir / "model_structure.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(str(model))
        f.write("\n\n")
        f.write(f"Total parameters: {total_params} ({human_num(total_params)})\n")
        f.write(
            f"Trainable parameters: {trainable_params} "
            f"({human_num(trainable_params)})\n"
        )
        f.write(f"Output shape: {list(y.shape)}\n")
        f.write(
            f"Approx. MACs (Conv2d/Linear + LiteMLA matmuls): "
            f"{total_macs} ({human_num(total_macs)})\n"
        )
        f.write("\nNon-MAC Operations\n")
        f.write(f"Approx. non-MAC scalar operations: {total_non_mac:,}\n")
        f.write(f"Non-MAC breakdown: {json.dumps(non_mac_totals, sort_keys=True)}\n")
        f.write(NON_MAC_NOTES + "\n")
        f.write(f"Unsupported non-MAC leaf types: {', '.join(unsupported) or 'none'}\n")

    # Markdown report
    md_path = output_dir / "architecture_report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Architecture Report — {args.model}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Input shape: `{list(x.shape)}`\n")
        f.write(f"- Output shape: `{list(y.shape)}`\n")
        f.write(f"- Total parameters: **{total_params:,} ({human_num(total_params)})**\n")
        f.write(
            f"- Trainable parameters: **{trainable_params:,} "
            f"({human_num(trainable_params)})**\n"
        )
        f.write(
            f"- Approx. MACs (Conv2d/Linear + LiteMLA matmuls): "
            f"**{total_macs:,} ({human_num(total_macs)})**\n\n"
        )
        f.write(
            f"- Conv2d/Linear MACs: **{conv_linear_macs:,}**\n"
            f"- LiteMLA matrix-product MACs: **{attention_macs:,}**\n\n"
            '> One multiply-accumulate is one MAC (approximately two FLOPs). Counts cover the entire input batch. LiteMLA rows count only the two attention matrix products, including the padded denominator channel in the linear branch; their convolution children are counted separately. The executed linear/quadratic branch and all scales are included. Bias additions, normalization, activations, pooling, residual additions, elementwise attention normalization, and data movement are excluded. Other custom tensor operations are not counted. --leaf-only retains LiteMLA rows for accounting.\n\n'
        )
        f.write("## Executed Module Map\n\n")
        f.write(
            "| # | Module | Category | Input | Output | Direct params | Approx. MACs |\n"
        )
        f.write(
            "|---:|---|---|---|---|---:|---:|\n"
        )

        for i, r in enumerate(mac_records, start=1):
            macs = "" if r["approx_macs"] is None else human_num(r["approx_macs"])
            f.write(
                f"| {i} | `{r['name']}` | {r['category']} | "
                f"`{r['input_shape']}` | `{r['output_shape']}` | "
                f"{r['direct_params']:,} | {macs} |\n"
            )

        f.write("\n## Operation Categories\n\n")
        category_counts = {}
        category_macs = {}
        category_non_macs = {}
        for r in mac_records:
            category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1
            category_macs[r["category"]] = category_macs.get(r["category"], 0) + (
                r["approx_macs"] or 0
            )

        f.write("| Category | Executed modules | Approx. MACs |\n")
        f.write("|---|---:|---:|\n")
        for cat in sorted(category_counts):
            f.write(
                f"| {cat} | {category_counts[cat]} | "
                f"{human_num(category_macs[cat]) if category_macs[cat] else ''} |\n"
            )

        for r in records:
            category_non_macs[r["category"]] = category_non_macs.get(r["category"], 0) + (r["approx_non_mac_ops"] or 0)
        f.write("\n## Non-MAC Operations\n\n")
        f.write(f"- Approx. non-MAC scalar operations: **{total_non_mac:,} ({human_num(total_non_mac)})**\n\n")
        f.write("### Operation Breakdown\n\n")
        f.write("| Operation | Count |\n|---|---:|\n")
        for kind, count in sorted(non_mac_totals.items()):
            label = "reduction additions (subset of add; do not add again)" if kind == "reduction_add" else kind
            f.write(f"| {label} | {count:,} |\n")
        f.write("\n" + NON_MAC_NOTES + "\n\n")
        f.write(f"Unsupported non-MAC leaf types: **{', '.join(unsupported) or 'none'}**.\n\n")

        f.write("### Non-MAC Operations by Category\n\n")
        f.write("| Category | Non-MAC ops |\n|---|---:|\n")
        for cat, count in sorted(category_non_macs.items()):
            f.write(f"| {cat} | {count:,} |\n")
        f.write("\n### Non-MAC Operations by Module\n\n")
        f.write("| Module | Non-MAC ops | Breakdown |\n|---|---:|---|\n")
        for r in records:
            if r["approx_non_mac_ops"] == 0:
                continue
            count = r["approx_non_mac_ops"]
            f.write(f"| `{r['name']}` | {count if count is not None else 'unsupported'} | `{r['non_mac_breakdown']}` |\n")

    print("\nDone.")
    print(f"Model parameters: {total_params:,} ({human_num(total_params)})")
    print(f"Model output:     {list(y.shape)}")
    print(f"Approx. MACs:     {total_macs:,} ({human_num(total_macs)})")
    print(f"Non-MAC ops:     {total_non_mac:,} ({human_num(total_non_mac)})")
    print(f"Unsupported non-MAC leaf types: {', '.join(unsupported) or 'none'}")
    print("\nGenerated:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {md_path}")
    print(f"  {non_mac_csv_path}")


if __name__ == "__main__":
    main()
