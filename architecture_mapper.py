import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from efficientvit.cls_model_zoo import create_efficientvit_cls_model


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

    if "attention" in cls or "attn" in cls:
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
    Approximate MAC count for Conv2d and Linear leaf modules.
    Other/custom modules return None; their internal Conv/Linear layers
    are still counted separately by their own hooks.
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
        return is_leaf(module)

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
        help="Only report leaf modules.",
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
        writer.writerows(records)

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
            f"Approx. MACs counted from Conv2d/Linear leaves: "
            f"{total_macs} ({human_num(total_macs)})\n"
        )

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
            f"- Approx. MACs counted from Conv2d/Linear leaf modules: "
            f"**{total_macs:,} ({human_num(total_macs)})**\n\n"
        )
        f.write(
            "> MAC count is an approximation. Custom attention operations may contain "
            "tensor/matrix operations that are not represented as nn.Conv2d or nn.Linear. "
            "Their internal Conv/Linear modules are counted where applicable.\n\n"
        )

        f.write("## Executed Module Map\n\n")
        f.write(
            "| # | Module | Category | Input | Output | Direct params | Approx. MACs |\n"
        )
        f.write(
            "|---:|---|---|---|---|---:|---:|\n"
        )

        for i, r in enumerate(records, start=1):
            macs = "" if r["approx_macs"] is None else human_num(r["approx_macs"])
            f.write(
                f"| {i} | `{r['name']}` | {r['category']} | "
                f"`{r['input_shape']}` | `{r['output_shape']}` | "
                f"{r['direct_params']:,} | {macs} |\n"
            )

        f.write("\n## Operation Categories\n\n")
        category_counts = {}
        category_macs = {}
        for r in records:
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

    print("\nDone.")
    print(f"Model parameters: {total_params:,} ({human_num(total_params)})")
    print(f"Model output:     {list(y.shape)}")
    print(f"Approx. MACs:     {total_macs:,} ({human_num(total_macs)})")
    print("\nGenerated:")
    print(f"  {csv_path}")
    print(f"  {txt_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
