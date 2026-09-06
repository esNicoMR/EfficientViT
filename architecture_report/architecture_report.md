# Architecture Report — efficientvit-b1-r224

## Summary

- Input shape: `[1, 3, 224, 224]`
- Output shape: `[1, 1000]`
- Total parameters: **9,102,024 (9.102M)**
- Trainable parameters: **9,102,024 (9.102M)**
- Approx. MACs (Conv2d/Linear + LiteMLA matmuls): **519,214,592 (519.215M)**

- Conv2d/Linear MACs: **510,684,672**
- LiteMLA matrix-product MACs: **8,529,920**

> One multiply-accumulate is one MAC (approximately two FLOPs). Counts cover the entire input batch. LiteMLA rows count only the two attention matrix products, including the padded denominator channel in the linear branch; their convolution children are counted separately. The executed linear/quadratic branch and all scales are included. Bias additions, normalization, activations, pooling, residual additions, elementwise attention normalization, and data movement are excluded. Other custom tensor operations are not counted. --leaf-only retains LiteMLA rows for accounting.

## Executed Module Map

| # | Module | Category | Input | Output | Direct params | Approx. MACs |
|---:|---|---|---|---|---:|---:|
| 1 | `backbone.input_stem.op_list.0.conv` | Conv2d | `[[1, 3, 224, 224]]` | `[1, 16, 112, 112]` | 432 | 5.419M |
| 2 | `backbone.input_stem.op_list.0.norm` | Normalization | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 32 |  |
| 3 | `backbone.input_stem.op_list.0.act` | Activation | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 0 |  |
| 4 | `backbone.input_stem.op_list.1.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 144 | 1.806M |
| 5 | `backbone.input_stem.op_list.1.main.depth_conv.norm` | Normalization | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 32 |  |
| 6 | `backbone.input_stem.op_list.1.main.depth_conv.act` | Activation | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 0 |  |
| 7 | `backbone.input_stem.op_list.1.main.point_conv.conv` | Pointwise Conv2d | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 256 | 3.211M |
| 8 | `backbone.input_stem.op_list.1.main.point_conv.norm` | Normalization | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 32 |  |
| 9 | `backbone.input_stem.op_list.1.shortcut` | IdentityLayer | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 0 |  |
| 10 | `backbone.input_stem.op_list.1` | Residual block | `[[1, 16, 112, 112]]` | `[1, 16, 112, 112]` | 0 |  |
| 11 | `backbone.stages.0.op_list.0.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 16, 112, 112]]` | `[1, 64, 112, 112]` | 1,024 | 12.845M |
| 12 | `backbone.stages.0.op_list.0.main.inverted_conv.norm` | Normalization | `[[1, 64, 112, 112]]` | `[1, 64, 112, 112]` | 128 |  |
| 13 | `backbone.stages.0.op_list.0.main.inverted_conv.act` | Activation | `[[1, 64, 112, 112]]` | `[1, 64, 112, 112]` | 0 |  |
| 14 | `backbone.stages.0.op_list.0.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 64, 112, 112]]` | `[1, 64, 56, 56]` | 576 | 1.806M |
| 15 | `backbone.stages.0.op_list.0.main.depth_conv.norm` | Normalization | `[[1, 64, 56, 56]]` | `[1, 64, 56, 56]` | 128 |  |
| 16 | `backbone.stages.0.op_list.0.main.depth_conv.act` | Activation | `[[1, 64, 56, 56]]` | `[1, 64, 56, 56]` | 0 |  |
| 17 | `backbone.stages.0.op_list.0.main.point_conv.conv` | Pointwise Conv2d | `[[1, 64, 56, 56]]` | `[1, 32, 56, 56]` | 2,048 | 6.423M |
| 18 | `backbone.stages.0.op_list.0.main.point_conv.norm` | Normalization | `[[1, 32, 56, 56]]` | `[1, 32, 56, 56]` | 64 |  |
| 19 | `backbone.stages.0.op_list.0.main` | MBConv block | `[[1, 16, 112, 112]]` | `[1, 32, 56, 56]` | 0 |  |
| 20 | `backbone.stages.0.op_list.0` | Residual block | `[[1, 16, 112, 112]]` | `[1, 32, 56, 56]` | 0 |  |
| 21 | `backbone.stages.0.op_list.1.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 32, 56, 56]]` | `[1, 128, 56, 56]` | 4,096 | 12.845M |
| 22 | `backbone.stages.0.op_list.1.main.inverted_conv.norm` | Normalization | `[[1, 128, 56, 56]]` | `[1, 128, 56, 56]` | 256 |  |
| 23 | `backbone.stages.0.op_list.1.main.inverted_conv.act` | Activation | `[[1, 128, 56, 56]]` | `[1, 128, 56, 56]` | 0 |  |
| 24 | `backbone.stages.0.op_list.1.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 128, 56, 56]]` | `[1, 128, 56, 56]` | 1,152 | 3.613M |
| 25 | `backbone.stages.0.op_list.1.main.depth_conv.norm` | Normalization | `[[1, 128, 56, 56]]` | `[1, 128, 56, 56]` | 256 |  |
| 26 | `backbone.stages.0.op_list.1.main.depth_conv.act` | Activation | `[[1, 128, 56, 56]]` | `[1, 128, 56, 56]` | 0 |  |
| 27 | `backbone.stages.0.op_list.1.main.point_conv.conv` | Pointwise Conv2d | `[[1, 128, 56, 56]]` | `[1, 32, 56, 56]` | 4,096 | 12.845M |
| 28 | `backbone.stages.0.op_list.1.main.point_conv.norm` | Normalization | `[[1, 32, 56, 56]]` | `[1, 32, 56, 56]` | 64 |  |
| 29 | `backbone.stages.0.op_list.1.main` | MBConv block | `[[1, 32, 56, 56]]` | `[1, 32, 56, 56]` | 0 |  |
| 30 | `backbone.stages.0.op_list.1.shortcut` | IdentityLayer | `[[1, 32, 56, 56]]` | `[1, 32, 56, 56]` | 0 |  |
| 31 | `backbone.stages.0.op_list.1` | Residual block | `[[1, 32, 56, 56]]` | `[1, 32, 56, 56]` | 0 |  |
| 32 | `backbone.stages.1.op_list.0.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 32, 56, 56]]` | `[1, 128, 56, 56]` | 4,096 | 12.845M |
| 33 | `backbone.stages.1.op_list.0.main.inverted_conv.norm` | Normalization | `[[1, 128, 56, 56]]` | `[1, 128, 56, 56]` | 256 |  |
| 34 | `backbone.stages.1.op_list.0.main.inverted_conv.act` | Activation | `[[1, 128, 56, 56]]` | `[1, 128, 56, 56]` | 0 |  |
| 35 | `backbone.stages.1.op_list.0.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 128, 56, 56]]` | `[1, 128, 28, 28]` | 1,152 | 903.168K |
| 36 | `backbone.stages.1.op_list.0.main.depth_conv.norm` | Normalization | `[[1, 128, 28, 28]]` | `[1, 128, 28, 28]` | 256 |  |
| 37 | `backbone.stages.1.op_list.0.main.depth_conv.act` | Activation | `[[1, 128, 28, 28]]` | `[1, 128, 28, 28]` | 0 |  |
| 38 | `backbone.stages.1.op_list.0.main.point_conv.conv` | Pointwise Conv2d | `[[1, 128, 28, 28]]` | `[1, 64, 28, 28]` | 8,192 | 6.423M |
| 39 | `backbone.stages.1.op_list.0.main.point_conv.norm` | Normalization | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 128 |  |
| 40 | `backbone.stages.1.op_list.0.main` | MBConv block | `[[1, 32, 56, 56]]` | `[1, 64, 28, 28]` | 0 |  |
| 41 | `backbone.stages.1.op_list.0` | Residual block | `[[1, 32, 56, 56]]` | `[1, 64, 28, 28]` | 0 |  |
| 42 | `backbone.stages.1.op_list.1.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 64, 28, 28]]` | `[1, 256, 28, 28]` | 16,384 | 12.845M |
| 43 | `backbone.stages.1.op_list.1.main.inverted_conv.norm` | Normalization | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 512 |  |
| 44 | `backbone.stages.1.op_list.1.main.inverted_conv.act` | Activation | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 0 |  |
| 45 | `backbone.stages.1.op_list.1.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 2,304 | 1.806M |
| 46 | `backbone.stages.1.op_list.1.main.depth_conv.norm` | Normalization | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 512 |  |
| 47 | `backbone.stages.1.op_list.1.main.depth_conv.act` | Activation | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 0 |  |
| 48 | `backbone.stages.1.op_list.1.main.point_conv.conv` | Pointwise Conv2d | `[[1, 256, 28, 28]]` | `[1, 64, 28, 28]` | 16,384 | 12.845M |
| 49 | `backbone.stages.1.op_list.1.main.point_conv.norm` | Normalization | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 128 |  |
| 50 | `backbone.stages.1.op_list.1.main` | MBConv block | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 0 |  |
| 51 | `backbone.stages.1.op_list.1.shortcut` | IdentityLayer | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 0 |  |
| 52 | `backbone.stages.1.op_list.1` | Residual block | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 0 |  |
| 53 | `backbone.stages.1.op_list.2.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 64, 28, 28]]` | `[1, 256, 28, 28]` | 16,384 | 12.845M |
| 54 | `backbone.stages.1.op_list.2.main.inverted_conv.norm` | Normalization | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 512 |  |
| 55 | `backbone.stages.1.op_list.2.main.inverted_conv.act` | Activation | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 0 |  |
| 56 | `backbone.stages.1.op_list.2.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 2,304 | 1.806M |
| 57 | `backbone.stages.1.op_list.2.main.depth_conv.norm` | Normalization | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 512 |  |
| 58 | `backbone.stages.1.op_list.2.main.depth_conv.act` | Activation | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 0 |  |
| 59 | `backbone.stages.1.op_list.2.main.point_conv.conv` | Pointwise Conv2d | `[[1, 256, 28, 28]]` | `[1, 64, 28, 28]` | 16,384 | 12.845M |
| 60 | `backbone.stages.1.op_list.2.main.point_conv.norm` | Normalization | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 128 |  |
| 61 | `backbone.stages.1.op_list.2.main` | MBConv block | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 0 |  |
| 62 | `backbone.stages.1.op_list.2.shortcut` | IdentityLayer | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 0 |  |
| 63 | `backbone.stages.1.op_list.2` | Residual block | `[[1, 64, 28, 28]]` | `[1, 64, 28, 28]` | 0 |  |
| 64 | `backbone.stages.2.op_list.0.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 64, 28, 28]]` | `[1, 256, 28, 28]` | 16,640 | 12.845M |
| 65 | `backbone.stages.2.op_list.0.main.inverted_conv.act` | Activation | `[[1, 256, 28, 28]]` | `[1, 256, 28, 28]` | 0 |  |
| 66 | `backbone.stages.2.op_list.0.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 256, 28, 28]]` | `[1, 256, 14, 14]` | 2,560 | 451.584K |
| 67 | `backbone.stages.2.op_list.0.main.depth_conv.act` | Activation | `[[1, 256, 14, 14]]` | `[1, 256, 14, 14]` | 0 |  |
| 68 | `backbone.stages.2.op_list.0.main.point_conv.conv` | Pointwise Conv2d | `[[1, 256, 14, 14]]` | `[1, 128, 14, 14]` | 32,768 | 6.423M |
| 69 | `backbone.stages.2.op_list.0.main.point_conv.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 70 | `backbone.stages.2.op_list.0.main` | MBConv block | `[[1, 64, 28, 28]]` | `[1, 128, 14, 14]` | 0 |  |
| 71 | `backbone.stages.2.op_list.0` | Residual block | `[[1, 64, 28, 28]]` | `[1, 128, 14, 14]` | 0 |  |
| 72 | `backbone.stages.2.op_list.1.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 384, 14, 14]` | 49,152 | 9.634M |
| 73 | `backbone.stages.2.op_list.1.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 9,600 | 1.882M |
| 74 | `backbone.stages.2.op_list.1.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 6,144 | 1.204M |
| 75 | `backbone.stages.2.op_list.1.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 76 | `backbone.stages.2.op_list.1.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 77 | `backbone.stages.2.op_list.1.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 256, 14, 14]]` | `[1, 128, 14, 14]` | 32,768 | 6.423M |
| 78 | `backbone.stages.2.op_list.1.context_module.main.proj.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 79 | `backbone.stages.2.op_list.1.context_module.main` | Attention block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 | 1.706M |
| 80 | `backbone.stages.2.op_list.1.context_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 81 | `backbone.stages.2.op_list.1.context_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 82 | `backbone.stages.2.op_list.1.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 83 | `backbone.stages.2.op_list.1.local_module.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 84 | `backbone.stages.2.op_list.1.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 5,120 | 903.168K |
| 85 | `backbone.stages.2.op_list.1.local_module.main.depth_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 86 | `backbone.stages.2.op_list.1.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 128, 14, 14]` | 65,536 | 12.845M |
| 87 | `backbone.stages.2.op_list.1.local_module.main.point_conv.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 88 | `backbone.stages.2.op_list.1.local_module.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 89 | `backbone.stages.2.op_list.1.local_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 90 | `backbone.stages.2.op_list.1.local_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 91 | `backbone.stages.2.op_list.2.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 384, 14, 14]` | 49,152 | 9.634M |
| 92 | `backbone.stages.2.op_list.2.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 9,600 | 1.882M |
| 93 | `backbone.stages.2.op_list.2.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 6,144 | 1.204M |
| 94 | `backbone.stages.2.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 95 | `backbone.stages.2.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 96 | `backbone.stages.2.op_list.2.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 256, 14, 14]]` | `[1, 128, 14, 14]` | 32,768 | 6.423M |
| 97 | `backbone.stages.2.op_list.2.context_module.main.proj.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 98 | `backbone.stages.2.op_list.2.context_module.main` | Attention block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 | 1.706M |
| 99 | `backbone.stages.2.op_list.2.context_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 100 | `backbone.stages.2.op_list.2.context_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 101 | `backbone.stages.2.op_list.2.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 102 | `backbone.stages.2.op_list.2.local_module.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 103 | `backbone.stages.2.op_list.2.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 5,120 | 903.168K |
| 104 | `backbone.stages.2.op_list.2.local_module.main.depth_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 105 | `backbone.stages.2.op_list.2.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 128, 14, 14]` | 65,536 | 12.845M |
| 106 | `backbone.stages.2.op_list.2.local_module.main.point_conv.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 107 | `backbone.stages.2.op_list.2.local_module.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 108 | `backbone.stages.2.op_list.2.local_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 109 | `backbone.stages.2.op_list.2.local_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 110 | `backbone.stages.2.op_list.3.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 384, 14, 14]` | 49,152 | 9.634M |
| 111 | `backbone.stages.2.op_list.3.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 9,600 | 1.882M |
| 112 | `backbone.stages.2.op_list.3.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 6,144 | 1.204M |
| 113 | `backbone.stages.2.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 114 | `backbone.stages.2.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 115 | `backbone.stages.2.op_list.3.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 256, 14, 14]]` | `[1, 128, 14, 14]` | 32,768 | 6.423M |
| 116 | `backbone.stages.2.op_list.3.context_module.main.proj.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 117 | `backbone.stages.2.op_list.3.context_module.main` | Attention block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 | 1.706M |
| 118 | `backbone.stages.2.op_list.3.context_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 119 | `backbone.stages.2.op_list.3.context_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 120 | `backbone.stages.2.op_list.3.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 121 | `backbone.stages.2.op_list.3.local_module.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 122 | `backbone.stages.2.op_list.3.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 5,120 | 903.168K |
| 123 | `backbone.stages.2.op_list.3.local_module.main.depth_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 124 | `backbone.stages.2.op_list.3.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 128, 14, 14]` | 65,536 | 12.845M |
| 125 | `backbone.stages.2.op_list.3.local_module.main.point_conv.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 126 | `backbone.stages.2.op_list.3.local_module.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 127 | `backbone.stages.2.op_list.3.local_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 128 | `backbone.stages.2.op_list.3.local_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 129 | `backbone.stages.3.op_list.0.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 130 | `backbone.stages.3.op_list.0.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 131 | `backbone.stages.3.op_list.0.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 7, 7]` | 5,120 | 225.792K |
| 132 | `backbone.stages.3.op_list.0.main.depth_conv.act` | Activation | `[[1, 512, 7, 7]]` | `[1, 512, 7, 7]` | 0 |  |
| 133 | `backbone.stages.3.op_list.0.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 134 | `backbone.stages.3.op_list.0.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 135 | `backbone.stages.3.op_list.0.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 256, 7, 7]` | 0 |  |
| 136 | `backbone.stages.3.op_list.0` | Residual block | `[[1, 128, 14, 14]]` | `[1, 256, 7, 7]` | 0 |  |
| 137 | `backbone.stages.3.op_list.1.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 138 | `backbone.stages.3.op_list.1.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 139 | `backbone.stages.3.op_list.1.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 140 | `backbone.stages.3.op_list.1.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 141 | `backbone.stages.3.op_list.1.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 142 | `backbone.stages.3.op_list.1.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 143 | `backbone.stages.3.op_list.1.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 144 | `backbone.stages.3.op_list.1.context_module.main` | Attention block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 | 852.992K |
| 145 | `backbone.stages.3.op_list.1.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 146 | `backbone.stages.3.op_list.1.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 147 | `backbone.stages.3.op_list.1.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 148 | `backbone.stages.3.op_list.1.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 149 | `backbone.stages.3.op_list.1.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 150 | `backbone.stages.3.op_list.1.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 151 | `backbone.stages.3.op_list.1.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 152 | `backbone.stages.3.op_list.1.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 153 | `backbone.stages.3.op_list.1.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 154 | `backbone.stages.3.op_list.1.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 155 | `backbone.stages.3.op_list.1.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 156 | `backbone.stages.3.op_list.2.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 157 | `backbone.stages.3.op_list.2.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 158 | `backbone.stages.3.op_list.2.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 159 | `backbone.stages.3.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 160 | `backbone.stages.3.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 161 | `backbone.stages.3.op_list.2.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 162 | `backbone.stages.3.op_list.2.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 163 | `backbone.stages.3.op_list.2.context_module.main` | Attention block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 | 852.992K |
| 164 | `backbone.stages.3.op_list.2.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 165 | `backbone.stages.3.op_list.2.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 166 | `backbone.stages.3.op_list.2.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 167 | `backbone.stages.3.op_list.2.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 168 | `backbone.stages.3.op_list.2.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 169 | `backbone.stages.3.op_list.2.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 170 | `backbone.stages.3.op_list.2.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 171 | `backbone.stages.3.op_list.2.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 172 | `backbone.stages.3.op_list.2.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 173 | `backbone.stages.3.op_list.2.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 174 | `backbone.stages.3.op_list.2.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 175 | `backbone.stages.3.op_list.3.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 176 | `backbone.stages.3.op_list.3.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 177 | `backbone.stages.3.op_list.3.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 178 | `backbone.stages.3.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 179 | `backbone.stages.3.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 180 | `backbone.stages.3.op_list.3.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 181 | `backbone.stages.3.op_list.3.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 182 | `backbone.stages.3.op_list.3.context_module.main` | Attention block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 | 852.992K |
| 183 | `backbone.stages.3.op_list.3.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 184 | `backbone.stages.3.op_list.3.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 185 | `backbone.stages.3.op_list.3.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 186 | `backbone.stages.3.op_list.3.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 187 | `backbone.stages.3.op_list.3.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 188 | `backbone.stages.3.op_list.3.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 189 | `backbone.stages.3.op_list.3.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 190 | `backbone.stages.3.op_list.3.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 191 | `backbone.stages.3.op_list.3.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 192 | `backbone.stages.3.op_list.3.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 193 | `backbone.stages.3.op_list.3.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 194 | `backbone.stages.3.op_list.4.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 195 | `backbone.stages.3.op_list.4.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 196 | `backbone.stages.3.op_list.4.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 197 | `backbone.stages.3.op_list.4.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 198 | `backbone.stages.3.op_list.4.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 199 | `backbone.stages.3.op_list.4.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 200 | `backbone.stages.3.op_list.4.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 201 | `backbone.stages.3.op_list.4.context_module.main` | Attention block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 | 852.992K |
| 202 | `backbone.stages.3.op_list.4.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 203 | `backbone.stages.3.op_list.4.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 204 | `backbone.stages.3.op_list.4.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 205 | `backbone.stages.3.op_list.4.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 206 | `backbone.stages.3.op_list.4.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 207 | `backbone.stages.3.op_list.4.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 208 | `backbone.stages.3.op_list.4.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 209 | `backbone.stages.3.op_list.4.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 210 | `backbone.stages.3.op_list.4.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 211 | `backbone.stages.3.op_list.4.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 212 | `backbone.stages.3.op_list.4.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 213 | `head.op_list.0.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1536, 7, 7]` | 393,216 | 19.268M |
| 214 | `head.op_list.0.norm` | Normalization | `[[1, 1536, 7, 7]]` | `[1, 1536, 7, 7]` | 3,072 |  |
| 215 | `head.op_list.0.act` | Activation | `[[1, 1536, 7, 7]]` | `[1, 1536, 7, 7]` | 0 |  |
| 216 | `head.op_list.1` | Pooling | `[[1, 1536, 7, 7]]` | `[1, 1536, 1, 1]` | 0 |  |
| 217 | `head.op_list.2.linear` | Linear | `[[1, 1536]]` | `[1, 1600]` | 2,457,600 | 2.458M |
| 218 | `head.op_list.2.norm` | Normalization | `[[1, 1600]]` | `[1, 1600]` | 3,200 |  |
| 219 | `head.op_list.2.act` | Activation | `[[1, 1600]]` | `[1, 1600]` | 0 |  |
| 220 | `head.op_list.3.linear` | Linear | `[[1, 1600]]` | `[1, 1000]` | 1,601,000 | 1.600M |

## Operation Categories

| Category | Executed modules | Approx. MACs |
|---|---:|---:|
| Activation | 46 |  |
| Attention block | 7 | 8.530M |
| Conv2d | 1 | 5.419M |
| Depthwise Conv2d | 22 | 26.342M |
| IdentityLayer | 18 |  |
| Linear | 2 | 4.058M |
| MBConv block | 14 |  |
| Normalization | 36 |  |
| Pointwise Conv2d | 51 | 474.866M |
| Pooling | 1 |  |
| Residual block | 22 |  |

## Non-MAC Operations

- Approx. non-MAC scalar operations: **45,458,570 (45.459M)**

### Operation Breakdown

| Operation | Count |
|---|---:|
| add | 15,960,727 |
| compare | 10,439,808 |
| div | 9,651,074 |
| mul | 9,400,256 |
| reduction additions (subset of add; do not add again) | 76,926 |
| sqrt | 6,705 |

Non-MAC counts are logical unfused scalar operations for evaluation, over the entire batch. Subtractions count as additions; comparisons and special functions each count as one operation, not one FLOP or equal hardware cost. Reduction additions are already included in additions. Biases, residual sums, LiteMLA denominator arithmetic, supported activations, eval BatchNorm, LayerNorm and adaptive average pooling are included. BatchNorm uses subtract/divide/affine arithmetic plus per-channel epsilon and square root; LayerNorm uses a two-pass mean/variance estimate. Actual kernels may fuse operations or use reciprocals. BatchNorm may fold into convolutions; these counts do not assert that all operations can be eliminated. Reshapes, concatenation, padding, casts, memory traffic and indexing are excluded. Unsupported leaf modules are listed explicitly; unrecognized functional arithmetic inside composites is not traced. --leaf-only retains LiteMLA and residual rows to preserve totals.

Unsupported non-MAC leaf types: **none**.

### Non-MAC Operations by Category

| Category | Non-MAC ops |
|---|---:|
| Activation | 25,346,880 |
| Attention block | 266,560 |
| Conv2d | 0 |
| Depthwise Conv2d | 577,024 |
| IdentityLayer | 0 |
| Linear | 1,000 |
| MBConv block | 0 |
| Normalization | 17,736,738 |
| Pointwise Conv2d | 802,816 |
| Pooling | 75,264 |
| Residual block | 652,288 |

### Non-MAC Operations by Module

| Module | Non-MAC ops | Breakdown |
|---|---:|---|
| `backbone.input_stem.op_list.0.norm` | 802848 | `{"add": 401424, "div": 200704, "mul": 200704, "sqrt": 16}` |
| `backbone.input_stem.op_list.0.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.input_stem.op_list.1.main.depth_conv.norm` | 802848 | `{"add": 401424, "div": 200704, "mul": 200704, "sqrt": 16}` |
| `backbone.input_stem.op_list.1.main.depth_conv.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.input_stem.op_list.1.main.point_conv.norm` | 802848 | `{"add": 401424, "div": 200704, "mul": 200704, "sqrt": 16}` |
| `backbone.input_stem.op_list.1` | 200704 | `{"add": 200704}` |
| `backbone.stages.0.op_list.0.main.inverted_conv.norm` | 3211392 | `{"add": 1605696, "div": 802816, "mul": 802816, "sqrt": 64}` |
| `backbone.stages.0.op_list.0.main.inverted_conv.act` | 4014080 | `{"add": 802816, "compare": 1605632, "div": 802816, "mul": 802816}` |
| `backbone.stages.0.op_list.0.main.depth_conv.norm` | 802944 | `{"add": 401472, "div": 200704, "mul": 200704, "sqrt": 64}` |
| `backbone.stages.0.op_list.0.main.depth_conv.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.stages.0.op_list.0.main.point_conv.norm` | 401472 | `{"add": 200736, "div": 100352, "mul": 100352, "sqrt": 32}` |
| `backbone.stages.0.op_list.1.main.inverted_conv.norm` | 1605888 | `{"add": 802944, "div": 401408, "mul": 401408, "sqrt": 128}` |
| `backbone.stages.0.op_list.1.main.inverted_conv.act` | 2007040 | `{"add": 401408, "compare": 802816, "div": 401408, "mul": 401408}` |
| `backbone.stages.0.op_list.1.main.depth_conv.norm` | 1605888 | `{"add": 802944, "div": 401408, "mul": 401408, "sqrt": 128}` |
| `backbone.stages.0.op_list.1.main.depth_conv.act` | 2007040 | `{"add": 401408, "compare": 802816, "div": 401408, "mul": 401408}` |
| `backbone.stages.0.op_list.1.main.point_conv.norm` | 401472 | `{"add": 200736, "div": 100352, "mul": 100352, "sqrt": 32}` |
| `backbone.stages.0.op_list.1` | 100352 | `{"add": 100352}` |
| `backbone.stages.1.op_list.0.main.inverted_conv.norm` | 1605888 | `{"add": 802944, "div": 401408, "mul": 401408, "sqrt": 128}` |
| `backbone.stages.1.op_list.0.main.inverted_conv.act` | 2007040 | `{"add": 401408, "compare": 802816, "div": 401408, "mul": 401408}` |
| `backbone.stages.1.op_list.0.main.depth_conv.norm` | 401664 | `{"add": 200832, "div": 100352, "mul": 100352, "sqrt": 128}` |
| `backbone.stages.1.op_list.0.main.depth_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.1.op_list.0.main.point_conv.norm` | 200832 | `{"add": 100416, "div": 50176, "mul": 50176, "sqrt": 64}` |
| `backbone.stages.1.op_list.1.main.inverted_conv.norm` | 803328 | `{"add": 401664, "div": 200704, "mul": 200704, "sqrt": 256}` |
| `backbone.stages.1.op_list.1.main.inverted_conv.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.stages.1.op_list.1.main.depth_conv.norm` | 803328 | `{"add": 401664, "div": 200704, "mul": 200704, "sqrt": 256}` |
| `backbone.stages.1.op_list.1.main.depth_conv.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.stages.1.op_list.1.main.point_conv.norm` | 200832 | `{"add": 100416, "div": 50176, "mul": 50176, "sqrt": 64}` |
| `backbone.stages.1.op_list.1` | 50176 | `{"add": 50176}` |
| `backbone.stages.1.op_list.2.main.inverted_conv.norm` | 803328 | `{"add": 401664, "div": 200704, "mul": 200704, "sqrt": 256}` |
| `backbone.stages.1.op_list.2.main.inverted_conv.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.stages.1.op_list.2.main.depth_conv.norm` | 803328 | `{"add": 401664, "div": 200704, "mul": 200704, "sqrt": 256}` |
| `backbone.stages.1.op_list.2.main.depth_conv.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.stages.1.op_list.2.main.point_conv.norm` | 200832 | `{"add": 100416, "div": 50176, "mul": 50176, "sqrt": 64}` |
| `backbone.stages.1.op_list.2` | 50176 | `{"add": 50176}` |
| `backbone.stages.2.op_list.0.main.inverted_conv.conv` | 200704 | `{"add": 200704}` |
| `backbone.stages.2.op_list.0.main.inverted_conv.act` | 1003520 | `{"add": 200704, "compare": 401408, "div": 200704, "mul": 200704}` |
| `backbone.stages.2.op_list.0.main.depth_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.2.op_list.0.main.depth_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.2.op_list.0.main.point_conv.norm` | 100608 | `{"add": 50304, "div": 25088, "mul": 25088, "sqrt": 128}` |
| `backbone.stages.2.op_list.1.context_module.main.kernel_func` | 50176 | `{"compare": 50176}` |
| `backbone.stages.2.op_list.1.context_module.main.kernel_func` | 50176 | `{"compare": 50176}` |
| `backbone.stages.2.op_list.1.context_module.main.proj.norm` | 100608 | `{"add": 50304, "div": 25088, "mul": 25088, "sqrt": 128}` |
| `backbone.stages.2.op_list.1.context_module.main` | 53312 | `{"add": 3136, "div": 50176}` |
| `backbone.stages.2.op_list.1.context_module` | 25088 | `{"add": 25088}` |
| `backbone.stages.2.op_list.1.local_module.main.inverted_conv.conv` | 100352 | `{"add": 100352}` |
| `backbone.stages.2.op_list.1.local_module.main.inverted_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.2.op_list.1.local_module.main.depth_conv.conv` | 100352 | `{"add": 100352}` |
| `backbone.stages.2.op_list.1.local_module.main.depth_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.2.op_list.1.local_module.main.point_conv.norm` | 100608 | `{"add": 50304, "div": 25088, "mul": 25088, "sqrt": 128}` |
| `backbone.stages.2.op_list.1.local_module` | 25088 | `{"add": 25088}` |
| `backbone.stages.2.op_list.2.context_module.main.kernel_func` | 50176 | `{"compare": 50176}` |
| `backbone.stages.2.op_list.2.context_module.main.kernel_func` | 50176 | `{"compare": 50176}` |
| `backbone.stages.2.op_list.2.context_module.main.proj.norm` | 100608 | `{"add": 50304, "div": 25088, "mul": 25088, "sqrt": 128}` |
| `backbone.stages.2.op_list.2.context_module.main` | 53312 | `{"add": 3136, "div": 50176}` |
| `backbone.stages.2.op_list.2.context_module` | 25088 | `{"add": 25088}` |
| `backbone.stages.2.op_list.2.local_module.main.inverted_conv.conv` | 100352 | `{"add": 100352}` |
| `backbone.stages.2.op_list.2.local_module.main.inverted_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.2.op_list.2.local_module.main.depth_conv.conv` | 100352 | `{"add": 100352}` |
| `backbone.stages.2.op_list.2.local_module.main.depth_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.2.op_list.2.local_module.main.point_conv.norm` | 100608 | `{"add": 50304, "div": 25088, "mul": 25088, "sqrt": 128}` |
| `backbone.stages.2.op_list.2.local_module` | 25088 | `{"add": 25088}` |
| `backbone.stages.2.op_list.3.context_module.main.kernel_func` | 50176 | `{"compare": 50176}` |
| `backbone.stages.2.op_list.3.context_module.main.kernel_func` | 50176 | `{"compare": 50176}` |
| `backbone.stages.2.op_list.3.context_module.main.proj.norm` | 100608 | `{"add": 50304, "div": 25088, "mul": 25088, "sqrt": 128}` |
| `backbone.stages.2.op_list.3.context_module.main` | 53312 | `{"add": 3136, "div": 50176}` |
| `backbone.stages.2.op_list.3.context_module` | 25088 | `{"add": 25088}` |
| `backbone.stages.2.op_list.3.local_module.main.inverted_conv.conv` | 100352 | `{"add": 100352}` |
| `backbone.stages.2.op_list.3.local_module.main.inverted_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.2.op_list.3.local_module.main.depth_conv.conv` | 100352 | `{"add": 100352}` |
| `backbone.stages.2.op_list.3.local_module.main.depth_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.2.op_list.3.local_module.main.point_conv.norm` | 100608 | `{"add": 50304, "div": 25088, "mul": 25088, "sqrt": 128}` |
| `backbone.stages.2.op_list.3.local_module` | 25088 | `{"add": 25088}` |
| `backbone.stages.3.op_list.0.main.inverted_conv.conv` | 100352 | `{"add": 100352}` |
| `backbone.stages.3.op_list.0.main.inverted_conv.act` | 501760 | `{"add": 100352, "compare": 200704, "div": 100352, "mul": 100352}` |
| `backbone.stages.3.op_list.0.main.depth_conv.conv` | 25088 | `{"add": 25088}` |
| `backbone.stages.3.op_list.0.main.depth_conv.act` | 125440 | `{"add": 25088, "compare": 50176, "div": 25088, "mul": 25088}` |
| `backbone.stages.3.op_list.0.main.point_conv.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.1.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.1.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.1.context_module.main.proj.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.1.context_module.main` | 26656 | `{"add": 1568, "div": 25088}` |
| `backbone.stages.3.op_list.1.context_module` | 12544 | `{"add": 12544}` |
| `backbone.stages.3.op_list.1.local_module.main.inverted_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.1.local_module.main.inverted_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.1.local_module.main.depth_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.1.local_module.main.depth_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.1.local_module.main.point_conv.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.1.local_module` | 12544 | `{"add": 12544}` |
| `backbone.stages.3.op_list.2.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.2.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.2.context_module.main.proj.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.2.context_module.main` | 26656 | `{"add": 1568, "div": 25088}` |
| `backbone.stages.3.op_list.2.context_module` | 12544 | `{"add": 12544}` |
| `backbone.stages.3.op_list.2.local_module.main.inverted_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.2.local_module.main.inverted_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.2.local_module.main.depth_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.2.local_module.main.depth_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.2.local_module.main.point_conv.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.2.local_module` | 12544 | `{"add": 12544}` |
| `backbone.stages.3.op_list.3.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.3.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.3.context_module.main.proj.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.3.context_module.main` | 26656 | `{"add": 1568, "div": 25088}` |
| `backbone.stages.3.op_list.3.context_module` | 12544 | `{"add": 12544}` |
| `backbone.stages.3.op_list.3.local_module.main.inverted_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.3.local_module.main.inverted_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.3.local_module.main.depth_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.3.local_module.main.depth_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.3.local_module.main.point_conv.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.3.local_module` | 12544 | `{"add": 12544}` |
| `backbone.stages.3.op_list.4.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.4.context_module.main.kernel_func` | 25088 | `{"compare": 25088}` |
| `backbone.stages.3.op_list.4.context_module.main.proj.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.4.context_module.main` | 26656 | `{"add": 1568, "div": 25088}` |
| `backbone.stages.3.op_list.4.context_module` | 12544 | `{"add": 12544}` |
| `backbone.stages.3.op_list.4.local_module.main.inverted_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.4.local_module.main.inverted_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.4.local_module.main.depth_conv.conv` | 50176 | `{"add": 50176}` |
| `backbone.stages.3.op_list.4.local_module.main.depth_conv.act` | 250880 | `{"add": 50176, "compare": 100352, "div": 50176, "mul": 50176}` |
| `backbone.stages.3.op_list.4.local_module.main.point_conv.norm` | 50688 | `{"add": 25344, "div": 12544, "mul": 12544, "sqrt": 256}` |
| `backbone.stages.3.op_list.4.local_module` | 12544 | `{"add": 12544}` |
| `head.op_list.0.norm` | 304128 | `{"add": 152064, "div": 75264, "mul": 75264, "sqrt": 1536}` |
| `head.op_list.0.act` | 376320 | `{"add": 75264, "compare": 150528, "div": 75264, "mul": 75264}` |
| `head.op_list.1` | 75264 | `{"add": 73728, "div": 1536, "reduction_add": 73728}` |
| `head.op_list.2.norm` | 11202 | `{"add": 6399, "div": 1602, "mul": 3200, "reduction_add": 3198, "sqrt": 1}` |
| `head.op_list.2.act` | 8000 | `{"add": 1600, "compare": 3200, "div": 1600, "mul": 1600}` |
| `head.op_list.3.linear` | 1000 | `{"add": 1000}` |
