# Architecture Report — efficientvit-b1-r224

## Summary

- Input shape: `[1, 3, 224, 224]`
- Output shape: `[1, 1000]`
- Total parameters: **9,102,024 (9.102M)**
- Trainable parameters: **9,102,024 (9.102M)**
- Approx. MACs counted from Conv2d/Linear leaf modules: **510,684,672 (510.685M)**

> MAC count is an approximation. Custom attention operations may contain tensor/matrix operations that are not represented as nn.Conv2d or nn.Linear. Their internal Conv/Linear modules are counted where applicable.

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
| 79 | `backbone.stages.2.op_list.1.context_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 80 | `backbone.stages.2.op_list.1.context_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 81 | `backbone.stages.2.op_list.1.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 82 | `backbone.stages.2.op_list.1.local_module.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 83 | `backbone.stages.2.op_list.1.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 5,120 | 903.168K |
| 84 | `backbone.stages.2.op_list.1.local_module.main.depth_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 85 | `backbone.stages.2.op_list.1.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 128, 14, 14]` | 65,536 | 12.845M |
| 86 | `backbone.stages.2.op_list.1.local_module.main.point_conv.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 87 | `backbone.stages.2.op_list.1.local_module.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 88 | `backbone.stages.2.op_list.1.local_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 89 | `backbone.stages.2.op_list.1.local_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 90 | `backbone.stages.2.op_list.2.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 384, 14, 14]` | 49,152 | 9.634M |
| 91 | `backbone.stages.2.op_list.2.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 9,600 | 1.882M |
| 92 | `backbone.stages.2.op_list.2.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 6,144 | 1.204M |
| 93 | `backbone.stages.2.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 94 | `backbone.stages.2.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 95 | `backbone.stages.2.op_list.2.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 256, 14, 14]]` | `[1, 128, 14, 14]` | 32,768 | 6.423M |
| 96 | `backbone.stages.2.op_list.2.context_module.main.proj.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 97 | `backbone.stages.2.op_list.2.context_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 98 | `backbone.stages.2.op_list.2.context_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 99 | `backbone.stages.2.op_list.2.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 100 | `backbone.stages.2.op_list.2.local_module.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 101 | `backbone.stages.2.op_list.2.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 5,120 | 903.168K |
| 102 | `backbone.stages.2.op_list.2.local_module.main.depth_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 103 | `backbone.stages.2.op_list.2.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 128, 14, 14]` | 65,536 | 12.845M |
| 104 | `backbone.stages.2.op_list.2.local_module.main.point_conv.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 105 | `backbone.stages.2.op_list.2.local_module.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 106 | `backbone.stages.2.op_list.2.local_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 107 | `backbone.stages.2.op_list.2.local_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 108 | `backbone.stages.2.op_list.3.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 384, 14, 14]` | 49,152 | 9.634M |
| 109 | `backbone.stages.2.op_list.3.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 9,600 | 1.882M |
| 110 | `backbone.stages.2.op_list.3.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 384, 14, 14]]` | `[1, 384, 14, 14]` | 6,144 | 1.204M |
| 111 | `backbone.stages.2.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 112 | `backbone.stages.2.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 16, 16, 196]]` | `[1, 16, 16, 196]` | 0 |  |
| 113 | `backbone.stages.2.op_list.3.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 256, 14, 14]]` | `[1, 128, 14, 14]` | 32,768 | 6.423M |
| 114 | `backbone.stages.2.op_list.3.context_module.main.proj.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 115 | `backbone.stages.2.op_list.3.context_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 116 | `backbone.stages.2.op_list.3.context_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 117 | `backbone.stages.2.op_list.3.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 118 | `backbone.stages.2.op_list.3.local_module.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 119 | `backbone.stages.2.op_list.3.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 5,120 | 903.168K |
| 120 | `backbone.stages.2.op_list.3.local_module.main.depth_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 121 | `backbone.stages.2.op_list.3.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 128, 14, 14]` | 65,536 | 12.845M |
| 122 | `backbone.stages.2.op_list.3.local_module.main.point_conv.norm` | Normalization | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 256 |  |
| 123 | `backbone.stages.2.op_list.3.local_module.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 124 | `backbone.stages.2.op_list.3.local_module.shortcut` | IdentityLayer | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 125 | `backbone.stages.2.op_list.3.local_module` | Residual block | `[[1, 128, 14, 14]]` | `[1, 128, 14, 14]` | 0 |  |
| 126 | `backbone.stages.3.op_list.0.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 128, 14, 14]]` | `[1, 512, 14, 14]` | 66,048 | 12.845M |
| 127 | `backbone.stages.3.op_list.0.main.inverted_conv.act` | Activation | `[[1, 512, 14, 14]]` | `[1, 512, 14, 14]` | 0 |  |
| 128 | `backbone.stages.3.op_list.0.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 512, 14, 14]]` | `[1, 512, 7, 7]` | 5,120 | 225.792K |
| 129 | `backbone.stages.3.op_list.0.main.depth_conv.act` | Activation | `[[1, 512, 7, 7]]` | `[1, 512, 7, 7]` | 0 |  |
| 130 | `backbone.stages.3.op_list.0.main.point_conv.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 131 | `backbone.stages.3.op_list.0.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 132 | `backbone.stages.3.op_list.0.main` | MBConv block | `[[1, 128, 14, 14]]` | `[1, 256, 7, 7]` | 0 |  |
| 133 | `backbone.stages.3.op_list.0` | Residual block | `[[1, 128, 14, 14]]` | `[1, 256, 7, 7]` | 0 |  |
| 134 | `backbone.stages.3.op_list.1.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 135 | `backbone.stages.3.op_list.1.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 136 | `backbone.stages.3.op_list.1.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 137 | `backbone.stages.3.op_list.1.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 138 | `backbone.stages.3.op_list.1.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 139 | `backbone.stages.3.op_list.1.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 140 | `backbone.stages.3.op_list.1.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 141 | `backbone.stages.3.op_list.1.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 142 | `backbone.stages.3.op_list.1.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 143 | `backbone.stages.3.op_list.1.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 144 | `backbone.stages.3.op_list.1.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 145 | `backbone.stages.3.op_list.1.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 146 | `backbone.stages.3.op_list.1.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 147 | `backbone.stages.3.op_list.1.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 148 | `backbone.stages.3.op_list.1.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 149 | `backbone.stages.3.op_list.1.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 150 | `backbone.stages.3.op_list.1.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 151 | `backbone.stages.3.op_list.1.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 152 | `backbone.stages.3.op_list.2.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 153 | `backbone.stages.3.op_list.2.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 154 | `backbone.stages.3.op_list.2.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 155 | `backbone.stages.3.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 156 | `backbone.stages.3.op_list.2.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 157 | `backbone.stages.3.op_list.2.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 158 | `backbone.stages.3.op_list.2.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 159 | `backbone.stages.3.op_list.2.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 160 | `backbone.stages.3.op_list.2.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 161 | `backbone.stages.3.op_list.2.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 162 | `backbone.stages.3.op_list.2.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 163 | `backbone.stages.3.op_list.2.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 164 | `backbone.stages.3.op_list.2.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 165 | `backbone.stages.3.op_list.2.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 166 | `backbone.stages.3.op_list.2.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 167 | `backbone.stages.3.op_list.2.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 168 | `backbone.stages.3.op_list.2.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 169 | `backbone.stages.3.op_list.2.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 170 | `backbone.stages.3.op_list.3.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 171 | `backbone.stages.3.op_list.3.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 172 | `backbone.stages.3.op_list.3.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 173 | `backbone.stages.3.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 174 | `backbone.stages.3.op_list.3.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 175 | `backbone.stages.3.op_list.3.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 176 | `backbone.stages.3.op_list.3.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 177 | `backbone.stages.3.op_list.3.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 178 | `backbone.stages.3.op_list.3.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 179 | `backbone.stages.3.op_list.3.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 180 | `backbone.stages.3.op_list.3.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 181 | `backbone.stages.3.op_list.3.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 182 | `backbone.stages.3.op_list.3.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 183 | `backbone.stages.3.op_list.3.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 184 | `backbone.stages.3.op_list.3.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 185 | `backbone.stages.3.op_list.3.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 186 | `backbone.stages.3.op_list.3.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 187 | `backbone.stages.3.op_list.3.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 188 | `backbone.stages.3.op_list.4.context_module.main.qkv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 768, 7, 7]` | 196,608 | 9.634M |
| 189 | `backbone.stages.3.op_list.4.context_module.main.aggreg.0.0` | Depthwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 19,200 | 940.800K |
| 190 | `backbone.stages.3.op_list.4.context_module.main.aggreg.0.1` | Pointwise Conv2d | `[[1, 768, 7, 7]]` | `[1, 768, 7, 7]` | 12,288 | 602.112K |
| 191 | `backbone.stages.3.op_list.4.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 192 | `backbone.stages.3.op_list.4.context_module.main.kernel_func` | Activation | `[[1, 32, 16, 49]]` | `[1, 32, 16, 49]` | 0 |  |
| 193 | `backbone.stages.3.op_list.4.context_module.main.proj.conv` | Pointwise Conv2d | `[[1, 512, 7, 7]]` | `[1, 256, 7, 7]` | 131,072 | 6.423M |
| 194 | `backbone.stages.3.op_list.4.context_module.main.proj.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 195 | `backbone.stages.3.op_list.4.context_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 196 | `backbone.stages.3.op_list.4.context_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 197 | `backbone.stages.3.op_list.4.local_module.main.inverted_conv.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1024, 7, 7]` | 263,168 | 12.845M |
| 198 | `backbone.stages.3.op_list.4.local_module.main.inverted_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 199 | `backbone.stages.3.op_list.4.local_module.main.depth_conv.conv` | Depthwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 10,240 | 451.584K |
| 200 | `backbone.stages.3.op_list.4.local_module.main.depth_conv.act` | Activation | `[[1, 1024, 7, 7]]` | `[1, 1024, 7, 7]` | 0 |  |
| 201 | `backbone.stages.3.op_list.4.local_module.main.point_conv.conv` | Pointwise Conv2d | `[[1, 1024, 7, 7]]` | `[1, 256, 7, 7]` | 262,144 | 12.845M |
| 202 | `backbone.stages.3.op_list.4.local_module.main.point_conv.norm` | Normalization | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 512 |  |
| 203 | `backbone.stages.3.op_list.4.local_module.main` | MBConv block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 204 | `backbone.stages.3.op_list.4.local_module.shortcut` | IdentityLayer | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 205 | `backbone.stages.3.op_list.4.local_module` | Residual block | `[[1, 256, 7, 7]]` | `[1, 256, 7, 7]` | 0 |  |
| 206 | `head.op_list.0.conv` | Pointwise Conv2d | `[[1, 256, 7, 7]]` | `[1, 1536, 7, 7]` | 393,216 | 19.268M |
| 207 | `head.op_list.0.norm` | Normalization | `[[1, 1536, 7, 7]]` | `[1, 1536, 7, 7]` | 3,072 |  |
| 208 | `head.op_list.0.act` | Activation | `[[1, 1536, 7, 7]]` | `[1, 1536, 7, 7]` | 0 |  |
| 209 | `head.op_list.1` | Pooling | `[[1, 1536, 7, 7]]` | `[1, 1536, 1, 1]` | 0 |  |
| 210 | `head.op_list.2.linear` | Linear | `[[1, 1536]]` | `[1, 1600]` | 2,457,600 | 2.458M |
| 211 | `head.op_list.2.norm` | Normalization | `[[1, 1600]]` | `[1, 1600]` | 3,200 |  |
| 212 | `head.op_list.2.act` | Activation | `[[1, 1600]]` | `[1, 1600]` | 0 |  |
| 213 | `head.op_list.3.linear` | Linear | `[[1, 1600]]` | `[1, 1000]` | 1,601,000 | 1.600M |

## Operation Categories

| Category | Executed modules | Approx. MACs |
|---|---:|---:|
| Activation | 46 |  |
| Conv2d | 1 | 5.419M |
| Depthwise Conv2d | 22 | 26.342M |
| IdentityLayer | 18 |  |
| Linear | 2 | 4.058M |
| MBConv block | 14 |  |
| Normalization | 36 |  |
| Pointwise Conv2d | 51 | 474.866M |
| Pooling | 1 |  |
| Residual block | 22 |  |
