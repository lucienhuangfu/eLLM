---
kind: external_dependency
name: Intel Advanced Matrix Extensions
slug: intel-amx
category: external_dependency
category_hints:
    - client_constraint
scope:
    - '**'
source_files:
    - src/kernel/x86_64/f16_amx
    - src/lib.rs
---

项目针对 Intel Xeon 4代及以上处理器的 AMX 指令集进行优化，利用其矩阵扩展能力加速深度学习计算。AMX 指令集特别适用于大矩阵乘法运算，在 Prefill 阶段显著提升性能。需要硬件支持 AMX 指令集（Xeon 4代+）。