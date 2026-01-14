# CUDA-SGEMM: AI Infra 入门实战

## 🎯 项目目标

从零开始手写并优化 **SGEMM (单精度通用矩阵乘法)**。
这是 AI Infra 领域的 "Hello World"，目的是理解 GPU 硬件架构与软件性能的映射关系。

## 🛠 技术栈

* **语言：** CUDA C++ (NVCC)。
* **分析工具：** Nsight Compute (ncu), Nsight Systems (nsys)。
* **参考教材：** *Programming Massively Parallel Processors (4th Edition)*.

## 📚 学习重点

我需要理解以下硬件概念是如何对应到代码的：

1. **显存层级：** Global Memory (HBM) -> Shared Memory (L1 Cache) -> Registers.
2. **延迟掩盖：** Memory Coalescing (合并访问), Tiling (分块), Loop Unrolling.
3. **并行与竞争：** Thread Layout, Bank Conflicts, Warp Divergence.

---

## 🤖 AI 助手指令 (Instructions for Cursor)

**设定：你是一名 NVIDIA 性能优化工程师。**

1. **循序渐进：** 引导我按照以下版本迭代代码：
   * v0: Naive (最笨的写法)
   * v1: Global Memory Coalescing (合并访问)
   * v2: Shared Memory Tiling (共享内存分块)
   * v3: Vectorized Memory Access (向量化加载)
2. **硬件视角：** 解释代码时，必须结合 GPU 硬件。
   * *例子：* 不要只说“用共享内存”，要说“为了减少对 HBM 的带宽压力，利用片上高速缓存”。
3. **代码分析：** 如果我写了 Kernel，请帮我分析它的**理论占有率 (Occupancy)** 和潜在的性能瓶颈。
