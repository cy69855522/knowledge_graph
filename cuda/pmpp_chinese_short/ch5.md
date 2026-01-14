# 第五章：内存架构和数据局部性 ⭐

## 5.1 内存访问效率的重要性

### 算术强度 (Arithmetic Intensity)
$$
\text{算术强度} = \frac{\text{FLOP}}{\text{访问的字节数}}
$$

### 矩阵乘法示例
```cpp
for (int k = 0; k < Width; k++) {
    Pvalue += M[Row * Width + k] * N[k * Width + Col];
}
```
- 每次迭代：2 FLOP (1 乘 + 1 加)，8 B 内存访问 (2 × 4B float)
- 算术强度：2 / 8 = **0.25 FLOP/B**

### A100 性能分析
| 指标 | 数值 |
|------|------|
| 内存带宽 | 1555 GB/s |
| 峰值算力 | 19,500 GFLOPS |
| Naive 矩阵乘法吞吐量 | 1555 × 0.25 = 389 GFLOPS |
| 峰值利用率 | 389 / 19500 = **2%** ❌ |

**结论**：Naive 实现是**内存密集型 (Memory Bound)**，性能受限于内存带宽。

### Roofline 模型
```
吞吐量
   ↑
   │     ╱ 计算密集型
   │    ╱
   │   ╱  ← 屋顶线
   │  ╱
   │ ╱ 内存密集型
   └──────────────→ 算术强度
```
- 斜线：内存带宽限制
- 水平线：计算能力限制
- 交点：从内存密集型转为计算密集型的临界点

## 5.2 CUDA 内存类型

| 内存类型 | 位置 | 延迟 | 作用域 | 生命周期 |
|----------|------|------|--------|----------|
| **寄存器** | 片上 | 最快 | 单线程 | Kernel |
| **共享内存** | 片上 | 快 | 块内 | Kernel |
| **全局内存** | 片外 (DRAM) | 慢 | 全局 | 应用程序 |
| **常量内存** | 片外 (缓存) | 快 (缓存命中) | 全局 | 应用程序 |
| **本地内存** | 片外 | 慢 | 单线程 | Kernel |

### 声明方式
```cpp
__global__ void kernel() {
    int a;                          // 寄存器 (自动变量)
    __shared__ float s[256];        // 共享内存
    // 全局内存通过 cudaMalloc 分配
}

__constant__ float c[1024];         // 常量内存 (函数外声明)
__device__ float g[1024];           // 全局变量 (函数外声明)
```

### 关键特性
- **寄存器**：最快，但数量有限（影响占用率）
- **共享内存**：块内线程共享，需要 `__syncthreads()` 同步
- **全局内存**：容量大但慢，是主要瓶颈

## 5.3 Tiling (分块) 技术 ⭐

### 核心思想
将数据分成小块 (Tile)，加载到共享内存，**复用**数据减少全局内存访问。

### 矩阵乘法中的数据复用
```
P[i][j] = Σ M[i][k] * N[k][j]
```
- M 的第 i 行被 **同一行的所有 P 元素** 使用
- N 的第 j 列被 **同一列的所有 P 元素** 使用

**复用机会**：如果 TILE_WIDTH = 16，每个数据可被复用 16 次！

### 分阶段执行
```
阶段 1: 加载 M 的 Tile 0 和 N 的 Tile 0 到共享内存
        计算部分点积
阶段 2: 加载 M 的 Tile 1 和 N 的 Tile 1 到共享内存
        累加到点积
...
```

## 5.4 Tiled 矩阵乘法 Kernel ⭐

```cpp
#define TILE_WIDTH 16

__global__ void matrixMulTiled(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    
    // 分阶段计算
    for (int ph = 0; ph < Width / TILE_WIDTH; ph++) {
        // 协作加载 Tile 到共享内存
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();  // 等待所有线程完成加载
        
        // 使用共享内存计算部分点积
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();  // 等待所有线程完成使用
    }
    
    P[Row * Width + Col] = Pvalue;
}
```

### 关键点
1. **协作加载**：每个线程加载 1 个 M 元素 + 1 个 N 元素
2. **两次同步**：
   - 第一次：确保数据加载完成（写后读依赖）
   - 第二次：确保数据使用完成（读后写依赖）
3. **复用共享内存**：每个阶段重用 Mds 和 Nds

### 性能提升
- 全局内存访问减少 **TILE_WIDTH 倍**
- TILE_WIDTH = 16 时：
  - 算术强度：0.25 × 16 = **4 FLOP/B**
  - 预期吞吐量：1555 × 4 = **6220 GFLOPS** (32% 峰值)

## 5.5 边界检查

当矩阵维度不是 TILE_WIDTH 的倍数时，需要边界检查：

```cpp
// 加载时检查边界
if (Row < Width && (ph * TILE_WIDTH + tx) < Width)
    Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
else
    Mds[ty][tx] = 0.0f;  // 越界填充 0

if ((ph * TILE_WIDTH + ty) < Width && Col < Width)
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
else
    Nds[ty][tx] = 0.0f;

// 存储时检查边界
if (Row < Width && Col < Width)
    P[Row * Width + Col] = Pvalue;
```

## 5.6 共享内存对占用率的影响

### A100 共享内存限制
- 每 SM 最大：164 KB
- 最大线程数：2048

### 计算平均使用量
$$
\text{平均共享内存/线程} = \frac{\text{块使用的共享内存}}{\text{块内线程数}}
$$

**示例**：32 KB 共享内存，256 线程/块
- 平均：32 KB / 256 = 128 B/线程
- 最大线程数：164 KB / 128 B = 1280 线程
- 占用率：1280 / 2048 = **62%** ⚠️

### 动态共享内存分配
```cpp
extern __shared__ float shared[];  // 动态大小

// 调用时指定大小
kernel<<<grid, block, sharedMemSize>>>(args...);
```

## 5.7 核心总结

| 优化技术 | 效果 |
|----------|------|
| **Tiling** | 减少全局内存访问，提升算术强度 |
| **共享内存** | 块内数据共享，低延迟访问 |
| **协作加载** | 线程协作，一次加载多次使用 |
| **边界检查** | 处理任意维度矩阵 |

### SGEMM 优化路线图
```
Naive (0.25 FLOP/B, 2%)
    ↓ Tiling
Shared Memory (4 FLOP/B, 32%)
    ↓ 更多优化 (寄存器 Tiling, 向量化加载...)
高度优化 (~80%+ 峰值)
```

### 关键公式
$$
\text{全局内存访问减少} = \text{TILE\_WIDTH} \times
$$

$$
\text{占用率} = \min\left(\frac{\text{SM 资源}}{\text{块资源需求}}\right)
$$
