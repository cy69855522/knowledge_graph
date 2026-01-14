# 第八章：Stencil 计算

## 核心概念

### 8.1 Stencil 定义
Stencil 是卷积的泛化，输出元素是输入邻域元素的函数（不限于加权和）。

**与卷积的区别**：
- 卷积：加权和
- Stencil：任意函数（如偏微分方程的有限差分）

### 8.2 3D Stencil 示例（7点 Laplacian）
```cpp
__global__ void stencil_3d(float *in, float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 && k < N-1) {
        out[i*N*N + j*N + k] = c0 * in[i*N*N + j*N + k]
            + c1 * in[(i-1)*N*N + j*N + k]
            + c1 * in[(i+1)*N*N + j*N + k]
            + c1 * in[i*N*N + (j-1)*N + k]
            + c1 * in[i*N*N + (j+1)*N + k]
            + c1 * in[i*N*N + j*N + (k-1)]
            + c1 * in[i*N*N + j*N + (k+1)];
    }
}
```

### 8.3 Tiling 优化

**挑战**：3D 共享内存需要大量空间

**解决方案**：2D Tiling + 寄存器
- 共享内存存储 X-Y 平面
- 寄存器存储 Z 方向的前后元素

```cpp
__global__ void stencil_3d_tiled(float *in, float *out, int N) {
    __shared__ float ds_in[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int i = blockIdx.x * (TILE_SIZE - 2) + tx;
    int j = blockIdx.y * (TILE_SIZE - 2) + ty;
    
    float inPrev, inCurr, inNext;
    
    // 初始化
    inPrev = in[i*N*N + j*N + 0];
    inCurr = in[i*N*N + j*N + 1];
    
    for (int k = 1; k < N-1; k++) {
        inNext = in[i*N*N + j*N + (k+1)];
        ds_in[ty][tx] = inCurr;
        __syncthreads();
        
        if (tx > 0 && tx < TILE_SIZE-1 && ty > 0 && ty < TILE_SIZE-1) {
            out[i*N*N + j*N + k] = c0 * inCurr
                + c1 * (ds_in[ty-1][tx] + ds_in[ty+1][tx]
                      + ds_in[ty][tx-1] + ds_in[ty][tx+1]
                      + inPrev + inNext);
        }
        __syncthreads();
        
        inPrev = inCurr;
        inCurr = inNext;
    }
}
```

### 8.4 线程粗化 (Thread Coarsening)

每个线程处理多个 Z 方向的输出点：
- 减少线程块数量
- 增加每个线程的工作量
- 更好的寄存器利用

## 关键要点

| 优化技术 | 效果 |
|---------|------|
| 2D Tiling | 减少 X-Y 方向全局内存访问 |
| 寄存器存储 | 处理 Z 方向依赖 |
| 线程粗化 | 减少并行开销 |

### 迭代求解器
Stencil 常用于迭代求解偏微分方程：
1. 初始化边界条件
2. 重复应用 stencil 直到收敛
3. 使用双缓冲避免读写冲突
