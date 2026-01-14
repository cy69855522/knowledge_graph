# 第七章：卷积 - 常量内存和缓存简介

## 核心概念

### 7.1 卷积定义
卷积是一种数组操作，每个输出元素是输入元素邻域的加权和，权重由**卷积核（滤波器）**定义。

```cpp
// 1D 卷积
for (int i = 0; i < N; i++) {
    float sum = 0;
    for (int j = -R; j <= R; j++) {
        sum += input[i + j] * filter[R + j];
    }
    output[i] = sum;
}
```

### 7.2 常量内存 (Constant Memory)
- **特点**：只读、被缓存、广播访问
- **大小**：64KB（设备级别）
- **优势**：当 warp 中所有线程访问同一地址时，只需一次内存访问
- **声明**：`__constant__ float F[MAX_SIZE];`
- **传输**：`cudaMemcpyToSymbol(F, h_F, size)`

```cpp
__constant__ float F[2*FILTER_RADIUS + 1];

__global__ void conv1D(float *N, float *P, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
        if (i + j >= 0 && i + j < width) {
            Pvalue += N[i + j] * F[FILTER_RADIUS + j];
        }
    }
    P[i] = Pvalue;
}
```

### 7.3 Tiling 优化
使用共享内存减少全局内存访问：

```cpp
__global__ void conv1D_tiled(float *N, float *P, int width) {
    __shared__ float Ns[TILE_SIZE + 2*FILTER_RADIUS];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = FILTER_RADIUS + threadIdx.x;
    
    // 加载中间部分
    Ns[n] = (i < width) ? N[i] : 0;
    
    // 加载 halo 元素
    if (threadIdx.x < FILTER_RADIUS) {
        int left = i - FILTER_RADIUS;
        int right = i + TILE_SIZE;
        Ns[threadIdx.x] = (left >= 0) ? N[left] : 0;
        Ns[n + TILE_SIZE] = (right < width) ? N[right] : 0;
    }
    __syncthreads();
    
    // 计算
    float Pvalue = 0;
    for (int j = 0; j < 2*FILTER_RADIUS + 1; j++) {
        Pvalue += Ns[threadIdx.x + j] * F[j];
    }
    if (i < width) P[i] = Pvalue;
}
```

### 7.4 2D 卷积
```cpp
__global__ void conv2D(float *N, float *P, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    float Pvalue = 0;
    for (int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++) {
        for (int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++) {
            int inRow = row + fRow;
            int inCol = col + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += N[inRow * width + inCol] * F[(fRow+R)*(2*R+1) + (fCol+R)];
            }
        }
    }
    if (row < height && col < width) P[row * width + col] = Pvalue;
}
```

## 关键要点

| 内存类型 | 位置 | 缓存 | 访问模式 | 适用场景 |
|---------|------|------|---------|---------|
| 常量内存 | 片外 | L1/L2 | 广播 | 卷积核、只读参数 |
| 共享内存 | 片上 | - | 随机 | Tiling、数据重用 |
| 全局内存 | 片外 | L2 | 合并 | 大数据 |

### 边界处理策略
1. **Ghost cells**：填充 0 或边界值
2. **边界检查**：if 条件判断
3. **Padding**：预先扩展输入数组

### 性能优化
- 使用常量内存存储滤波器（广播访问）
- 使用共享内存 tiling 减少全局内存访问
- 合理处理 halo 元素加载
- 考虑 L2 缓存对 halo 访问的优化
