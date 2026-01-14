# 第九章：并行直方图

## 核心概念

### 9.1 直方图定义
统计数据落入各个区间（bin）的频率分布。

```cpp
// 顺序实现
void histogram(char *data, int *hist, int N) {
    for (int i = 0; i < N; i++) {
        hist[data[i]]++;
    }
}
```

### 9.2 并行化挑战
**读-修改-写竞争**：多个线程可能同时更新同一个 bin

### 9.3 原子操作解决方案

```cpp
__global__ void histogram_atomic(char *data, int *hist, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&hist[data[i]], 1);
    }
}
```

**原子操作特点**：
- 保证读-修改-写的原子性
- 硬件支持，性能较好
- 但高竞争时仍有串行化

### 9.4 私有化优化

**思想**：每个线程块维护私有直方图，最后合并

```cpp
__global__ void histogram_privatized(char *data, int *hist, int N) {
    __shared__ int private_hist[NUM_BINS];
    
    // 初始化私有直方图
    if (threadIdx.x < NUM_BINS) {
        private_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // 更新私有直方图
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (i < N) {
        atomicAdd(&private_hist[data[i]], 1);
        i += stride;
    }
    __syncthreads();
    
    // 合并到全局直方图
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&hist[threadIdx.x], private_hist[threadIdx.x]);
    }
}
```

### 9.5 聚合优化

**思想**：连续相同值只做一次原子操作

```cpp
// 检查相邻元素是否相同
int prev = (i > 0) ? data[i-1] : -1;
int curr = data[i];
int count = 1;

// 聚合连续相同值
while (i + count < N && data[i + count] == curr) {
    count++;
}
atomicAdd(&hist[curr], count);
```

## 关键要点

| 优化技术 | 效果 | 适用场景 |
|---------|------|---------|
| 原子操作 | 保证正确性 | 基础实现 |
| 私有化 | 减少全局竞争 | 中等 bin 数量 |
| 聚合 | 减少原子操作次数 | 数据有局部性 |

### 性能考虑
- **bin 数量少**：私有化效果好（共享内存放得下）
- **bin 数量多**：考虑部分私有化或直接全局原子
- **数据分布**：均匀分布竞争少，倾斜分布竞争多

### 原子操作硬件支持
- 全局内存原子：L2 缓存执行
- 共享内存原子：SM 内执行，更快
- 现代 GPU 原子操作性能大幅提升
