# 第二章：异构数据并行计算

## 2.1 数据并行性

**数据并行 (Data parallelism)**：在数据集的不同部分上执行的计算可以独立并行执行。

**示例：彩色转灰度**

$$
L = 0.21r + 0.72g + 0.07b
$$

每个像素的计算独立，可并行执行——这就是数据并行性的基础。

## 2.2 CUDA C 程序结构

CUDA 程序执行流程：
1. **主机代码** (CPU) 开始执行
2. 调用**内核函数 (Kernel)** 在设备 (GPU) 上启动**线程网格 (Grid)**
3. 网格执行完毕后返回主机继续执行

```
Host Code → Kernel Launch → Grid Execution → Host Code → ...
```

## 2.3 向量加法内核

### 传统 C 代码
```cpp
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}
```

### CUDA 并行化结构
```
Part 1: 分配设备内存，拷贝数据到设备
Part 2: 调用 Kernel
Part 3: 拷贝结果回主机，释放设备内存
```

## 2.4 设备全局内存和数据传输

### 内存分配与释放
```cpp
float *A_d;
cudaMalloc((void**)&A_d, size);  // 分配设备内存
cudaFree(A_d);                   // 释放设备内存
```

### 数据传输
```cpp
cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);  // Host → Device
cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);  // Device → Host
```

### 完整示例
```cpp
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;
    
    // Part 1: 分配设备内存并传输数据
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    
    // Part 2: 调用 Kernel (下节介绍)
    
    // Part 3: 传输结果并释放内存
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}
```

### 错误检查
```cpp
cudaError_t err = cudaMalloc((void**)&A_d, size);
if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
}
```

## 2.5 核函数和线程

### 线程组织：两级层次结构
- **Grid (网格)**：由多个 Block 组成
- **Block (块)**：由多个 Thread 组成（最多 1024 个线程）

### 内建变量
| 变量 | 含义 |
|------|------|
| `threadIdx.x` | 线程在块内的索引 |
| `blockIdx.x` | 块在网格内的索引 |
| `blockDim.x` | 每个块的线程数 |
| `gridDim.x` | 网格中的块数 |

### 计算全局索引
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**示例**：blockDim = 256
- Block 0: i = 0~255
- Block 1: i = 256~511
- Block 2: i = 512~767

### 向量加法 Kernel
```cpp
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

**关键点**：
- `__global__` 声明这是一个 Kernel 函数
- `if (i < n)` 处理线程数不是块大小整数倍的情况
- 每个线程处理一个元素，**循环被线程网格替代**

### 函数声明关键字
| 关键字 | 执行位置 | 调用位置 |
|--------|----------|----------|
| `__global__` | Device | Host (或 Device，动态并行) |
| `__device__` | Device | Device |
| `__host__` | Host | Host |

## 2.6 调用核函数

### 执行配置语法
```cpp
kernel<<<numBlocks, threadsPerBlock>>>(args...);
```

### 向量加法调用
```cpp
int threadsPerBlock = 256;
int numBlocks = ceil(n / 256.0);  // 向上取整
vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
```

**示例**：n = 1000
- numBlocks = ceil(1000/256) = 4
- 总线程数 = 4 × 256 = 1024
- 前 1000 个线程执行加法，后 24 个被 `if (i < n)` 过滤

## 2.7 编译

```bash
nvcc program.cu -o program
```

NVCC 编译器：
1. 分离主机代码和设备代码
2. 主机代码 → 标准 C/C++ 编译器
3. 设备代码 → PTX (中间表示) → GPU 目标代码

## 2.8 核心总结

| 概念 | API/语法 |
|------|----------|
| 内存分配 | `cudaMalloc()` |
| 内存释放 | `cudaFree()` |
| 数据传输 | `cudaMemcpy()` |
| Kernel 声明 | `__global__` |
| Kernel 调用 | `<<<blocks, threads>>>` |
| 线程索引 | `threadIdx`, `blockIdx`, `blockDim` |
