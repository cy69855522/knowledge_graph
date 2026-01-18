# 第三章：多维网格和数据 🗺️

## 3.1 多维网格组织 🏗️

CUDA 网格和块都是**三维**的，通过 `dim3` 类型指定。

### 执行配置 ⚙️
```cpp
dim3 dimBlock(16, 16, 1);  // 每个块 16x16 线程
dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);  // 网格维度
kernel<<<dimGrid, dimBlock>>>(...);
```

### 1D 简写
```cpp
kernel<<<numBlocks, threadsPerBlock>>>(...);  // 等价于 dim3(n, 1, 1)
```

### 内建变量 🔑
| 变量 | 含义 |
|------|------|
| `blockDim.x/y/z` | 块在各维度的线程数 |
| `gridDim.x/y/z` | 网格在各维度的块数 |
| `threadIdx.x/y/z` | 线程在块内的坐标 |
| `blockIdx.x/y/z` | 块在网格内的坐标 |

### 限制 📏
- 每个块最多 **1024** 个线程 🛑
- `gridDim.x`: 最大 $2^{31}-1$
- `gridDim.y/z`: 最大 65535

## 3.2 将线程映射到多维数据 📍

### 2D 映射公式 📐
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 水平方向
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 垂直方向
```

### 示例：处理 76x62 图片，使用 16x16 块 🖼️
- 需要 5x4 = 20 个块（80x64 线程）
- 多余线程通过边界检查过滤 🛡️

### 多维数组线性化（行主序）🔢
```cpp
// 2D 数组 M[j][i] 的线性化访问
int index = j * Width + i;  // M[index]

// 3D 数组
int index = plane * m * n + row * m + col;
```

## 3.3 彩色转灰度 Kernel 🎨

```cpp
__global__ void colorToGrayscaleConversion(unsigned char* Pout, 
                                           unsigned char* Pin,
                                           int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * 3;  // RGB 三通道
        
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        
        Pout[grayOffset] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}
```

**关键点** ✨：
- 每个线程处理一个像素 🎯
- `if (col < width && row < height)` 过滤越界线程 🛡️
- RGB 数据连续存储，每像素 3 字节 💾

## 3.4 图像模糊 Kernel 🌫️

```cpp
#define BLUR_SIZE 1  // 3x3 模糊核

__global__ void blurKernel(unsigned char* Pout, unsigned char* Pin,
                           int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int pixVal = 0;
        int pixels = 0;
        
        // 遍历周围像素块
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                // 边界检查
                if (curRow >= 0 && curRow < height && 
                    curCol >= 0 && curCol < width) {
                    pixVal += Pin[curRow * width + curCol];
                    pixels++;
                }
            }
        }
        Pout[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
```

**边界处理** 🧱：
- 角落像素：只累加 4 个有效像素
- 边缘像素：只累加 6 个有效像素
- 内部像素：累加全部 9 个像素

## 3.5 矩阵乘法 Kernel ⭐

矩阵乘法 $C = A \times B$：$C_{i,j} = \sum_k A_{i,k} \cdot B_{k,j}$ 🔢

```cpp
__global__ void matrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; k++) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}
```

### 内存访问模式 💾
```
M[row][k] → M[row * Width + k]  // 同一行，连续访问 (Row-major)
N[k][col] → N[k * Width + col]  // 同一列，跨行访问 (Stride = Width)
```

### 调用示例 🚀
```cpp
dim3 dimBlock(16, 16);
dim3 dimGrid(ceil(Width/16.0), ceil(Width/16.0));
matrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
```

## 3.6 核心总结 💡

| 概念 | 要点 |
|------|------|
| **网格/块维度** | 最多 3D，通过 `dim3` 指定 🗺️ |
| **线程映射** | `blockIdx * blockDim + threadIdx` 📍 |
| **数组线性化** | 行主序：`row * Width + col` 🔢 |
| **边界检查** | `if (row < height && col < width)` 🛡️ |
| **矩阵乘法** | 每线程计算一个输出元素 🎯 |
