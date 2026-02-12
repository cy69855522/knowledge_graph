import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# 1. Define a simple model (Conv + BN + ReLU)
# This is a classic candidate for fusion.
class SimpleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def run_profiler(model, x, tag):
    print(f"Running Profiler for {tag}...")
    # Warmup
    # 为什么要 Warmup?
    # 1. CUDA Kernel 首次执行时会有初始化开销 (Context Creation, Kernel Loading)。
    # 2. PyTorch 的 Allocator (缓存分配器) 首次运行时需要向 OS 申请显存。
    # 3. torch.compile 首次运行需要编译 (Compilation Overhead)。
    # 如果不 Warmup，第一次运行的时间会极其长，污染平均值。
    for _ in range(5):
        _ = model(x)
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                _ = model(x)
                
    prof.export_chrome_trace(f"{tag}_trace.json")
    print(f"✅ Saved trace to {tag}_trace.json. Open in https://ui.perfetto.dev/ to view.")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    x = torch.randn(8, 64, 128, 128).to(device)
    model = SimpleBlock().to(device).eval()

    # 1. Run Eager Mode (No Compile)
    run_profiler(model, x, "eager")

    # 2. Run Compiled Mode (Fusion)
    # torch.compile uses Inductor backend which performs fusion
    if torch.cuda.is_available():
        print("\nCompiling model with torch.compile...")
        # 这一步做了很多事情：
        # 1. Graph Capture (Dynamo): 把上面的 Python 代码变成计算图 (FX Graph)。
        # 2. Optimization (Inductor): 分析图，发现 Conv+BN+ReLU 可以融合。
        # 3. Code Gen (Triton): 生成一个融合后的 Triton Kernel (triton_moi...)。
        compiled_model = torch.compile(model)
        run_profiler(compiled_model, x, "compiled")
    else:
        print("\nSkipping compile test (requires CUDA/Trition usually for best results)")
