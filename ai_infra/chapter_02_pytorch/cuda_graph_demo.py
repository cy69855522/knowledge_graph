import torch
import time

def verify_cuda_graph():
    if not torch.cuda.is_available():
        print("Skipping CUDA Graph verification (CUDA not available)")
        return

    print("Verifying CUDA Graph implementation...")
    
    # 1. 准备数据 (Static Input)
    N, D_in, D_out = 64, 1024, 1024
    # 必须是 CUDA tensor
    x = torch.randn(N, D_in, device='cuda')
    y = torch.randn(N, D_out, device='cuda')
    model = torch.nn.Linear(D_in, D_out).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # 2. Warmup (热身) - 极其重要！让 Pytorch 分配显存
    # 同时也作为一个 side_stream，避免干扰默认流
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # 3. Capture (录制) 🎥
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True) # Important: call outside capture

    # Capture context
    with torch.cuda.graph(g):
        y_pred = model(x)
        loss = (y_pred - y).pow(2).sum()
        loss.backward()
        optimizer.step()

    # 4. Replay (回放) 🎬
    print("Graph captured! Replaying...")
    start = time.time()
    # 减少迭代次数，避免跑太久
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize() # 等待 GPU 跑完
    print(f"10 iter finished in {time.time() - start:.4f}s")
    print("Verification SUCCESS! 🎉")

if __name__ == "__main__":
    try:
        verify_cuda_graph()
    except Exception as e:
        print(f"Verification FAILED: {e}")
        exit(1)
