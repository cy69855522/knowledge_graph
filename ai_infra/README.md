# 🗺️ AI Infra 转型实战指南 (CV 工程师版) 🚀

**目标:** 从 CV (Accuracy 导向) 转型为 AI Infra (Latency/Throughput 导向)。
**当前进度:** Chapter 1-3 Completed ✅

---

## 👣 学习步骤

- 提前准备第二天要看的视频链接
- 去图书馆
- 找视频观看相关资料
- 让AI根据标题生成课程还有课后作业
- 阅读AI教程

---

## ⚠️ 注意力 (Attention)

注意力涣散是一个极其典型的**“多巴胺戒断反应”** +  **“后台进程占用过高”** 。

### 🛠️ 步骤一：外挂缓存 (External Cache) —— 清理后台进程

你的大脑之所以不断跳出“别的事情”，是因为它害怕你忘记这些重要信息。这是进化的本能（Zeigarnik Effect，未完成任务会一直占用内存）。

* **操作：** 准备一张白纸和一支笔（不要用手机，手机是干扰源）。
* **协议：**
  1. 看书时，一旦脑子里跳出任何杂念（比如“要不要买个升降桌”、“那个期权税怎么算”），**立刻**把它记在纸上。
  2. 告诉大脑：“已归档（Acked）。我也许一小时后处理，也许明天处理，但现在你不需要再提醒我了。”
  3. **神奇效果：** 一旦写下来，大脑会认为任务已“被接管”，后台进程会自动 Kill 掉，你会瞬间感觉脑子清爽了。

### 🔋 步骤二：降低启动功耗 (Low-Power Boot) —— 视频预热

直接啃 DeepSpeed 的 Paper（文字+数学公式）是  **高能耗任务** 。当你注意力涣散时，强行启动高能耗任务会直接导致  **宕机（犯困/玩手机）** 。

* **操作：** 不要读 Paper。**去看视频。**
* **资源：** 打开 B站 或 YouTube，搜 **“DeepSpeed ZeRO 详解”** 或  **“Megatron-LM 图解”** 。
* **心态：** “我就当看个科普动画片，不需要背下来。”
* **原理：** 视频是 **被动输入** ，消耗极低。等看了 10 分钟视频，你的大脑对这些名词（Optimizer States, Gradients）有了 **热缓存 (Warm Cache)** ，这时候再切回 PDF 看文字，摩擦力会降低 90%。

### 🧬 步骤三：生物黑客特技 —— 40Hz 双耳节拍 (Binaural Beats)

这是神经科学界公认能提升专注力的“作弊码”。

* **操作：** 戴上耳机（降噪模式开启）。
* **资源：** 搜索 **"40Hz Binaural Beats"** 或  **"Focus Music"** 。
* **原理：** 40Hz 是大脑在高度专注（Gamma 波）时的频率。通过听觉引导，强制让你的脑波与这个频率共振（Entrainment）。
* **注意：** 不需要很大声，作为背景噪音即可。这能帮你屏蔽掉海浪声带来的困意。

### ⏱️ 步骤四：番茄钟的“微缩版” —— 10分钟冲刺

你觉得“进度推得慢”，是因为你潜意识里觉得“这一章要看 2 小时”，太痛苦了。

* **操作：** 拿出手机倒计时。**只设 10 分钟。**
* **自我欺骗：** “我只看 10 分钟 ZeRO-Stage 1 的图。10 分钟一到，如果我想停，我就立刻去阳台看海。”
* **结果：** 只要你开始了这 10 分钟，多巴胺就会开始分泌（Task Initiation）。通常 10 分钟后你根本不想停，因为你已经进入了 **心流 (Flow)** 的门槛。

---

### 💊 最后的“作弊”补剂 (Biohack Stack)

既然你是 Biohacker，如果你手头有补剂，现在可以吃：

* **NatureBell 胞磷胆碱 (Citicoline) + L-酪氨酸**

  * **为什么：**

    * **Citicoline：** 它提升注意力的效果比咖啡因更稳、更持久，而且 **它不缩血管，反而改善脑供血** （对 HCT 高的人极度友好）。
    * **不伤胃：** 它对胃黏膜没有刺激。
    * **无焦虑：** 它不会升高皮质醇。
  * **操作：**

    * 复习前吃  **1 粒胞磷胆碱** 。这就够了。你根本不需要咖啡因的抖动感。

---

### 🛡️ 给你的心理按摩

**进度慢一点也没关系。**

你要搞懂的是  **Infra 的底层逻辑** ，而不是去背诵全文。

哪怕你一下午只搞懂了一张图（比如 ZeRO 是怎么切分参数的），这一下午也是**价值连城**的。因为面试官只会问你核心原理，不会问你第几页写的什么。

**现在，执行步骤一：**

拿张纸，把刚才让你分心的念头写下来。

然后，打开 B站，搜一个 DeepSpeed 的视频，瘫在椅子上看。

---

### 🏆 燥热降温


#### 方案 A：物理冷却法 (The "Cold Plunge" Protocol) —— 最快

* **原理：** 利用“哺乳动物潜水反射”。冷水会瞬间激活交感神经，释放 **去甲肾上腺素 (Norepinephrine)** 。这种激素会立刻抑制冲动，并大幅提升专注力。
* **操作：**
  * 走进卫生间。
  * 用**最冷的水**洗一把脸，或者直接冲个冷水澡（30秒即可）。
  * **效果：** 你的欲望会被“冻住”，大脑瞬间清醒，非常适合继续看书。

#### 方案 B：血流重定向 (Blood Flow Redirection) —— 最健康

* **原理：** 冲动本质上是血液流向问题。通过大肌群运动，强行把血液抢到大腿和背部去。
* **操作：**
  * 做 **20 个深蹲 (Squats)** 或  **10 个波比跳 (Burpees)** 。
  * 做到心跳加速、轻微喘气的程度。
* **效果：** 此时身体进入“战斗/逃跑”模式，性欲会自然消退。而且运动产生的多巴胺能替代，帮你维持学习状态。


---

## 📚 目录 (Table of Contents)

### [第一章：基础概念 (Fundamentals)](./chapter_01_fundamentals/)

* [1.1 显存墙 (Memory Wall) &amp; Roofline Model](./chapter_01_fundamentals/1.1_memory_wall.md) 🧱
* [1.2 计算密集 (Compute Bound) vs 访存密集 (Memory Bound)](./chapter_01_fundamentals/1.2_compute_vs_memory.md) 🐢🐇
* [1.3 延迟 (Latency) vs 吞吐 (Throughput)](./chapter_01_fundamentals/1.3_latency_vs_throughput.md) ⏱️🌊

### [第二章：PyTorch 进阶 (PyTorch Enterprise)](./chapter_02_pytorch/)

* [2.1 Tensor 内存布局 (Stride &amp; View)](./chapter_02_pytorch/2.1_tensor_memory.md) 📐
* [2.2 算子融合 (Operator Fusion)](./chapter_02_pytorch/2.2_operator_fusion.md) 🔗
* [2.3 NCCL 通信原语 (Communication Primitives)](./chapter_02_pytorch/2.3_nccl_communication.md) 📞
* [2.4 PyTorch 2.0 &amp; torch.compile (Inductor)](./chapter_02_pytorch/2.4_torch_compile.md) 🛠️

### [第三章：vLLM &amp; LMcache (Efficient Serving)](./chapter_03_vllm/)

* [3.1 KV Cache (Key-Value Cache)](./chapter_03_vllm/3.1_kv_cache.md) 🔑
* [3.2 PagedAttention (The Core Magic)](./chapter_03_vllm/3.2_paged_attention.md) 🪄
* [3.3 Continuous Batching (持续批处理)](./chapter_03_vllm/3.3_continuous_batching.md) 🌊
* [3.4 LMcache (KV Cache Offloading)](./chapter_03_vllm/3.4_lmcache.md) 💾

---

## 🏃‍♀️ 下一步 (Next Steps)

准备好之后，我们将进入 **第四章 (TBA)**！
