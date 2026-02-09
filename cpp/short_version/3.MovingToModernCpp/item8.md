# 条款 8：优先考虑 nullptr 而非 0 和 NULL 🚫

## 8.1 0 和 NULL 的本质问题 🧩

字面值 `0` 是 `int`，`NULL` 通常被宏定义为 `0` 或 `0L`（也是整数）。它们都不是指针类型，只是在某些上下文被“勉强”解释为指针。

这就导致了重载决议时的二义性或错误匹配：

```cpp
void f(int);
void f(bool);
void f(void*);

f(0);    // 调用 f(int)，而不是 f(void*)
f(NULL); // 可能不通过编译，或者调用 f(int)
```

## 8.2 nullptr 的优势 ✨

`nullptr` 是 `std::nullptr_t` 类型的关键字，它可以**隐式转换**为任何类型的指针，但**不能**转换为整数。

```cpp
f(nullptr); // ✅ 明确调用 f(void*)
```

## 8.3 模板中的致命陷阱 🧨

在模板推导中，`0` 和 `NULL` 会被推导为 `int` 或 `long`，导致类型不匹配。

为了演示这个问题，我们需要先定义一些辅助工具（互斥量和锁）：

```cpp
// 1. 定义辅助类型和变量
using MuxGuard = std::lock_guard<std::mutex>; // 自动加锁/解锁的守卫
std::mutex f1m, f2m, f3m;                     // 对应函数的互斥锁 (f1m -> f1 mutex)

// 2. 只能被互斥量锁住调用的函数
int f1(std::shared_ptr<Widget> spw);
double f2(std::unique_ptr<Widget> upw);
bool f3(Widget* pw);

// 3. 通用调用模板：先加锁，再调用函数
template<typename Func, typename Mux, typename Ptr>
auto lockAndCall(Func func, Mux& mutex, Ptr ptr) -> decltype(func(ptr)) {
    MuxGuard g(mutex); // 构造时加锁，析构时解锁
    return func(ptr);
}

// 4. 调用测试
lockAndCall(f1, f1m, 0);       // ❌ 错误！0 被推导为 int，传递给 f1 失败
lockAndCall(f2, f2m, NULL);    // ❌ 错误！NULL 被推导为 int，传递给 f2 失败
lockAndCall(f3, f3m, nullptr); // ✅ 正确！nullptr 推导为 std::nullptr_t，隐式转为 Widget*
```

## 核心总结 💡

1.  **类型安全**：`nullptr` 拥有专属类型 `std::nullptr_t`，是指针而非整数。
2.  **消除歧义**：避免在重载决议中误调用整型版本。
3.  **模板友好**：在模板类型推导中表现正确，不会变成 `int`。
4.  **清晰表意**：`if (ptr == nullptr)` 比 `if (ptr == 0)` 更直观地表明我们在检查指针。
