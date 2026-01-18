# 条款 5：优先考虑 auto 而非显式类型声明 ✨

使用 `auto` 不仅仅是为了少打几个字，它在正确性、性能和可维护性上都有显著优势。

## 5.1 强制初始化 🛡️

`auto` 变量必须有初始化表达式，这从根本上杜绝了“只声明不初始化”导致的未定义行为。

```cpp
int x1;         // ⚠️ 潜在的未初始化变量
auto x2;        // ❌ 错误！必须初始化
auto x3 = 0;    // ✅ 安全
```

## 5.2 简化复杂类型 🏗️

处理迭代器或模板相关的长类型时，`auto` 让代码更整洁。

```cpp
template<typename It>
void dwim(It b, It e) {
    while (b != e) {
        auto currValue = *b; // 无需写冗长的 typename std::iterator_traits<It>::value_type
        // ...
    }
}
```

## 5.3 保存闭包（Lambda） 🚀

### auto vs std::function
*   **auto**：保存的变量类型与闭包一致，占用空间极小，调用速度快。
*   **std::function**：模板化的函数包装器，通常在堆上分配内存，有额外的调用开销，且可能抛出 OOM 异常。

```cpp
// ✅ 推荐：直接使用 auto
auto derefLess = [](const auto& p1, const auto& p2) { return *p1 < *p2; };

// ❌ 不推荐：语法冗长且效率较低
std::function<bool(const std::unique_ptr<Widget>&, const std::unique_ptr<Widget>&)> 
func = [](const std::unique_ptr<Widget>& p1, const std::unique_ptr<Widget>& p2) { return *p1 < *p2; };
```

## 5.4 避免“类型快捷方式”陷阱 🔀

### 1. 容器大小
```cpp
std::vector<int> v;
unsigned sz = v.size(); // ⚠️ 在 64 位系统上，size_type 是 64 位，unsigned 是 32 位
auto sz = v.size();     // ✅ 类型始终匹配
```

### 2. std::unordered_map 的 Key
`unordered_map` 的键类型其实是 `const Key`。
```cpp
std::unordered_map<std::string, int> m;

// ❌ 效率低下：会产生临时对象拷贝，因为 map 里的类型是 std::pair<const std::string, int>
for (const std::pair<std::string, int>& p : m) { ... }

// ✅ 高效：直接绑定引用，无拷贝
for (const auto& p : m) { ... }
```

## 核心总结 💡

1.  **安全**：强制初始化 🛡️。
2.  **高效**：避免不必要的类型转换和临时对象 🚀。
3.  **灵活**：重构更方便，代码更简洁 ✨。
4.  **注意**：小心 `auto` 的推导陷阱（见条款 2 & 6）⚠️。
