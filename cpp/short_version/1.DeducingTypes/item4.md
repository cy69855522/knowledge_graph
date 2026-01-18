# 条款 4：学会查看类型推导结果 🔍

了解编译器到底推导出了什么类型，是学习 C++ 类型推导的关键。我们可以从开发过程中的三个阶段来获取信息。

## 4.1 IDE 编辑器阶段 💻

在大多数现代 IDE 中，只需将**鼠标悬停**在变量、函数或参数名上，即可查看到推导出的类型。

- **适用场景**：简单的 `auto` 或模板推导。
- **局限性**：当类型极其复杂（如涉及长长的 STL 内部类型）时，IDE 提供的信息可能由于过度简化或格式混乱而难以理解。

## 4.2 编译阶段：TD 诊断大法 🛠️

利用编译器报错来显式打印类型。我们可以声明一个**不定义**的类模板：

```cpp
template<typename T>
class TD; // TD == "Type Displayer"
```

通过实例化该模板引发编译错误：
```cpp
int x = 27;
const int* y = &x;

TD<decltype(x)> xType; // 触发错误：TD<int> 不完整
TD<decltype(y)> yType; // 触发错误：TD<const int*> 不完整
```
**优点**：几乎所有编译器都会在错误信息中准确写出 `TD` 实例化时的具体类型。

## 4.3 运行时阶段：typeid vs Boost 🚀

### 1. std::typeid (不推荐 ❌)
使用 `typeid(var).name()` 可以打印类型，但存在两个严重问题：
1. **不可信**：根据规范，它会像**传值推导**一样剥离引用和 `const/volatile` 修饰符。
2. **格式难读**：GNU/Clang 可能会输出 `PKi`（代表 `pointer to const int`），需要 `c++filt` 转换。

### 2. Boost.TypeIndex (推荐 ✅)
如果你需要精确且可读的运行时类型信息，**Boost.TypeIndex** 是最佳选择：

```cpp
#include <boost/type_index.hpp>

template<typename T>
void f(const T& param) {
    using boost::typeindex::type_id_with_cvr;

    // 显示 T 和 param 的精确类型
    std::cout << "T =     " << type_id_with_cvr<T>().pretty_name() << "\n";
    std::cout << "param = " << type_id_with_cvr<decltype(param)>().pretty_name() << "\n";
}
```

**对比示例**：
对于 `const T& param`，如果传入左值：
- `typeid` 可能会错误地报告 `T` 和 `param` 类型相同。
- `Boost` 会准确报告：`T` 为 `Widget const *`，`param` 为 `Widget const * const &`。

## 核心总结 💡

1. **IDE**：最快但可能不准。
2. **TD 声明**：最精准的“编译期打印”。
3. **Boost.TypeIndex**：最完美的运行时输出 🚀。
4. **终极秘籍**：工具只是辅助，深入理解条款 1-3 的推导规则才是根本 🧠。
