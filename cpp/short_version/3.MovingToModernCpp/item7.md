# 条款 7：区别使用 () 和 {} 创建对象 ⚖️

C++11 引入了统一的**花括号初始化**（Uniform Initialization），旨在解决初始化语法的混乱。

## 7.1 为什么使用花括号初始化？ 🌟

### 1. 适用性最广
花括号几乎可以在所有上下文中使用，而 `()` 和 `=` 则有局限性。

```cpp
std::vector<int> v{ 1, 3, 5 };  // 初始化容器元素：唯一可行语法
std::atomic<int> ai{ 0 };       // 初始化不可拷贝对象：= 不可用
class Widget { int x{ 0 }; };   // 类成员默认初始化：() 不可用
```

### 2. 禁止变窄转换 (Narrowing Conversion) 🛡️
花括号不允许精度的隐式丢失，更加安全。

```cpp
double x, y, z;
int sum1{ x + y + z }; // ❌ 错误！禁止变窄转换
int sum2(x + y + z);   // ⚠️ 允许，但会截断
```

### 3. 免疫“最令人头疼的解析” (Most Vexing Parse) 💉
C++ 规定：任何能被解析为声明的东西，必须解析为声明。

```cpp
Widget w1(); // ⚠️ 声明了一个返回 Widget 的函数！
Widget w2{}; // ✅ 调用默认构造函数创建对象
```

## 7.2 花括号初始化的巨大陷阱：std::initializer_list 🕳️

如果在构造函数中加入了 `std::initializer_list` 重载，编译器会**极度倾向**于使用它，哪怕其他构造函数看起来更匹配。

```cpp
class Widget {
public:
    Widget(int i, bool b);      // ctor 1
    Widget(int i, double d);    // ctor 2
    Widget(std::initializer_list<long double> il); // ctor 3
};

Widget w1(10, true); // 调用 ctor 1
Widget w2{10, true}; // ⚠️ 强行调用 ctor 3 (int/bool -> long double)
```

### 只有在完全无法转换时，才会回退
```cpp
// 如果 ctor 3 改为 std::initializer_list<std::string>
Widget w3{10, true}; // ✅ 无法将 int/bool 转为 string，回退调用 ctor 1
```

### 空花括号的特例
```cpp
Widget w4{};   // ✅ 调用默认构造函数 (无参)
Widget w5{{}}; // ✅ 调用 std::initializer_list 构造函数 (空列表)
```

## 7.3 std::vector 的经典案例 📚

```cpp
std::vector<int> v1(10, 20); // 10 个元素，每个都是 20
std::vector<int> v2{10, 20}; // 2 个元素：10 和 20
```

## 核心总结 💡

1.  **推荐默认使用花括号** `{}`：因为它安全（禁止变窄）、通用、无歧义。
2.  **例外情况**：
    *   当需要调用非 `std::initializer_list` 构造函数，而该类又有 `std::initializer_list` 重载时（如 `std::vector` 的大小构造），**必须**使用圆括号 `()`。
    *   在模板中创建对象时要格外小心，因为你不知道 `T` 是否有 `std::initializer_list` 构造函数。
