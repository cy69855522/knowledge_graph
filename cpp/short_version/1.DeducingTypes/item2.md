# 条款 2：理解 auto 类型推导

`auto` 类型推导与模板类型推导几乎完全相同，只有**一个例外**。

## 核心映射

`auto` 扮演模板中的 `T`，变量的类型说明符扮演 `ParamType`：

```cpp
auto x = 27;           // 相当于 T param = 27;        → x 是 int
const auto cx = x;     // 相当于 const T param = x;   → cx 是 const int
const auto& rx = x;    // 相当于 const T& param = x;  → rx 是 const int&
```

## 三个情景（与模板推导一致）

```cpp
auto x = 27;           // 情景 3：传值，x 是 int
const auto cx = x;     // 情景 3：传值，cx 是 const int
const auto& rx = x;    // 情景 1：引用，rx 是 const int&

auto&& uref1 = x;      // 情景 2：万能引用，x 是左值 → int&
auto&& uref2 = 27;     // 情景 2：万能引用，27 是右值 → int&&
```

数组和函数退化规则同样适用：
```cpp
const char name[] = "Briggs";
auto arr1 = name;      // const char*（退化）
auto& arr2 = name;     // const char(&)[7]（不退化）
```

## 唯一例外：花括号初始化

**`auto` 会将花括号推导为 `std::initializer_list`，但模板不会：**

```cpp
auto x1 = 27;          // int
auto x2(27);           // int
auto x3 = { 27 };      // std::initializer_list<int>  ⚠️
auto x4{ 27 };         // std::initializer_list<int>  ⚠️

// 模板无法推导花括号
template<typename T>
void f(T param);
f({ 1, 2, 3 });        // ❌ 错误！无法推导
```

## C++14 注意事项

在函数返回值和 lambda 形参中使用 `auto` 时，采用的是**模板类型推导规则**（不是 auto 规则），因此**不支持花括号**：

```cpp
auto createList() {
    return { 1, 2, 3 };     // ❌ 错误！
}

auto resetV = [](const auto& val) { /*...*/ };
resetV({ 1, 2, 3 });        // ❌ 错误！
```

## 核心总结

1. **auto 推导 ≈ 模板推导**：三个情景规则完全相同
2. **唯一例外**：`auto` 将 `{}` 推导为 `std::initializer_list`，模板不会
3. **C++14 陷阱**：函数返回值和 lambda 中的 `auto` 使用模板规则，不支持 `{}`
