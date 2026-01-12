# 条款 1：理解模板类型推导

理解模板类型推导是掌握 `auto` 和 `decltype` 的基础。在模板调用中，编译器利用实参（`expr`）推导两个类型：一个是模板参数 `T`，另一个是函数形参类型 `ParamType`。

```cpp
template<typename T>
void f(ParamType param);

f(expr); // 从 expr 推导 T 和 ParamType
```

这两者通常不同，因为 `ParamType` 往往包含 `const` 或引用修饰符。例如：
```cpp
template<typename T>
void f(const T& param);

int x = 0;
f(x); // T 推导为 int，ParamType 推导为 const int&
```

类型推导规则分为三种情况：

## 情景 1：ParamType 是指针或引用（但非万能引用）

这是最简单的情况，推导遵循以下原则：
1. 如果 `expr` 是引用，忽略引用部分。
2. 将 `expr` 的类型与 `ParamType` 进行模式匹配以确定 `T`。

```cpp
template<typename T>
void f(T& param);

int x = 27;
const int cx = x;
const int& rx = x;

f(x);  // T 是 int, param 类型是 int&
f(cx); // T 是 const int, param 类型是 const int&
f(rx); // T 是 const int, param 类型是 const int&
```

**原理：** 当传递 `const` 对象给引用形参时，调用者期望对象保持不可变。因此，实参的常量性（`const`ness）会被保留并成为 `T` 的一部分。即使 `rx` 是引用，`T` 也会被推导为非引用，因为引用性在匹配前被忽略。

如果 `ParamType` 声明为 `const T&`：
```cpp
template<typename T>
void f(const T& param);

f(cx); // T 是 int, param 类型是 const int&
```
由于 `ParamType` 本身已包含 `const`，`T` 无需再包含它。

## 情景 2：ParamType 是万能引用（T&&）

万能引用的推导规则在处理左值时具有特殊性：
1. 如果 `expr` 是**左值**，`T` 和 `ParamType` 都会被推导为**左值引用**。这是模板推导中 `T` 被推导为引用的唯一情况。
2. 如果 `expr` 是**右值**，则应用情景 1 的常规规则。

```cpp
template<typename T>
void f(T&& param);

int x = 27;
const int cx = x;

f(x);  // x 是左值，T 是 int&，param 类型是 int&
f(cx); // cx 是左值，T 是 const int&，param 类型是 const int&
f(27); // 27 是右值，T 是 int，param 类型是 int&&
```

**原理：** 万能引用旨在区分左值和右值实参，从而支持完美转发。当传入左值时，推导结果会“坍缩”为左值引用。

## 情景 3：ParamType 既不是指针也不是引用（传值）

传值意味着 `param` 是实参的一个完全独立的拷贝：
1. 如果 `expr` 是引用，忽略引用部分。
2. 忽略 `const` 和 `volatile` 修饰符。

```cpp
int x = 27;
const int cx = x;

f(cx); // T 和 param 的类型都是 int
```

**原理：** 既然 `param` 是一个新对象，它的修改不会影响原始对象。因此，原始对象是否为 `const` 与新拷贝能否被修改无关。

**特殊情况：指向常量的常量指针**
```cpp
const char* const ptr = "Hello";
f(ptr); // T 和 param 的类型是 const char*
```
这里 `ptr` 自身的常量性（指针不能指向别处）被忽略，但它指向的数据的常量性被保留。

## 数组与函数实参

### 数组退化
在传值语境下，数组类型会退化为指向其首元素的指针：
```cpp
const char name[] = "Briggs"; // const char[7]
f(name); // T 推导为 const char*
```

但是，如果形参声明为数组的引用，则数组**不会退化**，且推导出的类型包含大小：
```cpp
template<typename T>
void f(T& param);

f(name); // T 推导为 const char[7], param 类型为 const char(&)[7]
```

**应用示例：编译期获取数组大小**
```cpp
template<typename T, std::size_t N>
constexpr std::size_t arraySize(T (&)[N]) noexcept {
    return N;
}
```

### 函数退化
函数名同样会退化为函数指针，除非用于初始化引用：
```cpp
void someFunc(int, double);

template<typename T> void f1(T param);
f1(someFunc); // T 为 void(*)(int, double)

template<typename T> void f2(T& param);
f2(someFunc); // T 为 void(&)(int, double)
```

## 核心总结
1. **引用忽略**：推导时实参的引用性被忽略。
2. **左值识别**：万能引用在接收左值时，会将 `T` 推导为左值引用。
3. **常量剥离**：传值推导会剥离 `const` 和 `volatile`。
4. **数组/函数退化**：传值时退化为指针，传引用时保留原始类型（含数组长度）。
