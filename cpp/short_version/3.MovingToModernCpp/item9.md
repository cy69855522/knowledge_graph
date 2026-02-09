# 条款 9：优先考虑别名声明而非 typedef 🏷️

## 9.1 基础语法对比 ⚖️

`using`（别名声明）和 `typedef` 在功能上完全等价，但 `using` 的可读性通常更好。

```cpp
// 函数指针：fp 是一个指向函数的指针，该函数参数为 int 和 string，无返回值

typedef void (*FP)(int, const std::string&);  // typedef: 名字埋在中间 😵
using FP = void (*)(int, const std::string&); // using: 名字 = 类型，清晰直观 ✅
```

## 9.2 杀手级特性：别名模板 (Alias Templates) 🚀

这是 `using` 完胜 `typedef` 的核心理由。`using` 可以模板化，而 `typedef` 不行。

### 场景：定义一个自定义分配器的链表

**使用 using (C++11):**
```cpp
template<typename T>
using MyAllocList = std::list<T, MyAlloc<T>>; // 简单、直接

MyAllocList<Widget> lw; // 像普通类型一样使用
```

**使用 typedef (C++98):**
必须将其包裹在 `struct` 中，使用起来极其繁琐：
```cpp
template<typename T>
struct MyAllocList {
    typedef std::list<T, MyAlloc<T>> type;
};

MyAllocList<Widget>::type lw; // 必须要加 ::type
```

### 在模板中使用时的痛苦 😖
如果在另一个模板中使用这个类型，`typedef` 版本必须加 `typename`：

```cpp
template<typename T>
class Widget {
private:
    typename MyAllocList<T>::type list; // ❌ 必须加 typename，必须加 ::type
    MyAllocList<T> list;                // ✅ using 版本：干净利落
};
```

## 9.3 Type Traits 的进化 🧬

C++11 的 Type Traits 使用了 `typedef` 实现（因为当时还没有 `using` 普及），导致语法冗长。C++14 利用 `using` 进行了全面升级。

```cpp
// C++11
std::remove_const<T>::type
std::remove_reference<T>::type

// C++14 (后缀 _t)
std::remove_const_t<T>
std::remove_reference_t<T>
```

如果你在用 C++11，可以自己简单封装一下：
```cpp
template <class T>
using remove_const_t = typename std::remove_const<T>::type;
```

## 核心总结 💡

1.  **可读性**：`using` 将名字和类型清晰分离。
2.  **模板支持**：`using` 支持别名模板，`typedef` 不支持。
3.  **减少冗余**：别名模板不需要 `typename` 前缀和 `::type` 后缀。
4.  **C++14 标准**：标准库的 `_t` 后缀 Type Traits 都是基于 `using` 实现的。
