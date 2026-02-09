# 简介

- `&` 是左值引用（绑定已有对象），`&&` 是右值引用（绑定临时对象以实现移动语义），两者是独立且不同的语法标记。

# Item 1：理解模板类型推导

- 万能引用 + std::forward = 完美转发 = 避免拷贝 = 高性能

- **万能引用（`T&&`）**：在模板中接收左值和右值，左值推导为 `T&`，右值推导为 `T`，通过引用坍缩保留原始身份，配合 `std::forward` 实现完美转发。其他都是模式匹配。

    ``` cpp
    // 完美转发的例子，保持左右值特性
    template<typename T>
    void good_wrapper(T&& param) {
        process(std::forward<T>(param)); // 根据 T 的真实身份转发
    }

    int x = 10;
    good_wrapper(x);   // T = int&，转发为左值，输出：int&
    good_wrapper(27);  // T = int，转发为右值，输出：int
    ```

- **std::forward**: `std::forward<T>` 根据 `T` 的身份保留原始参数的左值/右，这是完美转发的关键：值特性。

# Item 2: 理解auto类型推导
- 在 auto 推导中，花括号会被特殊处理为 std::initializer_list，这是与模板推导的唯一区别。🎯

# Item 3: 理解decltype
- `decltype(auto)` 用于完美推导并保留函数返回值的类型（如引用性），但要注意 `return (x);` 会返回引用的陷阱。🔍

# Item 4: 学会查看类型推导结果
- 善用 IDE 提示、编译器错误信息（TD法）和 `Boost.TypeIndex` 来准确查看推导出的类型，别轻信 `typeid`。👀

# Item 5: 优先考虑auto而非显式类型声明
- `auto` 能够强制初始化、避免因类型不匹配导致的隐式转换和性能损耗，并能简洁地表示复杂类型。✨

# Item 6: auto推导若非己愿，使用显式类型初始化惯用法
- 遇到 `std::vector<bool>` 等代理类（Proxy Class）时，使用 `static_cast` 明确告诉 `auto` 你真正想要的类型。🛠️

# Item 7: 区别使用()和{}创建对象
- 默认优先使用花括号初始化 `{}`（因为它禁止变窄转换且无歧义），但在涉及 `std::initializer_list` 构造函数时要格外小心。⚖️

# Item 8: 优先考虑nullptr而非0和NULL
- `nullptr` 是真正的指针类型，避免了 `0` 和 `NULL` 在重载决议和模板推导中被误认为是整数的尴尬。🚫

# Item 9: 优先考虑别名声明而非typedef
- 使用 `using` 定义别名不仅可读性更强（函数指针更明显），而且支持**别名模板**（Alias Templates），彻底告别 `typedef` 时代的 `typename ... ::type` 冗余。🏷️

# Item 10: 优先考虑限域enum而非未限域enum
- `enum class`（限域枚举）通过禁止隐式转为整数和限定作用域，解决了命名污染和类型安全问题，且默认底层类型为 `int`，支持前置声明。🛡️
