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