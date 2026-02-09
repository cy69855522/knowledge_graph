# 条款 10：优先考虑限域 enum 而非未限域 enum 🛡️

## 10.1 命名空间污染 (Namespace Pollution) 🌫️

C++98 的 `enum`（未限域）会将枚举名泄露到外部作用域。

```cpp
enum Color { black, white, red }; // black, white, red 泄露到外部
auto white = false;               // ❌ 错误！white 已经声明过了
```

C++11 的 `enum class`（限域）将名字限制在枚举内。

```cpp
enum class Color { black, white, red };
auto white = false;        // ✅ 没问题
Color c = Color::white;    // ✅ 必须使用 Color:: 限定符
```

## 10.2 强类型安全 (Strong Typing) 🔒

未限域 `enum` 会隐式转换为整数，导致荒谬的比较。

```cpp
enum Color { red, green, blue };
Color c = red;
if (c < 14.5) { ... } // ❌ 允许！Color 被隐式转为 int，再转为 double
```

限域 `enum` **禁止隐式转换**。

```cpp
enum class Color { red, green, blue };
Color c = Color::red;
if (c < 14.5) { ... } // ✅ 编译错误！禁止比较
```

如果确实需要转换，必须使用 `static_cast`，这让代码意图更明确。

## 10.3 前置声明 (Forward Declaration) ⏩

C++98 `enum` 默认无法前置声明，因为编译器不知道底层类型大小（取决于枚举值的范围）。这导致任何枚举值的修改都会引发包含该头文件的所有文件重新编译。

C++11 `enum class` 的底层类型默认是 `int`，大小固定，因此**总是支持前置声明**。

```cpp
enum class Status; // ✅ 前置声明
void process(Status s); // 只需要声明，无需包含完整定义
```

*注：C++11 也允许为普通 `enum` 指定底层类型来实现前置声明：`enum Color : int;`*

## 10.4 唯一反例：std::tuple 的索引 ⚠️

当使用 `std::tuple` 时，普通 `enum` 的隐式转换反而很方便。

```cpp
using UserInfo = std::tuple<std::string, std::string, std::size_t>;
enum UserInfoFields { uiName, uiEmail, uiReputation };

UserInfo uInfo;
auto val = std::get<uiEmail>(uInfo); // ✅ 自动转换为 size_t
```

如果使用 `enum class`，则需要写成冗长的 `std::get<static_cast<size_t>(UserInfoFields::uiEmail)>(uInfo)`。为此，可以编写一个 `toUType` 模板函数来辅助转换。

## 核心总结 💡

1.  **减少污染**：`enum class` 不会泄露名字到外部作用域 🌫️。
2.  **类型安全**：禁止隐式转换为整数，防止逻辑错误 🔒。
3.  **编译优化**：支持前置声明，减少编译依赖 ⏩。
4.  **默认类型**：底层类型默认为 `int`。
