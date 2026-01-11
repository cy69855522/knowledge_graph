# Effective Modern C++

- [X] 简介

## 类型推导

- [ ] Item 1:理解模板类型推导
- [ ] Item 2:理解auto类型推导
- [ ] Item 3:理解decltype
- [ ] Item 4:学会查看类型推导结果

## auto

- [ ] Item 5:优先考虑auto而非显式类型声明
- [ ] Item 6:auto推导若非己愿，使用显式类型初始化惯用法

## 移步现代C++

- [ ] Item 7:区别使用()和{}创建对象
- [ ] Item 8:优先考虑nullptr而非0和NULL
- [ ] Item 9:优先考虑别名声明而非typedefs
- [ ] Item 10:优先考虑限域枚举而非未限域枚举
- [ ] Item 11:优先考虑使用deleted函数而非使用未定义的私有声明
- [ ] Item 12:使用override声明重写函数
- [ ] Item 13:优先考虑const_iterator而非iterator
- [ ] Item 14:如果函数不抛出异常请使用noexcept
- [ ] Item 15:尽可能的使用constexpr
- [ ] Item 16:让const成员函数线程安全
- [ ] Item 17:理解特殊成员函数的生成

## 智能指针

- [ ] Item 18:对于独占资源使用std::unique_ptr
- [ ] Item 19:对于共享资源使用std::shared_ptr
- [ ] Item 20:当std::shared_ptr可能悬空时使用std::weak_ptr
- [ ] Item 21:优先考虑使用std::make_unique和std::make_shared，而非直接使用new
- [ ] Item 22:当使用Pimpl惯用法，请在实现文件中定义特殊成员函数

## 右值引用，移动语义，完美转发

- [ ] Item 23:理解std::move和std::forward
- [ ] Item 24:区别通用引用和右值引用
- [ ] Item 25:对于右值引用使用std::move，对于通用引用使用std::forward
- [ ] Item 26:避免重载通用引用
- [ ] Item 27:熟悉重载通用引用的替代品
- [ ] Item 28:理解引用折叠
- [ ] Item 29:认识移动操作的缺点
- [ ] Item 30:熟悉完美转发失败的情况

## Lambda表达式

- [ ] Item 31:避免使用默认捕获模式
- [ ] Item 32:使用初始化捕获来移动对象到闭包中
- [ ] Item 33:对于std::forward的auto&&形参使用decltype
- [ ] Item 34:优先考虑lambda表达式而非std::bind

## 并发API

- [ ] Item 35:优先考虑基于任务的编程而非基于线程的编程
- [ ] Item 36:如果有异步的必要请指定std::launch::async
- [ ] Item 37:从各个方面使得std::threads unjoinable
- [ ] Item 38:关注不同线程句柄析构行为
- [ ] Item 39:考虑对于单次事件通信使用void
- [ ] Item 40:对于并发使用std::atomic，volatile用于特殊内存区

## 微调

- [ ] Item 41:对于那些可移动总是被拷贝的形参使用传值方式
- [ ] Item 42:考虑就地创建而非插入
