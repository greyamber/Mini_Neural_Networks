# 已实现的API

## Base_Symbol.py
## class Base_Symbol.Variable
对Symbol的包装，仅限训练的变量使用。
### \_\_init\_\_(init, name=None, Trainable=True)

* name：符号名，通过python id使其独有

* init: 初始值,numpy数组或python常数

* Trainable：是否可训练, 如果不可训练，将设置keep_value
### 成员
* value:值，numpy数组或None（未初始化）

* gradiant:梯度，numpy数组或None
### 操作
* +,-，*，/（规则和numpy规则相同）

* Relu等操作

* 矩阵乘法
## class Base_Symbol.Symbol
基础类Symbol

实现前向，反向传播流程等基础功能

未定义非本项目操作对Symbol类的value操作或set_value()操作请使用Variable
### \_\_init\_\_(self, source_list=RootFunction(), name=None, Trainable=False, keep_value=None)
* source_list:该符号的来源，RootFunction代表是直接定义的，非计算得到

* name：符号名，通过python id使其独有

* Trainable：是否可训练

* keep_value：用于训练时的自动feed，见其fp和bp

#### class method:

* fp() ：前向传播，无返回

* bp() ：反向传播，无返回

* set_value(value)：设置该symbol的值

* clear_all()：暂时用于清除梯度和值，会有改动

* gradiant_decent(lr)：根据当前梯度做一次梯度下降

## ActivateFunction.py

#### 1.ReLu(x1)
输入:一个Symbol或其子类
输出：ReLu（x1）

#### 2.Sigmoid(x1)
类似ReLu

#### 3.reduce_sum(x1, x1shape, axis)
暂时用静态方法实现，会改为动态，就不需要
x1shape 参数了。

#### 4.softmax(x1)
类似ReLu,未经详细测试

## layers.py
layers.py中的测试函数大概规定以后框架的用法
### dense(inputs, init_w, init_b=None, activation_func=None)
* inputs: 输入的symbol或其子类
* init_w,init_b：w和b的初始值，如果b为None，不使用偏置
* activation_func：激活函数
* 返回:activation_func(XW + b) or XW + b or XW

## Base_Function.py
* 各个操作的实际实现，详情看Base_Function.py的注释
* 如果必要，可以在这里添加新的自定义操作
