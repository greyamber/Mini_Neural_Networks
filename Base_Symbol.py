# coding: utf-8
from abc import ABCMeta, abstractmethod
from Base_Function import *
import ActivateFunctions as AF
import Environment as env
import numpy as np


class Symbol(object):
    """
    基础类Symbol
    未定义非本项目操作对Symbol类的value操作或set_value()操作
    请使用Variable
    构造：
    source_list:该符号的来源，RootFunction代表是直接定义的，非计算得到
    name：符号名，通过python id使其独有
    Trainable： 是否可训练
    keep_value： 用于训练时的自动feed，见其fp和bp.不建议使用，仅用于框架内部
    成员：
    value：值
    gradient：梯度
    clear_all()：暂时用来清除旧gradient和value的方法，以后会有大的改动
    set_value(): 设置Symbol的value，主要考虑了是否是root function产生的符号
    fp（）：前向传播
    bp（）：反向传播
    push_grad()：暂存梯度的部分（和），用于bp（）
    """
    def __init__(self, source_list=RootFunction(), name=None, Trainable=False):
        if name is None:
            self.name = "unknown_name_" + str(id(self))
        else:
            self.name = name + "_" + str(id(self))

        self.Trainable = Trainable

        env.ALL_COLLECTION.append(self)
        if Trainable and (self not in env.TRAINABLE_COLLECTION):
            env.TRAINABLE_COLLECTION.append(self)
        if source_list.func_type == "Root" and (self not in env.ROOT_COLLECTION):
            env.ROOT_COLLECTION.append(self)

        self.value = None
        self.source_list = source_list  # 记录生成轨迹
        self.gradient = None
        self.gradient_collection = []

    def __add__(self, other):  # 只实现一个接口，实际在Base_Function.py中
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Addition(self, other), name="add")

    def __sub__(self, other):
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Subtraction(self, other), name="sub")

    def __mul__(self, other):
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Multiplication(self, other), name="mul")

    def __truediv__(self, other):
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Division(self, other), name="div")

    def __radd__(self, lhs):
        if not isinstance(lhs, Symbol):
            value = lhs
            lhs = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            lhs.set_value(value)
        return Symbol(Addition(lhs, self), name="add")

    def __rsub__(self, lhs):
        if not isinstance(lhs, Symbol):
            value = lhs
            lhs = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            lhs.set_value(value)
        return Symbol(Subtraction(lhs, self), name="sub")

    def __rmul__(self, lhs):
        if not isinstance(lhs, Symbol):
            value = lhs
            lhs = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            lhs.set_value(value)
        return Symbol(Multiplication(lhs, self), name="mul")

    def __rtruediv__(self, lhs):
        if not isinstance(lhs, Symbol):
            value = lhs
            lhs = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            lhs.set_value(value)
        return Symbol(Division(lhs, self), name="div")

    def push_gradient(self, grad):
        self.gradient = grad
        self.gradient_collection.append(grad)

    def set_value(self, value):
        self.value = np.array(value)
        if self.source_list.func_type == "Root" or self.source_list.func_type == "Constant":
            self.source_list.value = value

    @staticmethod
    def clear_all():  # 原递归算法，现改逻辑，不再从bp起始点开始
        # self.gradient_collection = []
        # self.gradient = None
        # if (self.source_list.func_type != "Constant") and (not self.Trainable) and (self not in env.VARIABLE_COLLECTION):
        #     self.value = None
        # x1 = self.source_list.x1
        # x2 = self.source_list.x2
        # if x1 is not None:
        #     x1.clear_all()
        # if x2 is not None:
        #     x2.clear_all()
        ##################################################################################
        for av in env.ALL_COLLECTION:
            av.gradient_collection = []
            av.gradient = None
            if av.source_list.func_type != "Constant":
                if (not av.Trainable) and (av not in env.VARIABLE_COLLECTION):
                    av.value = None

    def _fp(self):  # 递归算法，改成循环
        # x1, x2 = self.source_list.x1, self.source_list.x2
        # if x1.value is None:
        #     x1._fp()
        # if x2 is not None and x2.value is None:
        #     x2._fp()
        # self.value = self.source_list.fp_method()
        ##################################################
        pointer = self
        stack = []
        while True:
            x1, x2 = pointer.source_list.x1, pointer.source_list.x2
            if x1.value is None:
                stack.append(pointer)
                pointer = x1
            elif x2 is not None and x2.value is None:
                stack.append(pointer)
                pointer = x2
            else:
                pointer.value = pointer.source_list.fp_method()
                if len(stack) > 0:
                    pointer = stack.pop()
                else:
                    break

    def fp(self):
        for rs in env.ROOT_COLLECTION:  # 以后会消除该行, 只保留placeholder的检查
            assert rs.value is not None, env.FeedError("Placeholder " + str(rs.name) + " has no value.")
        for rs in env.PLACEHOLDER_COLLECTION:
            assert rs.value is not None, env.FeedError("Placeholder " + str(rs.name) + " has no value.")
        self._fp()

    def _bp(self, gradient):  # 递归算法，改成循环
        # x1 = self.source_list.x1
        # x2 = self.source_list.x2
        # x1g, x2g = self.source_list.bp_method(gradient)
        # if x1 is not None:
        #     x1.push_gradient(x1g)
        #     x1._bp(x1g)
        # if x2 is not None:
        #     x2.push_gradient(x2g)
        #     x2._bp(x2g)
        ################################################
        stack = []
        pointer = self
        while True:
            x1 = pointer.source_list.x1
            x2 = pointer.source_list.x2
            x1g, x2g = pointer.source_list.bp_method(gradient)
            if x1 is not None:
                gradient = x1g
                pointer = x1
                pointer.push_gradient(gradient)
                if x2 is not None:
                    stack.append([x2, x2g])
            else:
                if len(stack) > 0:
                    pointer, gradient = stack.pop()
                    pointer.push_gradient(gradient)
                else:
                    break

    @staticmethod
    def _done_collect(collect_all=True):  # 原递归算法，现改逻辑，不再从bp起始点开始
        # self.gradient = np.sum(np.array(self.gradient_collection), 0)
        # x1 = self.source_list.x1
        # x2 = self.source_list.x2
        # if x1 is not None:
        #     x1._done_collect()
        # if x2 is not None:
        #     x2._done_collect()
        #############################################################
        if collect_all:
            for av in env.ALL_COLLECTION:
                if av.gradient is not None:
                    np.sum(np.array(av.gradient_collection), 0)
        else:
            for av in env.TRAINABLE_COLLECTION:
                if av.gradient is not None:
                    np.sum(np.array(av.gradient_collection), 0)

    def bp(self, grad=None, collect_all=True):
        # 如果collect_all，则所有梯度，否则只收集可训练参数的梯度
        # 并不会改善太大的性能
        if self.value is None:
            self.fp()
        if grad is None:
            grad = np.ones(self.value.shape)
        self.push_gradient(grad)

        self._bp(grad)
        self._done_collect(collect_all=collect_all)

    @staticmethod
    def gradient_decent(lr=0.01):  # 原递归算法，现改逻辑，不再从bp起始点开始
        # if self.Trainable:
        #     self.set_value(self.value - self.gradient * lr)
        # x1 = self.source_list.x1
        # x2 = self.source_list.x2
        # if x1 is not None:
        #     x1.gradient_decent(lr)
        # if x2 is not None:
        #     x2.gradient_decent(lr)

        ##################################################
        for tv in env.TRAINABLE_COLLECTION:
            if tv.gradient is not None:
                tv.set_value(tv.value - tv.gradient * lr)


class Variable(Symbol):
    """
    对Symbol的包装，变量使用。
    clear时不清除Variable的值

    构造：
    name：符号名，通过python id使其独有
    Trainable： 是否可训练
    成员：
    value：值
    gradient：梯度
    clear_all()：暂时用来清除旧gradient和value的方法，以后会有大的改动
    set_value(): 设置Symbol的value，主要考虑了是否是root function产生的符号
    fp（）：前向传播
    bp（）：反向传播
    push_grad()：暂存梯度的部分（和），用于bp（）
    """
    def __init__(self, init, name=None, Trainable=True):
        super(Variable, self).__init__(source_list=RootFunction(), name=name,
                                       Trainable=Trainable)
        self.set_value(init)
        env.VARIABLE_COLLECTION.append(self)

    def assign(self, value):
        self.set_value(value)
        return [self, value]


class Placeholder(Symbol):
    def __init__(self, name=None):
        super(Placeholder, self).__init__(source_list=RootFunction(), name=name,
                                          Trainable=False)
        env.PLACEHOLDER_COLLECTION.append(self)

    def feed(self, value):
        self.set_value(value)


class Constant(Symbol):
    def __init__(self, value, name=None):
        super(Constant, self).__init__(source_list=ConstantFunction(), name=name,
                                       Trainable=False)
        self.set_value(value)
        env.CONSTANT_COLLECTION.append(self)
