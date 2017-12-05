# coding: utf-8
from abc import ABCMeta, abstractmethod
from Base_Function import *
import ActivateFunctions as AF
import Environment as env
import numpy as np


class Symbol(object):
    def __init__(self, source_list=RootFunction(), name=None, Trainable=False, keep_value=None):
        self.Trainable = Trainable
        if source_list.func_type == "Root" and keep_value is None:
            env.ROOT_COLLECTION.append(self)
        if keep_value is not None:
            env.AUTO_COLLECTION.append([self, keep_value])
        if name is None:
            self.name = "unknown_name_" + str(id(self))
        else:
            self.name = name + "_" + str(id(self))

        self.value = None
        self.source_list = source_list  # 记录生成轨迹
        self.gradient = None
        self.gradient_collection = []

    # 和说好的不一样啊。。。
    @staticmethod
    def _constant_wrapper(func):
        def wrapper(other):
            if not isinstance(other, Symbol):
                value = other
                other = Symbol()
                other.set_value(value)
                func(other)
        return wrapper

    def __add__(self, other):
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Addition(self, other), name=None)

    def __sub__(self, other):
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Subtraction(self, other), name=None)

    def __mul__(self, other):
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Multiplication(self, other), name=None)

    def __truediv__(self, other):
        if not isinstance(other, Symbol):
            value = other
            other = Symbol(ConstantFunction(), name="Constant", Trainable=False)
            other.set_value(value)
        return Symbol(Division(self, other), name=None)

    def push_gradient(self, grad):
        self.gradient = grad
        self.gradient_collection.append(grad)

    def set_value(self, value):
        self.value = np.array(value)
        if self.source_list.func_type == "Root" or self.source_list.func_type == "Constant":
            self.source_list.value = value

    def clear_all(self):  # 递归算法，将改成循环
        self.gradient_collection = []
        self.gradient = None
        if self.source_list.func_type != "Constant" and (not self.Trainable):
            self.value = None
        x1 = self.source_list.x1
        x2 = self.source_list.x2
        if x1 is not None:
            x1.clear_all()
        if x2 is not None:
            x2.clear_all()

    def _fp(self):  # 递归算法，将改成循环
        x1, x2 = self.source_list.x1, self.source_list.x2
        if x1.value is None:
            x1._fp()
        if x2 is not None and x2.value is None:
            x2._fp()
        self.value = self.source_list.fp_method()

    def fp(self):
        for item in env.AUTO_COLLECTION:
            item[0].set_value(item[1])
        for rs in env.ROOT_COLLECTION:
            assert rs.value is not None, "Root symbol " + str(rs.name) + " has no value." + str(rs)
        self._fp()

    def _bp(self, gradient):  # 递归算法，将改成循环
        x1 = self.source_list.x1
        x2 = self.source_list.x2
        x1g, x2g = self.source_list.bp_method(gradient)
        if x1 is not None:
            x1.push_gradient(x1g)
            x1._bp(x1g)
        if x2 is not None:
            x2.push_gradient(x2g)
            x2._bp(x2g)

    def _done_collect(self):  # 递归算法，将改成循环
        self.gradient = np.sum(np.array(self.gradient_collection), 0)
        x1 = self.source_list.x1
        x2 = self.source_list.x2
        if x1 is not None:
            x1._done_collect()
        if x2 is not None:
            x2._done_collect()

    def bp(self):
        if self.value is None:
            self.fp()
        grad = np.ones(self.value.shape)
        self.push_gradient(grad)

        self._bp(grad)
        self._done_collect()

    def gradient_decent(self, lr=0.01):
        if self.Trainable:
            self.set_value(self.value - self.gradient * lr)
        x1 = self.source_list.x1
        x2 = self.source_list.x2
        if x1 is not None:
            x1.gradient_decent(lr)
        if x2 is not None:
            x2.gradient_decent(lr)

