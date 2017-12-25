# coding: utf-8
import numpy as np


class FeedError(Exception):
    def __init__(self, massage):
        self.massage = massage


ALL_COLLECTION = []  # 所有Symbol和其子类
ROOT_COLLECTION = []  # 由RootFunction构造的Symbol
CONSTANT_COLLECTION = []  # 所有常量
TRAINABLE_COLLECTION = []  # 可训练的Symbol
PLACEHOLDER_COLLECTION = []  # 占位符
VARIABLE_COLLECTION = []  # 变量
