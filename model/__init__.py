# -*- coding:utf-8 -*-
"""
Project: Out-of-Scope-Intent-Detection-with-Supervised-Deep-Metric-Learning
File: __init__.py
Author: wangxudong
Create date: 2023/5/29 2:30 下午
Description: 
"""
from .bert import BERT, AdvBERT

backbones_map = {
                    'bert': BERT,
                    'bert_adv': AdvBERT
                }

if __name__ == "__main__":
    pass
