#!/usr/bin/python
# -*- coding: utf-8 -*-


# 一些基本配置
DB_TYPE = 'train' # test为连接测试数据库,  其他值表示连接生产库
PORT = "50070"
ONE_DAY_IN_SECONDS = 60 * 60 * 24
GENDER_COEFF = 0.95

#建模期
LIMIT_T_MODEL_BUILD = 30*2

#胎动有效特征
T_LIMIT_QUICKENING = 30*60 # Second

#模型参数
MODEL_VERSION = 'V0'
MODEL_PATH = f'model/{MODEL_VERSION}/model/'

#生理周期参数
CYCLE_MAX = 35
CYCLE_MIN = 18
KEEP_MAX = 15
KEEP_MIN = 3