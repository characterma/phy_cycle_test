import datetime as dt
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from email import message
import time
from concurrent import futures
import os
import json
import threading
from scipy import signal

import grpc
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from tensorflow import keras as k
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.model_gender import MAX, MIN, Net_gender_ppg
from model.model_hr import Net_gender_hr

from serv import iandun_pb2, iandun_pb2_grpc
from utils import log
from utils.andun_database import Mongo, Sql
from utils.config import *
from utils.women import Women
import warnings
from numba import cuda

# 解决cuda报错问题
ansql = Sql("pre")
ansql_test = Sql("test")
warnings.filterwarnings('ignore')
anlog = log.MyLogger('./log/server.log',level='info')
scaler_standard = StandardScaler()
women = Women()
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='测试线程')
#pool = ProcessPoolExecutor(max_workers=2)
model_uids = []

def prob2label(nparray):
    unique,counts = np.unique(nparray,return_counts=True)
    ind_max = np.where(counts==np.max(counts))
    return unique[ind_max]

def cdays_wear(id):
    return
def get_score(x,l,vs,o_index,o_end_index):
    if l[vs.index(x)]==0:
        return 0
    elif l[vs.index(x)]==1 or l[vs.index(x)]==3:
        return 10 if abs(vs.index(x) -l.index(2))<=2 or abs(vs.index(x)-o_end_index)<=2 else 8
    elif l[vs.index(x)]==2:
        bias = abs(o_index-vs.index(x))*10
        score = 90-bias
        return 60 if score<60 else score
    elif l[vs.index(x)]==4:
        return 90
    pass
def get_ovulation_index(i,x):
    if x==2:
        return i
def async_train_model(id,model_dir_name,bind_time):
    if id in model_uids:
        anlog.logger.info(f"任务已存在{id}任务列表{model_uids}")
        return
    model_uids.append(id)
    pool.submit(train_model_v2,id,model_dir_name,bind_time)
    anlog.logger.info(f"任务添加完成{id}任务列表{model_uids}")

    pass
def extractor_ppg(ppg_oneday):
    try:
        ppgs = ppg_oneday.split(';')
        res = []
        for ppg in ppgs:
            if int(ppg[:6]) <= 40000:
                data = np.fromstring(ppg[7:],float,sep=',')
                data = data.reshape(-1, 12)
                for j in data:
                    if True in np.isnan(j):
                        continue
                    res.append(j.tolist())
            else:
                break
        if len(res) > 0: return res
    except:
        return None

def extractor_hr(hr_oneday):
        try:
            n_limit_hr = 128
            hr_oneday = hr_oneday.replace(';', ',')
            np_hr_oneday = np.fromstring(hr_oneday, int, sep=',').reshape(-1, 2)
            res = []
            for i in np_hr_oneday:
                if i[0] <= 60000 and len(res) < n_limit_hr:
                    data = i[1]
                    res.append(data)
                else:
                    break
            if len(res) >= n_limit_hr:
                return res[:n_limit_hr]
        except:
            return

def cal_diff_t(t,t1):
    return

def cal_quickening(hr):
    res = []
    for i in range(len(hr)):
        try:
            if hr[i+1] > hr[i] and hr[i+1] > hr[i+2]:res.append(i+1)
        except:
            continue
    return res

# 判断某个用户的模型是否存在
def exist_model(uid):
    fp = f''' {MODEL_PATH}{uid} '''
    if os.path.exists(fp):
        return True
    return False

def try_fun(str_exec,str_eval,x):
    try:
        exec(str_exec)
        return eval(str_eval)
    except Exception as e:
        # print(e)
        return None

# 对PPG进行滤波
def filt_data(data):

    b,a = signal.butter(2,[0.00390625,0.01171875],'bandpass')
    filt = signal.filtfilt(b,a,data)
    return filt

# 训练个人的生理周期模型
def train_model_personal(uid):
    sql = f''' select * from andun_health.h_pregnant_analysis where wear_user_id='{uid}' '''
    df_status = ansql.ansql_read_mysql(sql)
    df_status.loc[df_status.status != 0,'status'] = 1

    did = ansql.did_uid(uid)

    with Mongo() as mongo:
        df_ppg = mongo.women_ppg_did(did)

    # 打标签和文本转数组
    df_ppg['createTime'] = df_ppg.createTime.apply(lambda x:x+dt.timedelta(hours=8))
    df_ppg['ct_date'] = df_ppg.createTime.apply(lambda x:x.date())
    for i in range(df_status.shape[0]):
        date = df_status.date.iloc[i]
        status = df_status.status.iloc[i]
        df_ppg.loc[df_ppg.ct_date == date,'label'] = status
    df_ppg.dropna(inplace=True)
    df_ppg['PPG_Green'] = df_ppg['PPG_Green'].apply(lambda x:np.fromstring(x[1:-1],float,sep=','))

    # 样本和标签
    ppgs = []
    labels = []
    ppgs_bfilt = df_ppg.PPG_Green.tolist()
    labels_bfilt = df_ppg.label.tolist()
    for i in range(len(ppgs_bfilt)):
        ppg = ppgs_bfilt[i]
        lab = labels_bfilt[i]
        if len(ppg) >= 11240:
            ppgs.append(filt_data(ppg))
            labels.append(lab)

    # 样本裁剪
    ppgs_cut = [ppg[500:10740] for ppg in ppgs]
    ppgs_np = np.array(ppgs_cut)
    ppgs_np = (ppgs_np - 1e5) * 50 / (2e6 - 1e5)

    labels_np = np.array(labels)

    # 样本量平衡
    counts_1 = (labels_np == 1).sum()
    counts_0 = labels_np.size - counts_1
    ppgs_np_repeat = ppgs_np[labels_np == 1].repeat(counts_0 // counts_1,0)
    labels_np = np.concatenate([labels_np,labels_np[labels_np==1].repeat(counts_0 // counts_1,0)])
    ppgs_np_repeat = np.concatenate((ppgs_np,ppgs_np_repeat))

    #切分训练，验证集
    labels_np = tf.one_hot(labels_np,2).numpy()

    # 判断训练数据的数量，少于100不训练
    if ppgs_np_repeat.shape[0] >= 100:
        anlog.logger.warn("{} 训练ppg数据大于100, 数据量{}  训练模型...".format(uid, ppgs_np_repeat.shape[0]))

        x_train,x_test,y_train,y_test = train_test_split(ppgs_np_repeat,labels_np,test_size=0.2)

        # 建模训练
        layers = k.layers
        model = k.Sequential([
            layers.Reshape((2560,4),input_shape=(10240,)),
            layers.Conv1D(16,256,128,activation='relu',kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Conv1D(8,4,activation='relu',kernel_initializer='normal'),
            layers.BatchNormalization(),
            # layers.Conv1D(4,2,activation='relu'),
            # layers.Conv1D(2,2,activation='relu'),
            layers.Flatten(),
            # layers.Dense(1024,activation='relu'),
            layers.Dense(256,activation='relu',kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Dense(128,activation='relu',kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Dense(64,activation='relu',kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Dense(2,activation='softmax')
        ])

        model.compile(
            k.optimizers.Adam(1e-3),
            loss = k.losses.BinaryCrossentropy(),
            metrics = ['accuracy']
            )
        #model.compile(k.optimizers.SGD(1e-5),loss=k.losses.BinaryCrossentropy(),metrics = ['accuracy'])
        early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', patience=3,restore_best_weights=True,mode='min')

        h = model.fit(
        x_train,
        y_train,
        batch_size=30,
        epochs=500,
        validation_data=(x_test,y_test),
        callbacks = [early_stopping]
        )

        # 训练可视化
        #training_vis(h)

        if model.evaluate(x_test,y_test)[1]>= 0.7:
            model.save(f'model/model_ppg_{uid}')
            anlog.logger.info("{} 训练ppg模型完成...保存model/model_ppg_{}".format(uid, uid))
        else:
            anlog.logger.warn("{} 训练ppg模型完成...模型精度小于0.7,不保存".format(uid))
    else:
        anlog.logger.warn("{} 训练ppg数据少于100, 数据量{}  不训练模型...".format(uid, ppgs_np_repeat.shape[0]))

def init():
    return k.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
'''
优化版本
'''
def model_predict_2(uid, date,all,temperture=False,model_dir_name=None):
    did = ansql.did_uid(uid)
    #did = "610812111000022"
    if model_dir_name is None:
        model_dir_name = uid
    if all:
        model = k.models.load_model(f'model_mbi/model_ppg_micro_{model_dir_name}')
    else:
        model = k.models.load_model(f'model_green_2/model_ppg_micro_{model_dir_name}')
    date = dt.datetime.combine(date, dt.time())
    t0 = date - dt.timedelta(hours=8)
    #t1 = date + dt.timedelta(hours=2)
    t1 = date - dt.timedelta(hours=2)
    with Mongo() as mongo:
        df_ppg = mongo.women_all_ppg_did_date(did, t0, t1)
    if df_ppg.empty:
        return None
    df_ppg['date'] = df_ppg.createTime.apply(lambda x: date.date())
    if temperture:
        get_women_temperature(df_ppg,uid)
        df_ppg.hr = df_ppg.hr.apply(lambda x:x/100.0)
        df_ppg.age = df_ppg.age.apply(lambda x: x / 100.0)
        df_ppg.sleep = df_ppg.sleep.apply(lambda x: x / 21600.0)
        df_ppg.temperature = df_ppg.temperature.apply(lambda x: x / 37.0)
        df_ppg.mbi = df_ppg.mbi.apply(lambda x:x/30.0)

        m_data = df_ppg[["hr", "mbi", "age", "medical_history", "temperature", "sleep"]]
        mean_data = np.nanmean(m_data, axis=0)

        df_ppg = df_ppg.fillna(
            {"hr": mean_data[0], "mbi": mean_data[1], "age": mean_data[2], "medical_history": mean_data[3],
             "temperature": mean_data[4], "sleep": mean_data[5]}, axis=0, inplace=False)
        df_ppg = df_ppg.fillna(0)

    drop_index = []
    if temperture:
        person_data = df_ppg[["hr", "mbi", "age", "medical_history", "temperature", "sleep"]]
    if all:
        df_ppg['PPG_Green'] = df_ppg['PPG_Green'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
        df_ppg['PPG_Red'] = df_ppg['PPG_Red'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
        df_ppg['PPG_IR'] = df_ppg['PPG_IR'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
        # 样本和标签
        ppgs = []
        ppgrs = []
        ppgirs = []
        labels = []

        ppgs_rfilt = df_ppg.PPG_Red.tolist()
        ppgs_irfilt = df_ppg.PPG_IR.tolist()
        ppgs_bfilt = df_ppg.PPG_Green.tolist()
        #labels_bfilt = df_ppg.label.tolist()
        for i in range(len(ppgs_bfilt)):
            ppg = ppgs_bfilt[i]
            ppgr = ppgs_rfilt[i]
            ppgir = ppgs_irfilt[i]
            #lab = labels_bfilt[i]
            if len(ppg) >= 11240:
                ppgs.append(filt_data(ppg))
                #labels.append(lab)
            else:
                drop_index.append(i)
            if len(ppgr) >= 11240:
                ppgrs.append(filt_data(ppgr))
            if len(ppgir) >= 11240:
                ppgirs.append(filt_data(ppgir))
        # 样本裁剪
        ppgs_cut = [ppg[500:10740] for ppg in ppgs]
        ppgs_cut1 = [ppg[500:10740] for ppg in ppgrs]
        ppgs_cut2 = [ppg[500:10740] for ppg in ppgirs]
        ppgs_np = np.array(ppgs_cut)
        ppgs_np1 = np.array(ppgs_cut1)
        ppgs_np2 = np.array(ppgs_cut2)

        ppgs_np = np.concatenate([ppgs_np, ppgs_np1, ppgs_np2], axis=1)

    else:
        df_ppg['ppg'] = df_ppg['PPG_Green'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
        ppgs_bfilt = df_ppg.ppg.tolist()

        ppgs = []
        for i in range(len(ppgs_bfilt)):
            ppg = ppgs_bfilt[i]
            if len(ppg) >= 11240:
                ppgs.append(filt_data(ppg))
            else:
                drop_index.append(i)

        ppgs_cut = [ppg[500:10740] for ppg in ppgs]
        ppgs_np = np.array(ppgs_cut)



    ppgs_np = (ppgs_np - 1e5) * 50 / (2e6 - 1e5)
    if temperture:
        person_data = person_data.drop(person_data.index[drop_index])
        ppgs_np = np.concatenate([ppgs_np, person_data], axis=1)
        ppgs_np = ppgs_np[:, 6:]

    res_model = model.predict(ppgs_np, batch_size=ppgs_np.shape[0])
    res_model = res_model[:, 0].sum() / res_model.shape[0]

    return res_model
'''
根据uid获取config
'''
def get_config(uid,dbtype="test"):
    if dbtype =="test":
        ansql_test = Sql("test")
    else:
        ansql_test = ansql
    sql = f"SELECT * FROM andun_health.`h_prengnant_config` where wear_user_id = '{uid}'"
    config = ansql_test.ansql_read_mysql(sql)
    if config.empty:
        return None
    else:
        return config
    pass
def update_config(uid,men_start_lastest,men_keep,men_cycle,near_seven_data,dbtype="test"):
    if dbtype == "test":
        ansql_test = Sql("test")
    else:
        ansql_test = ansql
    ansql_test.h_prengnant_config_update(uid,men_start_lastest,men_keep,men_cycle,near_seven_data)
    pass
def get_model_result_by_config(uid,date,res_model,res_formula,men_start_lastest,men_keep,men_cycle,status_yesterday,ind,dbtype="online",bind_time=None):
    # 根据用户个人模型配置 分析生理周期结果
    if res_model is None:
        if status_yesterday == 2 and res_formula == 1:
            return status_yesterday
        if status_yesterday == 3 and res_formula == 1:
            return status_yesterday
        if status_yesterday == 0 and res_formula == 3:
            return 1
        if status_yesterday == 1 and (res_formula == 0 or res_formula == 3 or res_formula == 4):
            return 1
        if status_yesterday == 2 and (res_formula == 0 or res_formula == 1):
            return status_yesterday
        if status_yesterday == 3 and res_formula == 2:
            return status_yesterday
        if status_yesterday == 3 and res_formula == 4:
            return status_yesterday
        if status_yesterday == 4 and res_formula != 2:
            return 2
        return res_formula
    config = get_config(uid,dbtype)

    ratio = 0.5
    ratio_array = []
    min_ratio = 0.146
    config_last_start = None
    model_result = 1
    if config is None or config.empty:
        # 未配置个人信息进行基础判断
        diff_config_days = (date - men_start_lastest).days
        if res_model >= ratio:
            if diff_config_days < men_keep:
                model_result = 0
            elif abs(diff_config_days - men_keep) <= 3:
                men_keep += 1
                model_result = 0
            elif abs(men_cycle - diff_config_days) <= 7:
                config_last_start = date
                model_result = 0
        else:
            # 异常因素波动导出识别错误
            if diff_config_days>0 and diff_config_days<men_keep and diff_config_days <= 0.5*men_keep:
                model_result = 0
            # 分析七天情况进行 判断概率情况
            if (diff_config_days>men_cycle or (diff_config_days<men_keep and diff_config_days>0)) and res_model>min_ratio:
                model_result = 0
            model_result = int(res_model <= ratio) if model_result!=0 else model_result
        # 查询上次为0 的date
        last_data = ansql.ansql_read_mysql(f"SELECT * FROM andun_health.`h_pregnant_analysis` where wear_user_id ='{uid}' and date<'{date}' and date>='{men_start_lastest}' and date>='{bind_time.date()}' and `status`=0 ORDER BY date asc limit 1")
        if not last_data.empty:
            data_men_start = last_data.date.iloc[0]
            data_diff = (date - data_men_start).days
            #data_diff = (men_start_lastest-data_men_start).days
            diff_config_days = data_diff
        if model_result == 1 and abs(diff_config_days - men_keep) >= 2:
            mens_list_config = []
            if diff_config_days<men_keep and diff_config_days>0:
                men_keep = diff_config_days
            if diff_config_days==0:
                diff_config_days = -1
            mens_list_config = mens_config_list(men_keep, men_cycle, mens_list_config)
            if diff_config_days<0:
                diff_config_days = men_cycle + diff_config_days
            men_index = abs(diff_config_days) % len(mens_list_config)
            model_result = mens_list_config[men_index]
            if status_yesterday ==2 and model_result ==1:
                return status_yesterday
            if status_yesterday ==3 and model_result ==1:
                return status_yesterday
            if status_yesterday ==0 and (model_result ==3 or model_result==2 or model_result==4):
                return 1
            if status_yesterday ==1 and (model_result ==0 or model_result==3 or model_result==4):
                return 1
            if status_yesterday ==2 and (model_result ==0 or model_result==1):
                return status_yesterday
            if status_yesterday ==3 and model_result ==2:
                return status_yesterday
            if status_yesterday ==3 and model_result ==4:
                return status_yesterday
            if status_yesterday ==4 and model_result!=2:
                return 2

        return model_result


    if not config.empty:
        ratio = config.ratio.iloc[0]
        min_ratio = config.min_ratio.iloc[0]
        history = config.near_seven_data.iloc[0]
        if history:
            ratio_array = str(history).split(",")
        config_last_start = config.men_start_lastest.iloc[0]
    # 根据获取的配置进行判断
    # if res_model >= ratio and ind<=men_keep or (men_cycle-ind)<=men_keep:
    #     return 0
    diff_days = (config_last_start - men_start_lastest).days
    ratio_array.append(date.strftime('%Y-%m-%d'))
    ratio_array.append(str(res_model))
    # 针对两个时间差进行处理一致的话不予更新

    if diff_days == 0:
        if res_model >= ratio and (ind<=men_keep or (men_cycle-ind)<=men_keep):
            model_result = 0

    else:
        # 有时间差用config
        #now_time = dt.datetime.now()
        diff_config_days = (date - config_last_start).days
        if res_model >= ratio:
            if diff_config_days<men_keep:
                model_result =  0
            elif abs(diff_config_days-men_keep) <= 3:
                men_keep += 1
                model_result = 0
            elif abs(men_cycle-diff_config_days) <= 7:
                config_last_start = date
                model_result = 0
        else:
            # 异常因素波动导出识别错误
            if diff_config_days>0 and diff_config_days<men_keep and diff_config_days <= 0.5*men_keep:
                model_result = 0
            # 分析七天情况进行 判断概率情况
            if diff_config_days>men_cycle and res_model>min_ratio:
                config_last_start = date
                model_result = 0
            model_result = int(res_model <= ratio) if model_result!=0 else model_result
        # 根据分析的结果进行入库更新
        near_seven_data = ratio_array[2:] if len(ratio_array)>14 else ratio_array
        near_seven_data = ",".join(near_seven_data)
        update_config(uid, config_last_start, men_keep, men_cycle, near_seven_data, dbtype)
    if model_result==1 and abs(diff_config_days-men_keep)>=2:
        mens_list_config = []
        mens_list_config = mens_config_list(men_keep, men_cycle, mens_list_config)
        men_index = abs(diff_config_days) % len(mens_list_config)
        model_result = mens_list_config[men_index]
    return model_result


        # if res_model>=ratio and abs(diff_config_days-men_keep)<=3:
        #     men_keep += 1
        #     return 0
        # elif res_model>=min_ratio and (diff_config_days-men_keep)<0.5*men_keep:
        #     return 0
        # elif res_model>=ratio and abs(men_cycle-diff_config_days)<=men_keep:
        #     config_last_start = date
        #     return 0

    pass


def mens_config_list(men_keep=6, men_cycle=28, mens_list_official=[]):
    mens_list_official.extend([0] * men_keep)
    while men_keep>=7 and men_cycle-men_keep-19<1:
        men_cycle+=1
    mens_list_official.extend([1] * max(0, men_cycle - len(mens_list_official) - 19))
    mens_list_official.extend([2] * max(0, men_cycle - len(mens_list_official) - 14))
    mens_list_official.extend([4] * (men_cycle - len(mens_list_official) == 14))
    mens_list_official.extend([2] * max(0, men_cycle - len(mens_list_official) - 9))
    mens_list_official.extend([3] * max(0, men_cycle - len(mens_list_official)))
    return mens_list_official


def mense_unnormal_list(men_keep=6, min_cycle=28, max_cycle=37, mens_list_official=[]):
    mens_list_official.extend([0] * men_keep)
    mens_list_official.extend([1] * max(min_cycle - 18 - men_keep, 0))
    mens_list_official.extend([2] * max(0, max_cycle - 11 - len(mens_list_official)))
    mens_list_official.extend([3] * max(0, min_cycle - len(mens_list_official)))
    return mens_list_official
def train_model_v2(uid,model_dir_name=None,bind_time=None):
    if model_dir_name is None:
        model_dir_name = uid
    try:
        ppgs_np_repeat, labels_np = get_data_set(uid,temperture=True,bind_time=bind_time)
        train_model(uid, ppgs_np_repeat, labels_np,  all=True, cls_num=2, big_model=False,model_dir_name=model_dir_name)
    except Exception as e:
        anlog.logger.info("训练模型error")
        anlog.logger.exception(e)
    cuda.select_device(0)
    cuda.close()
    model_uids.remove(uid)
    del ppgs_np_repeat
    del labels_np
    pass
'''
获取静息心率
'''
def get_hr_data(wear_user_id):
    sql = f"SELECT `date`,resting_heart_rate FROM andun_health.`h_pregnant_analysis` where wear_user_id='{wear_user_id}' ORDER BY date asc"
    return ansql.ansql_read_mysql(sql)
    pass
'''
获取温度数据
'''
def get_temperature_data(wear_user_id):
    sql = f"select * from andun_health.h_temperature_data where wear_user_id = '{wear_user_id}' ORDER BY date asc"
    return ansql.ansql_read_mysql(sql)
    pass
def get_conscious_sleep(uid,start_time,end_time):
    sql = f"select * from andun_health.h_analysis_sleep_conclusion where date>'{start_time}' and date<='{end_time}' and wear_user_id='{uid}'"

    return ansql.ansql_read_mysql(sql)
'''
获取步数
'''
def get_walk_data(uid,start_time,end_time):
    sql = f"select * from andun_watch.d_walk_data where WEAR_USER_ID = '{uid}' and date>='{start_time}' and date<='{end_time}'"
    return ansql.ansql_read_mysql(sql)
def get_walk_data(uid,date):
    sql = f"select * from andun_watch.d_walk_data where WEAR_USER_ID = '{uid}' and date='{date}'"
    return ansql.ansql_read_mysql(sql)
def get_bp_feature(uid,start_time,end_time):
    sql = f"select * from andun_watch.d_bp_feature where WEAR_USER_ID='{uid}' and DATE >='{start_time}' and DATE<='{end_time}'"
    return ansql.ansql_read_mysql(sql)
def get_bp_feature_by_date(uid,date_time):
    sql = f"select * from andun_watch.d_bp_feature where WEAR_USER_ID='{uid}' and DATE ='{date_time}'"
    return ansql.ansql_read_mysql(sql)
'''
获取心率数据
'''
def get_hr_watch(uid,date_time):
    sql = f"select * from andun_watch.d_hr_data where WEAR_USER_ID='{uid}' and DATE='{date_time}'"
    return ansql.ansql_read_mysql(sql)
    pass
'''
获取睡眠数据
'''
def get_sleep_data(wear_user_id):
    sql = f"select * from andun_health.h_exercise_sleep_conclusion where T_WEAR_USER_ID='{wear_user_id}' and date>'2022-06-01';"
    return ansql.ansql_read_mysql(sql)
'''
获取年龄及mbi及病史
'''
def get_mbi_age(uid):
    # 根据wear_user_id 获取mbi数据和age
    medical_history = get_wear_user_info(uid)
    personal_data = [medical_history["MEDICAL_HISTORY"],medical_history["AGE"],medical_history['WEIGHT']/((float(medical_history['STATURE'])/100.0)**2)]
    return personal_data
    pass
'''
获取基本信息
'''
def get_wear_user_info(uid):
    #sql = f"select MEDICAL_HISTORY from andunapp.a_wear_user where ID='{uid}'"
    sql = f"select * from andun_app.a_wear_user where ID='{uid}'"
    res = ansql.ansql_read_mysql(sql)
    if res.empty:
        return None
    return res
def get_hr(hr_data,x):
    #print(x,type(x))
    if pd.isna(x):
        return np.nan
    hr = hr_data.loc[hr_data.date ==dt.datetime(x.year,x.month,x.day).date(),'resting_heart_rate']
    #print(hr,type(hr),x)
    if hr.empty:
        #print("empty ",hr)
        return np.nan
    else:
        return hr.iloc[0]
    pass
def get_temperature(temperature_data,x):
    if pd.isna(x):
        return np.nan
    t = temperature_data.loc[temperature_data.date ==dt.datetime(x.year,x.month,x.day).date(), 'temperature_data']
    if t.empty:
        #print("empty ",t)
        return np.nan
    #print(t.iloc[0])

    sleep_array = t.iloc[0].split(";")
    #print(sleep_array)
    sleep_map = {}
    sleep_temperature_data = []
    all_temperature = []
    for s in sleep_array:
        t_arry = s.split(",")
        #print(t_arry)
        if int(t_arry[0])<50000 and float(t_arry[1])>0:
            sleep_temperature_data.append(float(t_arry[1]))
        if float(t_arry[1])>0:
            all_temperature.append(float(t_arry[1]))
        sleep_map[int(t_arry[0])] = float(t_arry[1])

    if len(sleep_temperature_data) > 0:
        return np.array(sleep_temperature_data).mean()
    elif len(all_temperature) > 0:
        return np.array(all_temperature).mean()
    return np.nan
    pass
def get_sleep_time(sleep_data,x):
    if pd.isna(x):
        return np.nan
    s = sleep_data.loc[sleep_data.date == dt.datetime(x.year,x.month,x.day).date(),'sleep_conclusion']
    if s.empty:
        return np.nan
    sleep_json = eval(s.iloc[0])
    #print(sleep_json["lightSleepTime"])
    return sleep_json["lightSleepTime"]
    pass
'''
根据day_date 获取体温及睡眠时间段及静息心率
'''
def get_women_temperature(day_data,uid):
    mbi_data = get_mbi_age(uid)
    hr_data = get_hr_data(uid)
    sleep_data = get_sleep_data(uid)
    temperature_data = get_temperature_data(uid)
    # 根据day_data 进行查询
    #day_data.dropna(inplace=True)
    day_data['hr'] = day_data["date"].apply(lambda x:get_hr(hr_data,x))
    day_data['mbi'] = day_data["date"].apply(lambda x:mbi_data[2])
    day_data['age'] = day_data["date"].apply(lambda x:mbi_data[1])
    day_data['medical_history'] = day_data["date"].apply(lambda x:int(str(mbi_data[0])=="1"))
    day_data['temperature'] = day_data["date"].apply(lambda x:get_temperature(temperature_data,x))
    day_data['sleep'] = day_data["date"].apply(lambda x:get_sleep_time(sleep_data,x))
    pass
def get_data_set(uid,cls_num=2,temperture=False,bind_time=None):
    # 目前都没有确认
    sql = f''' select * from andun_health.h_pregnant_analysis where wear_user_id='{uid}'  '''
    if bind_time:
        sql += f"and date>='{bind_time.date()}'"
    df_status = ansql.ansql_read_mysql(sql)
    df_status.loc[df_status.status != 0, 'status'] = 1

    did = ansql.did_uid(uid)
    if temperture:
        get_women_temperature(df_status,uid)
        m_data = df_status[["hr", "mbi", "age", "medical_history", "temperature", "sleep", "status"]]
        mean_data = np.nanmean(m_data, axis=0)
        df_status = df_status.fillna(
            {"hr": mean_data[0], "mbi": mean_data[1], "age": mean_data[2], "medical_history": mean_data[3],
             "temperature": mean_data[4], "sleep": mean_data[5]}, axis=0, inplace=False)

    with Mongo() as mongo:
        # t0 = min_date - dt.timedelta(hours=8)
        # t1 = max_date + dt.timedelta(hours=4)
        df_ppg = mongo.women_all_ppg_did(did)
    df_ppg['createTime'] = df_ppg.createTime.apply(lambda x: x + dt.timedelta(hours=8))
    df_ppg['ct_date'] = df_ppg.createTime.apply(lambda x: x.date())
    for i in range(df_status.shape[0]):
        date = df_status.date.iloc[i]
        status = df_status.status.iloc[i]
        if temperture:
            age = df_status.age.iloc[i]
            temperature = df_status.temperature.iloc[i]
            hr = df_status.hr.iloc[i]
            sleep = df_status.sleep.iloc[i]
            medical_history = df_status.medical_history.iloc[i]
            mbi = df_status.mbi.iloc[i]
            df_ppg.loc[df_ppg.ct_date == date, 'age'] = age / 100.0
            df_ppg.loc[df_ppg.ct_date == date, 'temperature'] = temperature / 37.0
            df_ppg.loc[df_ppg.ct_date == date, 'hr'] = hr / 100.0
            df_ppg.loc[df_ppg.ct_date == date, 'sleep'] = sleep / 21600.0
            df_ppg.loc[df_ppg.ct_date == date, 'medical_history'] = medical_history
            df_ppg.loc[df_ppg.ct_date == date, 'mbi'] = mbi / 30.0
        df_ppg.loc[df_ppg.ct_date == date, 'label'] = status

    df_ppg.dropna(inplace=True)
    df_ppg['PPG_Green'] = df_ppg['PPG_Green'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
    df_ppg['PPG_Red'] = df_ppg['PPG_Red'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
    df_ppg['PPG_IR'] = df_ppg['PPG_IR'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
    df_ppg['hour'] = df_ppg.createTime.apply(lambda x: x.hour)
    df_ppg = df_ppg.query('0 <= hour < 5')
    # 样本和标签
    ppgs = []
    ppgrs = []
    ppgirs = []
    labels = []
    ppgs_rfilt = df_ppg.PPG_Red.tolist()
    ppgs_irfilt = df_ppg.PPG_IR.tolist()
    ppgs_bfilt = df_ppg.PPG_Green.tolist()
    labels_bfilt = df_ppg.label.tolist()
    if temperture:
        person_data = df_ppg[["hr", "mbi", "age", "medical_history", "temperature", "sleep"]]
    drop_index = []
    for i in range(len(ppgs_bfilt)):
        ppg = ppgs_bfilt[i]
        ppgr = ppgs_rfilt[i]
        ppgir = ppgs_irfilt[i]
        lab = labels_bfilt[i]
        if len(ppg) >= 11240:
            ppgs.append(filt_data(ppg))
            labels.append(lab)
        else:
            drop_index.append(i)
        if len(ppgr) >= 11240:
            ppgrs.append(filt_data(ppgr))
        if len(ppgir) >= 11240:
            ppgirs.append(filt_data(ppgr))
    # 样本裁剪
    ppgs_cut = [ppg[500:10740] for ppg in ppgs]
    ppgs_cut1 = [ppg[500:10740] for ppg in ppgrs]
    ppgs_cut2 = [ppg[500:10740] for ppg in ppgirs]
    ppgs_np = np.array(ppgs_cut)
    ppgs_np1 = np.array(ppgs_cut1)
    ppgs_np2 = np.array(ppgs_cut2)
    if all:
        ppgs_np = np.concatenate([ppgs_np, ppgs_np1, ppgs_np2], axis=1)
    ppgs_np = (ppgs_np - 1e5) * 50 / (2e6 - 1e5)
    if temperture:
        person_data = person_data.drop(person_data.index[drop_index])
        ppgs_np = np.concatenate([ppgs_np, person_data], axis=1)
        ppgs_np = ppgs_np[:, 6:]
    del df_ppg

    labels_np = np.array(labels)
    # 样本量平衡
    counts_1 = (labels_np == 1).sum()
    counts_2 = (labels_np == 2).sum()
    counts_0 = labels_np.size - counts_1 - counts_2
    count = 100
    # anlog.logger.info((labels_np == 0).sum(),(labels_np == 1).sum(),(labels_np == 2).sum())
    # counts_1 = 10

    if cls_num == 2:
        ppgs_np_repeat_0 = ppgs_np[labels_np == 0].repeat(count // counts_0, 0)
        ppgs_np_repeat_2 = ppgs_np[labels_np == 1].repeat(count // counts_1, 0)
        labels_np = np.concatenate([labels_np, labels_np[labels_np == 0].repeat(count // counts_0, 0),
                                    labels_np[labels_np == 1].repeat(count // counts_1, 0)])
        ppgs_np_repeat = np.concatenate((ppgs_np, ppgs_np_repeat_0, ppgs_np_repeat_2))
    else:
        ppgs_np_repeat_0 = ppgs_np[labels_np == 0].repeat(count // counts_0, 0)
        ppgs_np_repeat_1 = ppgs_np[labels_np == 1].repeat(count // counts_1, 0)
        ppgs_np_repeat_2 = ppgs_np[labels_np == 2].repeat(count // counts_2, 0)
        labels_np = np.concatenate([labels_np, labels_np[labels_np == 0].repeat(count // counts_1, 0),
                                    labels_np[labels_np == 1].repeat(count // counts_1, 0),
                                    labels_np[labels_np == 2].repeat(count // counts_2, 0)])
        ppgs_np_repeat = np.concatenate((ppgs_np, ppgs_np_repeat_0, ppgs_np_repeat_1, ppgs_np_repeat_2))

    # 切分训练，验证集
    labels_np = tf.one_hot(labels_np, cls_num).numpy()
    return ppgs_np_repeat, labels_np
    pass

def train_model(uid,ppgs_np_repeat,labels_np,all=False,cls_num=2,big_model=False,model_dir_name=None):
    if model_dir_name is None:
        model_dir_name = uid
    # 判断训练数据的数量，少于100不训练
    if ppgs_np_repeat.shape[0] >= 100:
        anlog.logger.warn("{} 训练ppg数据大于100, 数据量{}  训练模型...".format(uid, ppgs_np_repeat.shape[0]))

        x_train, x_test, y_train, y_test = train_test_split(ppgs_np_repeat, labels_np, test_size=0.2)

        # 建模训练
        layers = k.layers
        in_channel = 10240
        reshape_channel = 4
        if all:
            in_channel = in_channel*3
            reshape_channel = reshape_channel*3

        model = k.Sequential([
            layers.Reshape((2560, reshape_channel), input_shape=(in_channel,)),
            #layers.Reshape((2560, 4), input_shape=(10240,)),
            layers.Conv1D(16, 256, 128, activation='relu', kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Conv1D(8, 4, activation='relu', kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(256, activation='relu', kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_initializer='normal'),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu', kernel_initializer='normal'),
            layers.BatchNormalization(),
            #layers.Dropout(0.1),
            #layers.Dense(cls_num, activation='softmax')
            layers.Dense(cls_num, activation='sigmoid')
        ])

        model.compile(
            k.optimizers.Adam(1e-3),
            loss=k.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        # model.compile(k.optimizers.SGD(1e-5),loss=k.losses.BinaryCrossentropy(),metrics = ['accuracy'])
        early_stopping = k.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True,
                                                   mode='min')

        h = model.fit(
            x_train,
            y_train,
            batch_size=100,
            epochs=500,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping]
        )

        # 训练可视化
        #training_vis(h)
        result_acc = model.evaluate(x_test, y_test)[1]
        #result_arry.append([uid,result_acc])
        if result_acc >= 0.7:
            if big_model:
                model.save(f'model_big/model_ppg_micro_{model_dir_name}')
            else:
                if all:
                    if cls_num == 3:
                        model.save(f'model_mbi_3/model_ppg_micro_{model_dir_name}')
                    else:
                        model.save(f'model_mbi/model_ppg_micro_{model_dir_name}')
                else:
                    model.save(f'model_green_2/model_ppg_micro_{model_dir_name}')
            anlog.logger.info("{} 训练ppg模型完成...保存model/model_ppg_micro_{}".format(uid, uid))
        else:
            anlog.logger.warn("{} 训练ppg模型完成...模型精度小于0.7,不保存".format(uid))
        #cuda.select_device(0)
        #cuda.close()
        #k.backend.clear_session()
        #k.clear_session()
    else:
        anlog.logger.warn("{} 训练ppg数据少于100, 数据量{}  不训练模型...".format(uid, ppgs_np_repeat.shape[0]))
    pass
import matplotlib.pyplot as plt
def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(8, 4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss')
    ax1.plot(val_loss, label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc')
    ax2.plot(val_acc, label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('plot_nogenerator.png')
    # plt.show() 因为使用的是Agg，所以无法使用


#服务实例
class Greeter(iandun_pb2_grpc.GreeterServicer):

    # 性别预测
    def GenderResults(self, request, context):
        try:
            id = request.ids
            date = dt.datetime.fromisoformat(request.date)
            print(id,date)
            ppg = ansql.ppg_oneday(id,date)[0]
            ppg = extractor_ppg(ppg)
            hr = ansql.hr_oneday(id,date)[0]
            hr = extractor_hr(hr)
            if ppg and hr:
                device = torch.device('cpu')
                model_ppg = Net_gender_ppg(12,512,64,1).to(device)
                weights_path = 'model/weights/gender_ppg.pth'
                model_ppg.load_state_dict(torch.load(weights_path,map_location='cpu'))

                model_hr = Net_gender_hr().to(device)
                weights_path_hr = 'model/weights/gender_hr.pth'
                model_hr.load_state_dict(torch.load(weights_path_hr,map_location='cpu'))

                all_ppg=[]
                for i in range(len(ppg)):
                    all_ppg.append(ppg[i])
                model_ppg.eval()
                with torch.no_grad():
                    X_std = (all_ppg - MIN) / (MAX - MIN)
                    x = X_std.astype(float)
                    nans = np.isnan(x)
                    X_tensor = torch.from_numpy(x).float()
                    out_t = model_ppg(X_tensor.to(device))
                    out_ppg = out_t.cpu().sum() / len(all_ppg)
                    flag1=int(out_ppg>GENDER_COEFF)

                model_hr.eval()
                with torch.no_grad():
                    temp = torch.tensor(np.array(hr), dtype=torch.float32)
                    val_x = temp.reshape(1, 1, 128)
                    out_hr = model_hr(val_x.to(device))
                    flag2 = int(out_hr.ge(GENDER_COEFF).int())

                if flag1 == 0 or flag2 == 0:
                    anlog.logger.info(f'{id}:result:{0}')
                    return iandun_pb2.GenderReply(message='0')
                else:
                    #0.31 0.9 0.7421
                    if out_ppg + out_hr >= 1.98:
                        anlog.logger.info(f'{id}:result:{1}')
                        return iandun_pb2.GenderReply(message='1')
                    else:
                        anlog.logger.info(f'{id}:result:{0}')
                        return iandun_pb2.GenderReply(message='0')
            return iandun_pb2.GenderReply(message='-1')
        except Exception as e: 
            anlog.logger.error(f'GenderError:uid:{id} date:{request.date}->{e}')
            return iandun_pb2.GenderReply(message='-1')

    # 生理接口
    def PhyResults(self, request,context):
        try:
            #基本信息
            uid = request.ids
            date = dt.date.fromisoformat(request.date)
            men_start_latest = dt.date.fromisoformat(request.men_start_latest)
            men_keep = int(request.men_keep)
            men_cycle = int(request.men_cycle) 

            anlog.logger.info("uid:{},date:{},men_start_latest:{},men_keep:{},men_cycle:{}".format(uid,date,men_start_latest,men_keep,men_cycle))           


            #生理周期公式枳实版,周期天数不够优先保证下一周期前14天为排卵日,返回生理状态分布列表
            mens_list_official = []
            try:
                mens_list_official = self.mens_list(men_keep, men_cycle, mens_list_official)
            except:
                # return iandun_pb2.PhyReply('-1')
                anlog.logger.error("{} self.mens_list错误...".format(uid))
                return iandun_pb2.PhyReply(message='-1')

            #生理周期公式,保证本排卵期合理性
            # mens_list = []
            # try:
            #     mens_list.extend([0]*men_keep)
            #     mens_list.extend([1]*max(0,men_cycle - len(mens_list) - 19))
            #     mens_list.extend([2]*max(5*(10 <= men_cycle - len(mens_list)),(10 > men_cycle - len(mens_list))*(int((men_cycle-len(mens_list)-1) / 2) + 1)))
            #     mens_list.append(4)
            #     mens_list.extend([2]*max(4*(4 <= men_cycle - len(mens_list)),(4 > men_cycle - len(mens_list))*men_cycle-len(mens_list)))
            #     mens_list.extend([3]*max(0,men_cycle - len(mens_list)))
            # except:
            #     return iandun_pb2.PhyReply('-1')

            #生理周期公式,保证本排卵期合理性 + 参数修正
            # mens_list_refix = []
            # try:
            #     mens_list_refix.extend([0]*men_keep)
            #     mens_list_refix.extend([1]*max(0,men_cycle - len(mens_list_refix) - 19))
            #     mens_list_refix.extend([2]*max(5*(10 <= men_cycle - len(mens_list_refix)),(10 > men_cycle - len(mens_list_refix))*(int((men_cycle-len(mens_list_refix)-1) / 2) + 1)))
            #     mens_list_refix.append(4)
            #     mens_list_refix.extend([2]*max(4*(4 <= men_cycle - len(mens_list_refix)),(4 > men_cycle - len(mens_list_refix))*men_cycle-len(mens_list_refix)))
            #     mens_list_refix.extend([3]*max(0,men_cycle - len(mens_list_refix)))
            # except:
            #     return iandun_pb2.PhyReply('-1')

            # 使用日期差索引当日状态
            diff_days = (date - men_start_latest).days
            ind = diff_days % len(mens_list_official)
            res_formula = mens_list_official[ind]

            # 4p版本进行查询
            bind_sql = f"select * from andun_app.t_binding_log where WEAR_USER_ID = '{uid}' ORDER BY Binding_time desc limit 1;"
            device_info = ansql.by_sql(bind_sql)
            bind_time = device_info[3]
            device_id = device_info[1]
            version_sql = f"select Device_version from andun_cms.a_device where Id='{device_id}'"
            version_v = ansql.by_sql(version_sql)[0]
            model_dir_name = uid
            if version_v == "4P":
                model_dir_name = uid + "_" + device_id
            #用户状态确认，状态是否已经由模型结果修正
            over_write = False
            sql_status_yesterday = f''' select last_menstruation_date from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date>='{date - dt.timedelta(2)}' '''
            status_yesterday = ansql.list_by_sql(sql_status_yesterday)
            anlog.logger.info("status_yesterday {},{},{}".format(status_yesterday[0][0].strftime('%Y-%m-%d'),status_yesterday[1][0].strftime('%Y-%m-%d'),bind_time))
            if status_yesterday and status_yesterday[0][0] != status_yesterday[1][0]:
                over_write = True
            #status_yesterday_official = mens_list_official[ind-1]

            # 进入模型条件 存在个人模型使用模型出值，不存在且 生理周期出值满 28天 使用器官模型同时训练个人模型， 都不满足走公式出值
            flag_model = False
            sql_count_ana = f''' select count(*) from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date>'{bind_time.date()}' '''
            counts_ana = ansql.by_sql(sql_count_ana)[0]

            if counts_ana >= 28 and (not os.path.exists(f'model_mbi/model_ppg_micro_{model_dir_name}') or over_write):
                flag_model = 'organ'
                async_train_model(uid,model_dir_name,bind_time)
                # t = threading.Thread(target=train_model_v2,kwargs={'uid':uid})
                # t.start()
                # anlog.logger.info("{} 开始训练ppg模型...".format(uid))
                #train_model_personal(uid)
                #train_model_v2(uid)

            if os.path.exists(f'model_mbi/model_ppg_micro_{model_dir_name}'):
                flag_model = 'ppg'


            if not flag_model:
                anlog.logger.info("if not flag_model: res_formula: {}".format(res_formula))
                return iandun_pb2.PhyReply(message=f'{res_formula}')


            #ppg模型出值，取当日所有ppg平均结果出值
            if flag_model == 'ppg':
                #uid,date,res_formula,men_cycle,ind
                return self.predict_v_1(uid, date, res_formula,men_start_latest, men_keep, men_cycle, ind,model_dir_name=model_dir_name,bind_time=bind_time)
                # try:
                #     sql_status_yesterday = f''' select status from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date='{date - dt.timedelta(1)}' '''
                #     status_yesterday = ansql.by_sql(sql_status_yesterday)
                #     if status_yesterday is None:
                #         status_yesterday=-1
                #     else:
                #         status_yesterday = status_yesterday[0]
                #
                #     if res_formula==1 and status_yesterday==0 and abs(men_cycle-ind)<2:
                #         res_model = model_predict_2(uid, date,True)
                #         print("模型预测结果0", res_model, date)
                #         return iandun_pb2.PhyReply(message=f'{int(float(res_model) <= 0.146)}')
                #     elif res_formula==0 and status_yesterday==1:
                #         res_model = model_predict_2(uid, date, True)
                #         print("模型预测结果1",res_model,date)
                #         return iandun_pb2.PhyReply(message='1')
                #
                #     else:
                #         res_model = model_predict_2(uid, date, True)
                #         print("模型预测结果11111",res_model,date)
                #         result_res = int(float(res_model)< 0.146)
                #         if res_formula != 0 and result_res == 0 and abs(men_cycle-ind)<4:
                #             return iandun_pb2.PhyReply(message=f'{result_res}')
                #         print("模型预测结果122221", result_res,ind,date)
                #         if res_formula ==0 and result_res !=0 and ind <2 and status_yesterday!=0 and float(res_model)>0.06:
                #             return iandun_pb2.PhyReply(message='3')
                #         anlog.logger.info("flag_model: {}: res_formula: {} result_res {}".format(flag_model, res_formula,result_res))
                #         return iandun_pb2.PhyReply(message=f'{res_formula}')
                #
                # except Exception as e:
                #     anlog.logger.error("{} ppg模型预测错误: {}...".format(uid, e))
                #     anlog.logger.exception("{} ppg模型预测错误: {}...".format(uid, e))
                #     return iandun_pb2.PhyReply(message=f'{res_formula}')


            # 器官模型出值，器官模型属于对公式进行修正，目前为满足一定条件提前结束当次月经期
            else:
                try:
                    res_organ = self.predict_organ(uid, date)

                    sql_preg_ana = f''' select status from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date='{date - dt.timedelta(1)}' '''
                    status_preg = ansql.by_sql(sql_preg_ana)[0]
                    if men_keep >= 6 and ind > 4 and res_organ <= (ind - 4)*0.15 and status_preg == 0:
                        men_keep = ind + 1
                        mens_list_official = self.mens_list(men_keep,men_cycle,mens_list_official)

                        diff_days = (date - men_start_latest).days
                        ind = diff_days % len(mens_list_official)
                        res = mens_list_official[ind]
                        anlog.logger.info("{} ...res: {}".format(uid, res))
                        return iandun_pb2.PhyReply(message=f'{res}')
                    elif status_preg != 0 and res_formula == 0:
                        if status_preg == 4 :
                            anlog.logger.info("{} ...status_preg: {} --- 2".format(uid, status_preg))
                            return iandun_pb2.PhyReply(message=f'{2}')
                        anlog.logger.info("{} ...status_preg: {}".format(uid, status_preg))
                        if ind<2 and status_preg==3:
                            return iandun_pb2.PhyReply(message=f'{res_formula}')
                        return iandun_pb2.PhyReply(message=f'{status_preg}')
                    else:
                        anlog.logger.error("{} 器官模型预测条件不足,返回公式出值....res_organ: {}, men_keep: {}, status_preg: {}, res_formula: {}".format(uid,res_organ,men_keep,status_preg,res_formula))
                        return iandun_pb2.PhyReply(message=f'{res_formula}')
                except Exception as e:
                    anlog.logger.error("{} 器官模型预测错误：{}...返回res_formula: {}".format(uid, e, res_formula))
                    return iandun_pb2.PhyReply(message=f'{res_formula}')

        except Exception as e:
            try:
                mens_list_official = self.mens_list()
            except:
                # return iandun_pb2.PhyReply('-1')
                anlog.logger.error("{} self.mens_list错误...".format(uid))
                return iandun_pb2.PhyReply(message='-1')
            diff_days = (date - men_start_latest).days
            ind = diff_days % len(mens_list_official)
            res_formula = mens_list_official[ind]
            anlog.logger.exception(e)
            anlog.logger.error(f'PhyError:{e}\nwear_user_id:{uid};date:{request.date}')
            return iandun_pb2.PhyReply(message=f'{res_formula}')

    '''
    考虑个人其他因素进行模型预测
    1. 模型预测和公式相等，安装公式推算进行
    2. 模型预测为月经期，但是公式偏差很大，则返回公式，反之，更新公式如调整月经周期，（预测月经到来，但是偏差在7天以上，则认为数据问题，如果再7天之内进行周期调整，适当增加或减少月经周期或上次月经到来时间）
    3. 未生成ppg模型不予处理
    '''
    def predict_v_1(self,uid,date,res_formula,men_start_lastest, men_keep,men_cycle,ind,dbtype="test",model_dir_name=None,bind_time=None):
        flag_model = 'ppg'
        try:
            sql_status_yesterday = f''' select status from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date='{date - dt.timedelta(1)}' '''
            status_yesterday = ansql.by_sql(sql_status_yesterday)
            if status_yesterday is None:
                status_yesterday = -1
            else:
                status_yesterday = status_yesterday[0]
            # 根据模型预测结果进行分析，如果判断是
            res_model = model_predict_2(uid, date, True,model_dir_name=model_dir_name)
            anlog.logger.info("模型预测结果 res_model {} date {}".format(res_model, date))
            # 验证公式的正确性，如果开始开日期模型预测正确则返回正常状态 针对月经周期延长或者缩短进行分析
            #model_result = int(float(res_model) <= 0.5)
            model_result = get_model_result_by_config(uid, date, res_model,res_formula, men_start_lastest, men_keep, men_cycle,status_yesterday, ind,dbtype,bind_time=bind_time)
            anlog.logger.info("flag_model: {}: res_formula: {} result_res {} ind {}".format(flag_model, res_formula, model_result,ind))
            did = ansql.did_uid(uid)
            model_type =1
            ratio = -1
            if res_model is None:
                model_type=0
            else:
                ratio = round(res_model,2)
            data_ansql = ansql
            if dbtype =="test":
                data_ansql = Sql("test")
            self.insert_pregnant_data(data_ansql,uid,did,date,model_result,model_type,ratio,men_keep,men_start_lastest,men_cycle)
            return iandun_pb2.PhyReply(message=f'{int(model_result)}')


        except Exception as e:
            anlog.logger.error("{} ppg模型预测错误: {}...".format(uid, e))
            anlog.logger.exception("{} ppg模型预测错误: {}...".format(uid, e))
            return iandun_pb2.PhyReply(message=f'{res_formula}')
        pass
    def insert_pregnant_data(self,data_ansql,uid,did,date,status_res,model_type,ratio,men_keep,last_menstruation_date,men_cycle):
        # 1. 查询sql 查询本个周期的数据截至到今天的预测数据 计算目前为止下标然后根据公式推断
        data_sql = f"select * from andun_health.h_pregnant_cycle_data where wear_user_id ='{uid}' and date>='{last_menstruation_date}' order by date asc"
        # 根据查询的数据遍历解析得到已经
        data = data_ansql.ansql_read_mysql(data_sql)
        if data.empty:
            data_sql = f"select * from andun_health.h_pregnant_analysis where wear_user_id ='{uid}' and date>='{last_menstruation_date}' order by date asc"
            data = ansql.ansql_read_mysql(data_sql)
        df_status = data[["status"]].values
        data["day"] = data["date"].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_day = data[["day"]].values
        print(df_status)
        print(type(df_status))
        df_status = np.squeeze(df_status).tolist()
        df_day = np.squeeze(df_day).tolist()
        if isinstance(df_status,int):
            df_status = [df_status]
            df_day = [df_day]
        print(df_status)
        print(df_day)
        # 根据开始日期推算算出总共的天数
        # start_day = dt.date.fromisoformat(last_menstruation_date)
        data_diff = (date - last_menstruation_date).days
        start_day = last_menstruation_date
        if data_diff>=men_cycle and status_res==0:
            start_day = date


        days_arr = []
        for i in range(int(men_cycle)):
            d = start_day + dt.timedelta(days=i)
            days_arr.append(d.strftime('%Y-%m-%d'))
        print(days_arr)

        # 获取到整个周期
        mens_list_official = self.get_all_pre_json(int(men_keep), int(men_cycle), 0, 0, True)
        print(mens_list_official)
        # 2.组装pre_json，判断组装过程中是否有违背生理周期正常的规律的预测然后进行适当纠正
        real_data = []
        last_status = 0
        has_ori_day = 0
        for d in days_arr:
            last_status = 0 if len(real_data) <= 0 else real_data[len(real_data) - 1]
            if last_status==4:
                has_ori_day=1
            if d in df_day:
                if (last_status <= df_status[df_day.index(d)] and (df_status[df_day.index(d)] ==4 and has_ori_day!=1)) or (df_status[df_day.index(d)] == 2 and last_status == 4) :
                    real_data.append(df_status[df_day.index(d)])
                else:
                    real_data.append(last_status)
            else:
                if last_status <= mens_list_official[days_arr.index(d)] or (
                        mens_list_official[days_arr.index(d)] == 2 and last_status == 4):
                    real_data.append(mens_list_official[days_arr.index(d)])
                else:
                    real_data.append(last_status)

        print(real_data)
        pre_json = dict(zip(days_arr, real_data))
        # 纠正部分预测问题
        # 3.返回整个周期的json数据
        ori = -1
        results = {}
        for i, r in enumerate(real_data):
            if ori != r:
                if r == 0:
                    results["menstrual_period"] = days_arr[i]
                elif r == 1:
                    results["safe_period"] = days_arr[i]
                elif r == 2 and ori != 4:
                    results["ovulation_period"] = days_arr[i]
                elif r == 4:
                    results["ovulation_day"] = days_arr[i]
                elif r == 3:
                    results["luteal_phase"] = days_arr[i]
            ori = r
        result_json = {"detail": pre_json, "period": results}
        # results["safe_period"] = safe_period
        # results["ovulation_period"] = "2022-10-11"
        # results["ovulation_day"] = "2022-10-16"
        # results["luteal_phase"] = "2022-10-21"

        self.db_insert(data_ansql, uid, did, date, status_res, ratio, result_json, model_type, men_keep,
                       last_menstruation_date, men_cycle)
    # 使用已训练的生理周期模型预测
    def model_predict(self, uid, date):
        did = ansql.did_uid(uid)
        model = k.models.load_model(f'model/model_ppg_{uid}')
        date = dt.datetime.combine(date, dt.time())
        t0 = date - dt.timedelta(hours=8)
        t1 = date + dt.timedelta(hours=4)
        with Mongo() as mongo:
            df_ppg = mongo.women_ppg_did_date(did,t0,t1)
        df_ppg['ppg'] = df_ppg['PPG_Green'].apply(lambda x:np.fromstring(x[1:-1],float,sep=','))
        ppgs_bfilt = df_ppg.ppg.tolist()

        ppgs = []
        for i in range(len(ppgs_bfilt)):
            ppg = ppgs_bfilt[i]
            if len(ppg) >= 11240:
                ppgs.append(filt_data(ppg))

        ppgs_cut = [ppg[500:10740] for ppg in ppgs]
        ppgs_np = np.array(ppgs_cut)
        ppgs_np = (ppgs_np - 1e5) * 50 / (2e6 - 1e5)

        res_model = model.predict(ppgs_np,batch_size=ppgs_np.shape[0])

        res_model = res_model[:,0].sum() / res_model.shape[0]

        return res_model

    def PhySiologicalCyclePersonalModel(self,request,context):
        try:
            uid = request.uid
            update = request.update
            print(uid,update)

            if update == '1':
                if self.get_count_ppg(uid):
                    #train_model_v2(uid)
                    t = threading.Thread(target=train_model_v2,kwargs={'uid':uid})
                    t.start()
                    anlog.logger.info("{} 开始训练ppg模型...".format(uid))
                #else:
                    #return iandun_pb2.PhyModelReply(message='-1')
                #return iandun_pb2.PhyModelReply(message='2')
            else:
                if os.path.exists(f'model_gr_2/model_ppg_micro_{uid}'):
                    return
                    #return iandun_pb2.PhyModelReply(message='0')
                else:
                    if self.get_count_ppg(uid):
                        train_model_v2(uid)
                    else:
                        return
                        #return iandun_pb2.PhyModelReply(message='-1')
            return iandun_pb2.PhySiologicalCyclePersonalModelReply(message='1')
        except Exception as e:
            anlog.logger.error(f'phyModelResults:uid:{id} ->{e}')
            return iandun_pb2.PhySiologicalCyclePersonalModelReply(message='-1')
    def PhySiologicalCycleCommonModel(self,request,context):
        return iandun_pb2.PhySiologicalCycleCommonModelReply(message='-1')
    def get_count_ppg(self,uid):
        sql_count_ana = f''' select count(*) from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' '''
        counts_ana = ansql.by_sql(sql_count_ana)[0]
        return counts_ana >= 28

    # 使用器官模型预测生理周期
    def predict_organ(self, uid, date):
        sql_organ = f''' select models from andun_health.h_health_analysis_women where t_wear_user_id='{uid}' and create_time between '{date}' and '{date+dt.timedelta(days=1)}' limit 1'''
        data_organ = ansql.by_sql(sql_organ)[0]
        data_organ = [i['status'] for i in json.loads(data_organ)['organs']]
        data_organ = np.array(data_organ).reshape(1,14)

        model_organ = k.models.load_model('model/model_organ')
        res_organ = model_organ.predict(data_organ / 4).reshape(1)
        return res_organ[0]

    # 专门计算生理周期分布列表的, 例如周期28天 列表长28 里面填0～4各种状态，计算哪天的状态，就直接索引
    def mens_list(self, men_keep=6, men_cycle=28, mens_list_official=[]):
        mens_list_official.extend([0]*men_keep)
        mens_list_official.extend([1]*max(0,men_cycle - len(mens_list_official) - 19))
        mens_list_official.extend([2]*max(0,men_cycle - len(mens_list_official) - 14))
        mens_list_official.extend([4]*(men_cycle - len(mens_list_official) == 14))
        mens_list_official.extend([2]*max(0,men_cycle-len(mens_list_official) - 9))
        mens_list_official.extend([3]*max(0,men_cycle - len(mens_list_official)))
        return mens_list_official

    # 计算胎动
    def QuickResults(self, request, context):
        try:
            anlog.logger.info('Cal Quickening Start')
            hrs = np.fromstring(request.hrs.replace(';',','),int,sep=',').reshape(-1,2)
            t_limit = T_LIMIT_QUICKENING
            hr_hrs = hrs[:,1]
            res = cal_quickening(hr_hrs)
            if len(res) > 0:
                res = hrs[:,0][[res]]
            else:
                res = 0
            anlog.logger.info(f'Result Quickening:{res}')
            res = res.tolist()
            return iandun_pb2.QuickReply(message=f''' {str(res)[1:-1].replace(' ', '').replace(',', ';')} ''')
        except Exception as e:
            anlog.logger.error(f'Cal Quickening Error:{e}')
            return iandun_pb2.QuickReply('-1')

    '''
    月经周期预测，整个周期预测
    1.默认公式（根据用户填写信息进行判断公式的变异性）
    2.模型构建 lightgbm ppg模型
    3.纠正模块 （通过最近的数据进行验证之前的周期预测，对不符的数据进行纠正然后重新训练模型）
    4.周期数据封装
    '''
    def PhySiologicalCycleStage(self,request,context):
        try:
            uid = request.ids
            date = dt.date.fromisoformat(request.date)
            anlog.logger.info(f"PhyAllResults {uid},{date}")
            # results = {}
            # results["menstrual_period"] = "2022-10-01"
            # results["safe_period"] = "2022-10-06"
            # results["ovulation_period"] = "2022-10-11"
            # results["ovulation_day"] = "2022-10-16"
            # results["luteal_phase"] = "2022-10-21"
            # #results["menstrual_period"] = "2022-10-01"
            # #results["menstrual_period"] = "2022-10-01"
            # json_str = json.dumps(results)
            sql = f''' select * from andun_health.h_pregnant_analysis where wear_user_id='{uid}' order by date desc limit 5 '''
            df = ansql.ansql_read_mysql(sql)
            if df.empty:
                config_sql = f'''SELECT * FROM andun_app.`a_pregnant_config` where wear_user_id='{uid}' '''
                df_config = ansql.ansql_read_mysql(config_sql)
                if df_config.empty:
                    raise Exception("a_pregnant_config is None")
                anlog.logger.info(f"df_config is {df_config['config_param'].iloc[0]}")
                config_json = json.loads(df_config["config_param"].iloc[0])
                anlog.logger.info(f"df_config is {config_json}",)
                last_menstruation_date = config_json["lastMensesTime"]
                menses_period =config_json["mensesPeriod"] if "mensesPeriod" in config_json.keys() else 30
                menses_duration = config_json["mensesDuration"] if "mensesDuration" in config_json.keys() else 5
                anlog.logger.info(f"{last_menstruation_date},{menses_period},{menses_duration}")

                print("=="*5)
                print(df)
            else:
                data_person = df[['date', 'wear_user_id', 'menses_duration', 'menses_period','last_menstruation_date', 'status']]

                menses_duration = data_person["menses_duration"][0]
                menses_period = data_person["menses_period"][0]
                last_menstruation_date = data_person["last_menstruation_date"][0]
            if menses_duration==-1:
                menses_duration=5
            if menses_period==-1:
                menses_period=31
            # 针对预测的结果进行入库后对当前的状态进行整个周期的预测
            #rd, pre_json, json_str = self.get_all_period(uid,ansql_test, date,menses_duration, menses_period, last_menstruation_date)
            json_str = self.get_db_search(ansql_test,uid,last_menstruation_date,menses_period,menses_duration)
            # return iandun_pb2.PhyAllReply(menstrual_period='null',
            #                               safe_period='2022-10-06:2022-10-10', ovulation_period='2022-10-11:2022-10-20',
            #                               ovulation_day='2022-10-16', luteal_phase='2022-10-21:2022-10-30',message=f'{json_str}')
            return iandun_pb2.PhySiologicalCycleStageReply(message=f'{json_str}')

            #return iandun_pb2.PhyAllReply(menstrual_period=f'2022-10-01:2022-10-05',safe_period='2022-10-06:2022-10-10',ovulation_period='2022-10-11:2022-10-20',ovulation_day='2022-10-16',luteal_phase='2022-10-21:2022-10-30')
        except Exception as e:
            anlog.logger.info("PhySiologicalCycleStage error")
            anlog.logger.exception(e)
            return  iandun_pb2.PhySiologicalCycleStageReply(message='null')
    '''
    获取模型结果参数
    '''
    def get_model_ratio(self,uid,date, men_keep, men_cycle, men_start_latest,dbtype="online"):
        model_type = 1
        res_model = 1
        mens_list_official = []
        mens_list_official = self.mens_list(men_keep, men_cycle, mens_list_official)
        # 使用日期差索引当日状态
        diff_days = (date - men_start_latest).days
        ind = diff_days % len(mens_list_official)
        res_formula = mens_list_official[ind]

        # 4p版本进行查询
        bind_sql = f"select * from andun_app.t_binding_log where WEAR_USER_ID = '{uid}' ORDER BY Binding_time desc limit 1;"
        device_info = ansql.by_sql(bind_sql)
        bind_time = device_info[3]
        device_id = device_info[1]
        version_sql = f"select Device_version from andun_cms.a_device where Id='{device_id}'"
        version_v = ansql.by_sql(version_sql)[0]
        model_dir_name = uid
        if version_v == "4P":
            model_dir_name = uid + "_" + device_id
        over_write = False
        sql_status_yesterday = f''' select last_menstruation_date from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date>='{date - dt.timedelta(2)}' '''
        status_yesterday = ansql.list_by_sql(sql_status_yesterday)
        anlog.logger.info("status_yesterday {},{},{}".format(status_yesterday[0][0].strftime('%Y-%m-%d'),
                                                             status_yesterday[1][0].strftime('%Y-%m-%d'), bind_time))
        if status_yesterday and status_yesterday[0][0] != status_yesterday[1][0]:
            over_write = True
        flag_model = False
        sql_count_ana = f''' select count(*) from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date>'{bind_time.date()}' '''
        counts_ana = ansql.by_sql(sql_count_ana)[0]

        if counts_ana >= 28 and (not os.path.exists(f'model_mbi/model_ppg_micro_{model_dir_name}') or over_write):
            flag_model = 'organ'
            async_train_model(uid, model_dir_name, bind_time)
        if os.path.exists(f'model_mbi/model_ppg_micro_{model_dir_name}'):
            flag_model = 'ppg'

        if not flag_model:
            anlog.logger.info("if not flag_model: res_formula: {}".format(res_formula))
            return iandun_pb2.PhyReply(message=f'{res_formula}')
        if flag_model == 'ppg':
            try:
                sql_status_yesterday = f''' select status from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date='{date - dt.timedelta(1)}' '''
                status_yesterday = ansql.by_sql(sql_status_yesterday)
                if status_yesterday is None:
                    status_yesterday = -1
                else:
                    status_yesterday = status_yesterday[0]
                # 根据模型预测结果进行分析，如果判断是
                try:
                    res_model = model_predict_2(uid, date, True, model_dir_name=model_dir_name)
                except Exception as e1:
                    anlog.logger.exception(e1)
                anlog.logger.info("模型预测结果 res_model {} date {}".format(res_model, date))
                # 验证公式的正确性，如果开始开日期模型预测正确则返回正常状态 针对月经周期延长或者缩短进行分析
                # model_result = int(float(res_model) <= 0.5)
                model_type = 1
                model_result = get_model_result_by_config(uid, date, res_model, res_formula, men_start_latest,
                                                          men_keep, men_cycle, status_yesterday, ind, dbtype,
                                                          bind_time=bind_time)
                anlog.logger.info(
                    "flag_model: {}: res_formula: {} result_res {} ind {}".format(flag_model, res_formula, model_result,ind))
            except Exception as e:
                model_type = 0
                anlog.logger.exception(e)
        if res_model is None:
            res_model = 0
            model_result = res_formula
        return device_id,model_result,model_type,res_model
    def get_db_search(self,ansql_test,uid,last_menstruation_date,men_cycle,men_keep):
        sql = f"select * from andun_health.`h_pregnant_cycle_data` where wear_user_id='{uid}' order by date desc limit 1"
        pregnant_data = ansql_test.ansql_read_mysql(sql)
        if pregnant_data.empty:
            # 查询analysis表并进行拼装
            data_sql = f"select * from andun_health.h_pregnant_analysis where wear_user_id = '{uid}' and date>='{last_menstruation_date}' order by date asc"

            data = ansql.ansql_read_mysql(data_sql)
            df_status = data[["status"]].values
            data["day"] = data["date"].apply(lambda x: x.strftime('%Y-%m-%d'))
            df_day = data[["day"]].values
            df_status = np.squeeze(df_status).tolist()
            df_day = np.squeeze(df_day).tolist()
            date = dt.date.today()
            data_diff = (date - last_menstruation_date).days
            start_day = last_menstruation_date
            if data_diff >= men_cycle:
                start_day = date
            days_arr = []
            for i in range(int(men_cycle)):
                d = start_day + dt.timedelta(days=i)
                days_arr.append(d.strftime('%Y-%m-%d'))
            mens_list_official = self.get_all_pre_json(int(men_keep), int(men_cycle), 0, 0, True)
            real_data = []
            last_status = 0
            for d in days_arr:
                last_status = 0 if len(real_data) <= 0 else real_data[len(real_data) - 1]
                if d in df_day:
                    if last_status <= df_status[df_day.index(d)] or (
                            df_status[df_day.index(d)] == 2 and last_status == 4):
                        real_data.append(df_status[df_day.index(d)])
                    else:
                        real_data.append(last_status)
                else:
                    if last_status <= mens_list_official[days_arr.index(d)] or (
                            mens_list_official[days_arr.index(d)] == 2 and last_status == 4):
                        real_data.append(mens_list_official[days_arr.index(d)])
                    else:
                        real_data.append(last_status)

            pre_json = dict(zip(days_arr, real_data))
            if real_data[0]!=0:
                real_data = mens_list_official
                pre_json = dict(zip(days_arr,mens_list_official))
            ori = -1
            results = {}
            for i, r in enumerate(real_data):
                if ori != r:
                    print(len(days_arr),i)
                    if i >= len(days_arr):
                        break
                    if r == 0:
                        results["menstrual_period"] = days_arr[i]
                    elif r == 1:
                        results["safe_period"] = days_arr[i]
                    elif r == 2 and ori != 4:
                        results["ovulation_period"] = days_arr[i]
                    elif r == 4:
                        results["ovulation_day"] = days_arr[i]
                    elif r == 3:
                        results["luteal_phase"] = days_arr[i]
                ori = r
            result_json = {"detail": pre_json, "period": results}

            score_dic = self.get_pregnancy_score(result_json)
            result_json["pregnant_score"] = score_dic
            pre_json = json.dumps(result_json)
            return pre_json
        else:
            pre_json = pregnant_data["pre_json"][0]
            score_dic = self.get_pregnancy_score(pre_json)
            pre_json["pregnant_score"] = score_dic
            #pre_json = json.dumps(pre_json)
        return pre_json

    def get_pregnancy_score(self,pre_json):
        # 获取detail 数据进行分析汇总
        js_list = pre_json["detail"]
        # 获取json详细情况
        js_list= sorted(js_list.items())
        dic = dict(js_list)
        l = list(dic.values())
        vs = list(dic.keys())
        o_index = list(dic.values()).index(4)
        o_end_index = [get_ovulation_index(i, x) for i, x in enumerate(l)]
        o_index_list = list(filter(lambda x: x is not None, o_end_index))
        print(o_index_list)
        # [i for i,v in enumerate(l)]
        o_start_index = l.index(2)
        print("=====" * 5)
        print(o_index)

        o_end_index = max(o_index_list)
        result_list = [get_score(i, l, vs, o_index, o_end_index) for i in vs]
        score_dic = dict(zip(vs, result_list))
        return score_dic
        pass

    '''
    根据周期进行整个周期预测
    '''
    def get_all_period(self,uid,ansql, date, men_keep, men_cycle, last_menstruation_date):
        # 调用模型预测获取结果
        # 根据当前数据进行预测
        did,status_res,model_type,ratio = self.get_model_ratio(uid,date, men_keep, men_cycle, last_menstruation_date)
        # 1.查询本周期开始日期之后的预测数据
        # 获取新的的周期和例假时间
        # 1. 查询sql 查询本个周期的数据截至到今天的预测数据 计算目前为止下标然后根据公式推断
        data_sql = f"select * from andun_health.h_pregnant_cycle_data where date>='{last_menstruation_date}' order by date asc"
        # 根据查询的数据遍历解析得到已经
        data = ansql.ansql_read_mysql(data_sql)
        df_status = data[["status"]].values
        data["day"] = data["date"].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_day = data[["day"]].values
        print(df_status)
        print(type(df_status))
        df_status = np.squeeze(df_status).tolist()
        df_day = np.squeeze(df_day).tolist()
        print(df_status)
        print(df_day)
        # 根据开始日期推算算出总共的天数
        #start_day = dt.date.fromisoformat(last_menstruation_date)
        start_day = last_menstruation_date
        days_arr = []
        for i in range(int(men_cycle)):
            d = start_day + dt.timedelta(days=i)
            days_arr.append(d.strftime('%Y-%m-%d'))
        print(days_arr)

        # 获取到整个周期
        mens_list_official = self.get_all_pre_json(int(men_keep), int(men_cycle), 0, 0, True)
        print(mens_list_official)
        # 2.组装pre_json，判断组装过程中是否有违背生理周期正常的规律的预测然后进行适当纠正
        real_data = []
        last_status = 0
        for d in days_arr:
            last_status = 0 if len(real_data) <= 0 else real_data[len(real_data) - 1]
            if d in df_day:
                if last_status <= df_status[df_day.index(d)] or (df_status[df_day.index(d)] == 2 and last_status == 4):
                    real_data.append(df_status[df_day.index(d)])
                else:
                    real_data.append(last_status)
            else:
                if last_status <= mens_list_official[days_arr.index(d)] or (
                        mens_list_official[days_arr.index(d)] == 2 and last_status == 4):
                    real_data.append(mens_list_official[days_arr.index(d)])
                else:
                    real_data.append(last_status)
        print(real_data)
        pre_json = dict(zip(days_arr, real_data))
        # 纠正部分预测问题
        # 3.返回整个周期的json数据
        ori = -1
        results = {}
        for i, r in enumerate(real_data):
            if ori != r:
                if r == 0:
                    results["menstrual_period"] = days_arr[i]
                elif r == 1:
                    results["safe_period"] = days_arr[i]
                elif r == 2 and ori != 4:
                    results["ovulation_period"] = days_arr[i]
                elif r == 4:
                    results["ovulation_day"] = days_arr[i]
                elif r == 3:
                    results["luteal_phase"] = days_arr[i]
            ori = r

        # results["safe_period"] = safe_period
        # results["ovulation_period"] = "2022-10-11"
        # results["ovulation_day"] = "2022-10-16"
        # results["luteal_phase"] = "2022-10-21"
        result_json = {"detail":pre_json,"period":results}
        self.db_insert(ansql, uid, did, date, status_res, ratio, result_json, model_type, men_keep,
                       last_menstruation_date, men_cycle)
        return real_data, pre_json, results

    def db_insert(self,ansql, uid, did, date, status_res, ratio, per_json, model_type, men_keep, men_start_latest,
                  men_cycle):
        per_json = json.dumps(per_json)
        #model_type = 1
        ansql.h_prengnant_data_insert(uid, did, date, status_res, ratio, per_json, model_type, men_keep,
                                      men_start_latest,
                                      men_cycle)

    '''
    根据用户之前的历史数据进行预测
    '''
    def get_all_pre_json(self,men_keep, men_cycle, min_cycle, max_cycle, regular=True):

        mens_list_official = []
        if regular:
            mens_list_official = mens_config_list(men_keep, men_cycle)
        else:
            mens_list_official = mense_unnormal_list(men_keep, min_cycle, max_cycle)
        return mens_list_official


#入口
def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    iandun_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port(f'[::]:{PORT}') 
    server.start()
    print('服务启动成功')
    anlog.logger.info('Server Started')
    try:
        while True:
            time.sleep(ONE_DAY_IN_SECONDS) # one day in seconds
    except KeyboardInterrupt:
        anlog.logger.info('Server Stoped')
        print('服务关闭')
        server.stop(0)



#启动
if __name__ == '__main__':
    main()

# 测试
# sql = f''' SELECT wear_user_id as uid,date,is_man FROM andun_health.h_pregnant_analysis'''
# df_pregnant = ansql.ansql_read_mysql(sql)
# ser = Greeter()
# id = 'd4f6b2c4'
# date = '2022-01-19'
# ids = df_pregnant.uid.tolist()
# dates = df_pregnant.date.tolist()
# mans = df_pregnant.is_man.tolist()
# for i in range(df_pregnant.shape[0]):
#     sql_gender = f''' SELECT GENDER as gender FROM andun_app.a_wear_user WHERE ID='{ids[i]}' '''
#     df_gender = ansql.ansql_read_mysql(sql_gender).gender[0]
#     res = ser.GenderResults(iandun_pb2.GenderRequest(ids=ids[i],date=dates[i].strftime(r'%Y-%m-%d')),'')

# stub = Greeter()

# uid = 'c0113092'
# date = '2022-06-19'
# men_start_latest = '2022-06-11'
# men_keep = '5'
# men_cycle = '28'

# sql = 'SELECT wear_user_id as uid,date FROM andun_health.h_pregnant_analysis WHERE is_man = 1 LIMIT 27'
# df_isman = ansql.ansql_read_mysql(sql)


# res = []
# for i in range(df_isman.shape[0]):
#     df = df_isman.iloc[i]
#     uid = df.uid
#     date = df.date.strftime(r'%Y-%m-%d')
#     res.append(stub.GenderResults(iandun_pb2.GenderRequest(ids=uid,date=date),context=''))

# print(res)

# stub = Greeter()
# stub.PhyResults(iandun_pb2.PhyRequest(ids=uid,date=date,men_start_latest=men_start_latest,men_keep=men_keep,men_cycle=men_cycle),context='')
# stub.GenderResults(iandun_pb2.GenderRequest(ids=uid,date=date),context='')