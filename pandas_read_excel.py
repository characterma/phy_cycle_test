import os
import traceback

import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras as k

from utils import log
from utils.andun_database import Mongo, Sql

anlog = log.MyLogger('./log/micro.log',level='info')
ansql = Sql()
'''
读取excel并进行数据处理
'''
def get_content():
    pass
def getFlist(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print('root_dir:', root)  #当前路径
        print('sub_dirs:', dirs)   #子文件夹
        print('files:', files)     #文件名称，返回list类型
    #return files



def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
def getlabe(x,cls_num,plq=True):
    #-1计算失败,0月经,1卵泡,2排卵,3黄体,4排卵日
    # #未知 月经期、安全期、排卵期、排卵日
    # if np.nan == x:
    #     return None
    x = str(x)
    if x =="月经期" or x=="姨妈期" or x=="生理期" or "月经" in x:
        return 1 if plq and cls_num==2 else 0
    elif "安全期" in x or x == "安全期" or x=="黄体期" or x == "卵泡期":
        return 1
    elif "排卵期" in x or x == "排卵期":
        if cls_num == 2:
            return 0 if plq else 1
        else:
            return 2
        #return 1
    elif x == "排卵日":
        if cls_num == 2:
            return 0 if plq else 1
        else:
            return 2
    elif x == "黄体期":
        return 1
    elif "未知" in x:
        return 1
    else:
        return -1
def get_day_data(path,list_name_index,cls_num):
    excel_path = os.path.join(path, list_name_index)
    df = pd.read_excel(excel_path)
    df['status'] = df["真实状态【月经期、安全期、排卵期、排卵日、适孕期（疑似怀孕）、未知】"].apply(lambda x: getlabe(x,cls_num))
    print(df)
    return df
'''
根据时间获取前几天的状态
'''
def get_last_five_day(df):
    df["pre_1"] = df.apply(lambda x:get_pre(x,-1,df),axis=1)
    df["pre_2"] = df.apply(lambda x: get_pre(x, -2, df), axis=1)
    df["pre_3"] = df.apply(lambda x: get_pre(x, -3, df), axis=1)
    df["pre_4"] = df.apply(lambda x: get_pre(x, -4, df), axis=1)
    df["pre_5"] = df.apply(lambda x: get_pre(x, -5, df), axis=1)
    return df

'''
获取当前时间并推算前一天状态
'''
def get_pre(x,n,df):
    day_time = x["时间"]
    print(n)
    t1 = pd.to_datetime(day_time)+dt.timedelta(days=n)
    print(day_time,t1)
    d = df[df["时间"]==t1]
    print(d)
    if d.empty:
        return 1
    else:
        print(type(d["status"]))
        return d["status"].iloc[0]
    return 1

# 数据处理
def try_fun(str_exec,str_eval,x):
    try:
        exec(str_exec)
        return eval(str_eval)
    except Exception as e:
        # print(e)
        return None
def get_wear_user_id(x,file_name):
    if file_name == x:
        return x["wearuserID"]
    else:
        return None
    pass
def get_ppg_data(uid,data):

    pass
def train_model(uid,ppgs_np_repeat,labels_np,result_arry,all=False,cls_num=2,big_model=False):
    from server import filt_data, training_vis
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
            # layers.Dense(16, activation='relu', kernel_initializer='normal'),
            # layers.BatchNormalization(),
            #layers.Dropout(0.1),
            layers.Dense(cls_num, activation='sigmoid')
            #layers.Dense(cls_num, activation='softmax')
        ])
        model.summary()
        model.compile(
            k.optimizers.Adam(1e-3),
            loss=k.losses.BinaryCrossentropy(),
            #loss=k.losses.CategoricalCrossentropy(),
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
        training_vis(h)
        result_acc = model.evaluate(x_test, y_test)[1]
        result_arry.append([uid,result_acc])
        if result_acc >= 0.7:
            if big_model:
                model.save(f'model_big/model_ppg_micro_{uid}')
            else:
                if all:
                    if cls_num ==3:
                        model.save(f'model_mbi_3/model_ppg_micro_{uid}')
                    else:
                        model.save(f'model_mbi/model_ppg_micro_{uid}')
                else:
                    model.save(f'model_green_2/model_ppg_micro_{uid}')
            anlog.logger.info("ALL{} {} 训练ppg模型完成...保存model/model_ppg_micro_{}".format(all,uid, uid))
        else:
            anlog.logger.warn("{} 训练ppg模型完成...模型精度小于0.7,不保存".format(uid))
    else:
        anlog.logger.warn("{} 训练ppg数据少于100, 数据量{}  不训练模型...".format(uid, ppgs_np_repeat.shape[0]))
    pass
def get_data_set(did,day_data,all=False,cls_num=2,onehot=True):
    from server import filt_data, training_vis
    with Mongo() as mongo:
        #t0 = min_date - dt.timedelta(hours=8)
        #t1 = max_date + dt.timedelta(hours=4)
        df_ppg = mongo.women_all_ppg_did(did)
    df_ppg['createTime'] = df_ppg.createTime.apply(lambda x: x + dt.timedelta(hours=8))
    df_ppg['ct_date'] = df_ppg.createTime.apply(lambda x: x.date())
    df_ppg['hour'] = df_ppg.createTime.apply(lambda x: x.hour)
    df_ppg = df_ppg.query('0 <= hour < 5')
    for i in range(day_data.shape[0]):
        date = day_data["时间"].iloc[i]
        status = day_data.status.iloc[i]
        age = day_data.age.iloc[i]
        temperature = day_data.temperature.iloc[i]
        hr = day_data.hr.iloc[i]
        sleep = day_data.sleep.iloc[i]
        medical_history = day_data.medical_history.iloc[i]
        mbi = day_data.mbi.iloc[i]

        if status !=-1:
            df_ppg.loc[df_ppg.ct_date == date, 'label'] = status
            df_ppg.loc[df_ppg.ct_date == date, 'age'] = age/100.0
            df_ppg.loc[df_ppg.ct_date == date, 'temperature'] = temperature/37.0
            df_ppg.loc[df_ppg.ct_date == date, 'hr'] = hr/100.0
            df_ppg.loc[df_ppg.ct_date == date, 'sleep'] = sleep/21600.0
            df_ppg.loc[df_ppg.ct_date == date, 'medical_history'] = medical_history
            df_ppg.loc[df_ppg.ct_date == date, 'mbi'] = mbi/30.0
    df_ppg.dropna(inplace=True)
    df_ppg['PPG_Green'] = df_ppg['PPG_Green'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
    df_ppg['PPG_Red'] = df_ppg['PPG_Red'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
    df_ppg['PPG_IR'] = df_ppg['PPG_IR'].apply(lambda x: np.fromstring(x[1:-1], float, sep=','))
    #df_ppg = df_ppg.drop(df_ppg[len(df_ppg.PPG_Green.tolist())<11240].index)
    person_data = df_ppg[["hr", "mbi", "age", "medical_history", "temperature", "sleep"]]
    drop_index = []
    # 样本和标签
    ppgs = []
    ppgrs = []
    ppgirs = []
    labels = []
    ppgs_rfilt = df_ppg.PPG_Red.tolist()
    ppgs_irfilt = df_ppg.PPG_IR.tolist()
    ppgs_bfilt = df_ppg.PPG_Green.tolist()
    labels_bfilt = df_ppg.label.tolist()
    for i in range(len(ppgs_bfilt)):
        ppg = ppgs_bfilt[i]
        ppgr =ppgs_rfilt[i]
        ppgir =ppgs_irfilt[i]
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
        ppgs_np = np.concatenate([ppgs_np,ppgs_np1,ppgs_np2],axis=1)
    ppgs_np = (ppgs_np - 1e5) * 50 / (2e6 - 1e5)
    person_data = person_data.drop(person_data.index[drop_index])
    print(ppgs_np.shape,person_data.shape)
    ppgs_np = np.concatenate([ppgs_np,person_data],axis=1)
    labels_np = np.array(labels)
    # 样本量平衡
    counts_1 = (labels_np == 1).sum()
    counts_2 = (labels_np == 2).sum()
    counts_0 = labels_np.size - counts_1-counts_2
    count = max(counts_0,counts_1,counts_2)
    #count = 10
    #anlog.logger.info((labels_np == 0).sum(),(labels_np == 1).sum(),(labels_np == 2).sum())
    #counts_1 = 10


    if cls_num ==2:
        ppgs_np_repeat_0 = ppgs_np[labels_np == 0].repeat(count // counts_0, 0)
        ppgs_np_repeat_2 = ppgs_np[labels_np == 1].repeat(count // counts_1, 0)
        labels_np = np.concatenate([labels_np, labels_np[labels_np == 0].repeat(count // counts_0, 0),
                                    labels_np[labels_np == 1].repeat(count // counts_1, 0)])
        ppgs_np_repeat = np.concatenate((ppgs_np, ppgs_np_repeat_0, ppgs_np_repeat_2))
    else:
        ppgs_np_repeat_0 = ppgs_np[labels_np == 0].repeat(count // counts_0, 0)
        ppgs_np_repeat_1 = ppgs_np[labels_np == 1].repeat(count // counts_1, 0)
        ppgs_np_repeat_2 = ppgs_np[labels_np == 2].repeat(count // counts_2, 0)
        labels_np = np.concatenate([labels_np, labels_np[labels_np == 0].repeat(count // counts_0, 0),labels_np[labels_np == 1].repeat(count // counts_1, 0),labels_np[labels_np == 2].repeat(count // counts_2, 0)])
        ppgs_np_repeat = np.concatenate([ppgs_np, ppgs_np_repeat_0,ppgs_np_repeat_1, ppgs_np_repeat_2],axis=0)

    # 切分训练，验证集
    if onehot:
        labels_np = tf.one_hot(labels_np, cls_num).numpy()
    return ppgs_np_repeat,labels_np
def get_person_data(day_data,womens_msg_info,file_pre_name):
    df = womens_msg_info[womens_msg_info["问题反馈表"] == file_pre_name]
    print(df.wearuserID.iloc[0])
    # day_data = pd.concat(day_data,pd.DataFrame(columns= list('wear_user_id'),fill_value=df["wearuserID"]))
    day_data["wear_user_id"] = [df["wearuserID"].iloc[0]] * len(day_data["status"])
    print(day_data)
    day_data["did"] = [df["设备号"].iloc[0]] * len(day_data["status"])
    min_date = day_data["时间"].iloc[0]
    max_date = day_data["时间"].iloc[-1]
    uid = df["wearuserID"].iloc[0]
    did = df["设备号"].iloc[0]
    return did,uid
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

'''
获取睡眠数据
'''
def get_sleep_data(wear_user_id):
    sql = f"select * from andun_health.h_exercise_sleep_conclusion where T_WEAR_USER_ID='{wear_user_id}' and date>'2022-06-01';"
    return ansql.ansql_read_mysql(sql)
'''
获取年龄及mbi及病史
'''
def get_mbi_age(wear_user_id):
    # 根据wear_user_id 获取mbi数据和age
    medical_history = get_wear_user_info(wear_user_id)
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
    print(x,type(x))
    if pd.isna(x):
        return np.nan
    hr = hr_data.loc[hr_data.date ==dt.datetime(x.year,x.month,x.day).date(),'resting_heart_rate']
    print(hr,type(hr),x)
    if hr.empty:
        print("empty ",hr)
        return np.nan
    else:
        return hr.iloc[0]
    pass
def get_temperature(temperature_data,x):
    if pd.isna(x):
        return np.nan
    t = temperature_data.loc[temperature_data.date ==dt.datetime(x.year,x.month,x.day).date(), 'temperature_data']
    if t.empty:
        print("empty ",t)
        return np.nan
    print(t.iloc[0])

    sleep_array = t.iloc[0].split(";")
    print(sleep_array)
    sleep_map = {}
    sleep_temperature_data = []
    all_temperature = []
    for s in sleep_array:
        t_arry = s.split(",")
        print(t_arry)
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
    print(sleep_json["lightSleepTime"])
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
    day_data['hr'] = day_data["时间"].apply(lambda x:get_hr(hr_data,x))
    day_data['mbi'] = day_data["时间"].apply(lambda x:mbi_data[2])
    day_data['age'] = day_data["时间"].apply(lambda x:mbi_data[1])
    day_data['medical_history'] = day_data["时间"].apply(lambda x:int(str(mbi_data[0])=="1"))
    day_data['temperature'] = day_data["时间"].apply(lambda x:get_temperature(temperature_data,x))
    day_data['sleep'] = day_data["时间"].apply(lambda x:get_sleep_time(sleep_data,x))
    pass
if __name__ == '__main__':
    path = "D:/andun/pregnant_doc_3"
    list_name = []
    listdir(path,list_name)
    print(list_name)
    #print(files_list)

    all_person_file = "D:/andun/all_women/孕妇版手表测试.xlsx"
    all_person_df = pd.read_excel(all_person_file)
    print(all_person_df)
    womens_msg_info = all_person_df[["姓名","设备号","校验码","wearuserID","问题反馈表"]]
    print(womens_msg_info)

    cls_num = 2
    result_arry = []
    error_list = []
    all_data_set = []
    all_label = []

    all = True
    uid = ""
    for list_name_index in list_name:
        if "韩红霞：盖依娅使用问题或体验反馈表" not in list_name_index:
            continue
        try:
            day_data = get_day_data(path, list_name_index, cls_num)
            file_path = list_name_index
            file_name = os.path.split(file_path)
            file_path = os.path.dirname(file_path)
            # 获取前缀
            file_pre_name, ext = os.path.splitext(file_name[1])
            did, uid = get_person_data(day_data, womens_msg_info, file_pre_name)
            if os.path.exists(f'model_mbi/model_ppg_micro_{uid}'):
                continue
            #day_data = day_data[[]]
            day_data.dropna(inplace=True)
            get_women_temperature(day_data,uid)
            m_data = day_data[["hr", "mbi", "age", "medical_history", "temperature", "sleep", "status"]]
            mean_data = np.nanmean(m_data, axis=0)
            day_data = day_data.fillna({"hr":mean_data[0],"mbi":mean_data[1],"age":mean_data[2],"medical_history":mean_data[3],"temperature":mean_data[4],"sleep":mean_data[5]},axis=0,inplace=False)
            print(mean_data)
            print(day_data.isna().any(axis=0))
            day_data = day_data.fillna(0)
            #break

            ppgs_np_repeat, labels_np = get_data_set(str(did), day_data, all, cls_num=cls_num)
            print(ppgs_np_repeat.shape)
            ppgs_np_repeat = ppgs_np_repeat[:,6:]
            print(ppgs_np_repeat.shape)
            all_data_set.append(ppgs_np_repeat)
            all_label.append(labels_np)
            all_d = pd.DataFrame(ppgs_np_repeat).describe(include='all')
            print(all_d)
            train_model(uid, ppgs_np_repeat, labels_np, result_arry, all, cls_num)
        except Exception as e:
            error_list.append(list_name_index)
            anlog.logger.exception(e)
    # try:
    #     train_model(uid, ppgs_np_repeat, labels_np, result_arry, all, cls_num)
    # except Exception as e:
    #     traceback.print_exc()
    print("训练完成")

    print(result_arry)
    print(error_list)
