#
import datetime
import os

import pandas as pd

from pandas_read_excel import listdir, get_day_data, get_person_data
from serv import iandun_pb2,iandun_pb2_grpc
import grpc
from utils.andun_database import Mongo, Sql
ansql = Sql()
channel = grpc.insecure_channel('localhost:50070')
#channel = grpc.insecure_channel('39.97.104.203:50070')
'''
根据id获取周期，月经持续时间，上次月经时间
'''
def get_res(id,men_cycle,men_keep,men_start_latest,date):
    #men_cycle = '27'
    #men_keep = '5'
    #date = '2022-06-16'
    #id = 'DMbs7c1y'
    # id = 'aT7PIwF6'
    #men_start_latest = '2022-06-11'
    # %% 链接服务端
    if not isinstance(date,str):
        date = date.strftime(r'%Y-%m-%d')
        men_start_latest = men_start_latest.strftime(r'%Y-%m-%d')
        men_keep = str(men_keep)
        men_cycle = str(men_cycle)
    stub = iandun_pb2_grpc.GreeterStub(channel)
    # 调用rcp服务
    res = stub.PhyResults(
        iandun_pb2.PhyRequest(ids=id, date=date, men_start_latest=men_start_latest, men_cycle=men_cycle,
                                men_keep=men_keep))
    # res = stub.PhyResults(iandun_pb2.PhyRequest(ids=id,date=date))
    # res = stub.GenderResults(iandun_pb2.GenderRequest(ids=id,date=date))
    print(res)
    #res = res.replace("\"message\":","")
    return res.message
def get_all_res(id,date):
    if not isinstance(date,str):
        date = date.strftime(r'%Y-%m-%d')
    stub = iandun_pb2_grpc.GreeterStub(channel)
    # res,date=date
    res = stub.PhySiologicalCycleStage(iandun_pb2.PhySiologicalCycleStageRequest(ids=id,date=date))
    print(res)
    return res.message
    pass
def get_db_data(uid):
    sql = f''' select * from andun_health.h_pregnant_analysis where wear_user_id='{uid}' '''
    df_status = ansql.ansql_read_mysql(sql)
    return df_status
    pass
def get_all_uid():
    all_person_file = "D:/andun/all_women/孕妇版手表测试.xlsx"
    all_person_file = "D:/andun/all_women/孕妇版4p手表测试.xlsx"
    all_person_df = pd.read_excel(all_person_file)
    print(all_person_df)
    womens_msg_info = all_person_df[["姓名", "设备号", "校验码", "wearuserID", "问题反馈表"]]
    return womens_msg_info
    pass
'''
根据excel和数据库里的数据进行
'''
def get_params(day_data,df):
    for i in range(day_data.shape[0]):
        date = day_data["时间"].iloc[i]
        status = day_data.status.iloc[i]
        real_chinese = day_data["真实状态【月经期、安全期、排卵期、排卵日、适孕期（疑似怀孕）、未知】"].iloc[i]
        df.loc[df.date == date,'real'] = status
        df.loc[df.date == date, 'real_chinese'] = real_chinese
    pass
def get_all_predict():
    womens_msg_info = get_all_uid()
    path = "D:/andun/pregnant_doc_8"
    # 获取列表
    list_name = []
    listdir(path, list_name)
    model_path = "model_mbi/model_ppg_micro_"
    cls_num = 2
    date_str = datetime.datetime.now().date().strftime(r'%Y-%m-%d')
    result_path_dir = f"D:/andun/pw_result/{date_str}"
    if not os.path.exists(result_path_dir):
        os.makedirs(result_path_dir)
    for list_name_index in list_name:
        if "刘欢：盖伊娅使用问题或体验反馈表" not in list_name_index:
            continue
        # 读取用户标注的数据
        day_data = get_day_data(path, list_name_index, cls_num)
        file_path = list_name_index
        file_name = os.path.split(file_path)
        file_path = os.path.dirname(file_path)
        # 获取前缀
        file_pre_name, ext = os.path.splitext(file_name[1])
        did, uid = get_person_data(day_data, womens_msg_info, file_pre_name)
        df = get_db_data(uid)
        # if os.path.exists(f'D:/andun/result_3/{file_pre_name}.xlsx'):
        #     continue
        # 根据
        get_params(day_data, df)
        # df.dropna(inplace=True)
        data_person = df[['date', 'wear_user_id', 'real', 'real_chinese', 'menses_duration', 'menses_period',
                          'last_menstruation_date', 'status']]
        print(data_person.shape[0])
        data_person.dropna(subset=['real'], inplace=True)
        new_predict = []
        for d in range(data_person.shape[0]):
            day_str = data_person["date"].iloc[d]
            wear_user_id = data_person["wear_user_id"].iloc[d]
            # if not os.path.exists(f'{model_path}{wear_user_id}'):
            #     continue
            real = data_person["real"].iloc[d]
            real_chinese = data_person["real_chinese"].iloc[d]
            # if data_person["menses_duration"].iloc[d].isnull():
            #     continue
            menses_duration = 7
            menses_period = 39
            try:
                menses_duration = int(data_person["menses_duration"].iloc[d])
                menses_period = int(data_person["menses_period"].iloc[d])
                #last_menstruation_date = data_person["last_menstruation_date"].iloc[d]
                last_menstruation_date = datetime.date.fromisoformat("2022-09-02")
                print(day_str, wear_user_id, real, real_chinese, menses_duration, menses_period, last_menstruation_date)
                res = get_res(wear_user_id, menses_period, menses_duration, last_menstruation_date, day_str)
            except:
                res = -1

            new_predict.append(res)
        try:
            data_person["预测结果"] = new_predict
            result_path = os.path.join(result_path_dir, file_pre_name + ".xlsx")
            data_person.to_excel(result_path)
        except Exception as e:
            print("file_pre_name error ",file_pre_name)
def get_singnal():
    # day_str = "2022-06-13"
    # wear_user_id = "52ceb071"
    # menses_period = "30"
    # menses_duration = "5"
    # last_menstruation_date = "2022-06-01"
    # 使君子
    day_str = "2022-09-24"
    wear_user_id = "0f9908f4"
    menses_period = "28"
    menses_duration = "7"
    last_menstruation_date = "2022-09-07"
    # 孙万玲
    day_str = "2022-10-14"
    wear_user_id = "c2e8f3a7"
    menses_period = "28"
    menses_duration = "7"
    last_menstruation_date = "2022-10-05"

    day_str = "2022-10-14"
    wear_user_id = "mOkREJwv"
    menses_period = "34"
    menses_duration = "6"
    last_menstruation_date = "2022-09-17"

    day_str = "2022-10-14"
    wear_user_id = "c2e8f3a7"
    menses_period = "34"
    menses_duration = "6"
    last_menstruation_date = "2022-09-17"

    day_str = "2022-10-25"
    wear_user_id = "yySqA37q"
    menses_period = "26"
    menses_duration = "5"
    last_menstruation_date = "2022-09-25"

    day_str = "2022-09-25"
    wear_user_id = "pbAlwZiI"
    menses_period = "28"
    menses_duration = "7"
    last_menstruation_date = "2022-09-07"

    day_str = "2022-10-12"
    wear_user_id = "eDVXyLok"
    menses_period = "31"
    menses_duration = "5"
    last_menstruation_date = "2022-09-14"

    day_str = "2022-11-20"
    wear_user_id = "c0113092"
    menses_period = "28"
    menses_duration = "1"
    last_menstruation_date = "2022-10-27"
    last_menstruation_date = "2022-11-04"
    res = get_res(wear_user_id, menses_period, menses_duration, last_menstruation_date, day_str)

'''
预测整个周期的数据
'''
def get_all_model():
    # 调用rpc 预测
    day_str = "2022-11-20"
    wear_user_id = "yySqA37q"
    wear_user_id = "0f9908f4"
    wear_user_id = "eDVXyLok"
    #wear_user_id = "q1oK8PcI"
    wear_user_id = "nF6TrofI"
    day_str = "2022-11-20"
    day_str ="2022-12-05"
    #wear_user_id = "c0113092"
    res = get_all_res(wear_user_id,day_str)
    print(res)
    pass
'''
根据uid获取config
'''
def get_config(uid):
    ansql_test = Sql("test")
    sql = f"SELECT * FROM andun_health.`h_prengnant_config` where wear_user_id = '{uid}'"
    config = ansql_test.ansql_read_mysql(sql)
    if config.empty:
        return None
    else:
        return config
    pass

'''
获取所有测试的结果
'''
def get_prediect_period_by_uid(uid,start_time,end_time):
    # 根据uid 获取数据进行推断
    pass
if __name__ == '__main__':
    # uid = "0KAInDSE"
    # df = get_db_data(uid)
    # print(df)
    #get_all_predict()
    #get_singnal()
    get_all_model()
    # uid = "vHPLweob"
    # config = get_config(uid)
    # if config:
    #     print("暂无配置")
    # print(config.ratio.iloc[0])
    # print(config.ratio.iloc[0])


