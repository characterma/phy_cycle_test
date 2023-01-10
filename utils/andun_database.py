import pymysql
import datetime
import pandas as pd
from urllib.parse import quote_plus
import pymongo

class Sql():
    def __init__(self, db_type=None):

        if db_type == "test":
            # self.host = '101.200.161.50'
            # self.account = 'root'
            # self.pwd = 'andun12180680#'
            # self.port = 3306
            self.host = '192.168.100.245'
            self.account = 'python_mijunwen'
            self.pwd = 'andun1819'
            self.port = 3307
            self.db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port,
                                      charset='utf8')
            self.cursor = self.db.cursor()
        elif db_type == "pre":
            self.host = '101.200.161.50'
            self.account = 'mijunwen'
            self.pwd = 'KhrapWw*Kjs8'
            self.port = 3306
            self.db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port,
                                      charset='utf8')
            self.cursor = self.db.cursor()
        else:
            # 旧账号
            # self._DB_HOST = 'rm-2ze3zk2327k92186do.mysql.rds.aliyuncs.com'
            # self._DB_ACCOUNT = 'andun_mingkun'
            # self._DB_PASSWORD = 'mpiw9+fMhRMjdMEN'

            # 新账号
            self.host = 'rm-2ze3zk2327k92186do.mysql.rds.aliyuncs.com'
            self.account = 'py_guojiawang'
            self.pwd = r'2rE*msbvgPJhvRa56qNYTG#CC0'
            self.port = 3306
            self.db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port,charset='utf8')
            self.cursor = self.db.cursor()

    @staticmethod
    def cal_age_by_birthday(birthday_str):
        return int((datetime.datetime.now() - datetime.datetime.strptime(str(birthday_str), '%Y-%m-%d')).days / 365)
    def conn(self):
        return pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port, charset='utf8')
    def ansql_read_mysql(self, sql_str):
        db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port,
                             charset='utf8')
        df = pd.read_sql(sql_str, db, params=None)
        db.close()
        return df

    # def ansql_read_mysql_test_db(self, sql_str):
    #     db = pymysql.connect(host=self._TEST_DB_HOST, user=self._TEST_DB_ACCOUNT, password=self._TEST_DB_PASSWORD, port=self._TEST_DB_PORT, charset='utf8')
    #     df = pd.read_sql(sql_str, db, params=None)
    #     db.close()
    #     return df

    def ansql_bp_feature(self, wear_user_id, date):

        if len(date) > 1:
            # sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, date)
            sql_s_t = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(
                wear_user_id, date)
        elif len(date) == 1:

            # sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, date[0])
            sql_s_t = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(
                wear_user_id, date[0])
        else:
            return None

        # res_1 = self.ansql_read_mysql(sql_s)
        res_2 = self.ansql_read_mysql(sql_s_t)

        # if len(res_1) >0:
        #     return res_1
        # elif len(res_2) >0:
        #     return res_2
        # else:
        #     return res_1

        return res_2

    def ansql_bp_feature_train(self, wear_user_id, date):

        if len(date) > 1:
            sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_model WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(
                wear_user_id, date)
        elif len(date) == 1:
            sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_model WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(
                wear_user_id, date[0])
        else:
            return None

        res = self.ansql_read_mysql(sql_s)

        return res

    # def ansql_bp_feature_with_date_range(self, wear_user_id, date_range):

    #     results = pd.DataFrame(columns=['WEAR_USER_ID','FROMPPG','DATE'])

    #     for dr in date_range:
    #         sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, dr)

    #         # if datetime.datetime.strptime(str(date), '%Y-%m-%d') < datetime.datetime.strptime("2020-04-26", '%Y-%m-%d'):
    #         #     sql_s = "SELECT FROMPPG FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, date)
    #         res = self.ansql_read_mysql(sql_s)
    #         if res.empty:
    #             continue
    #         else:
    #             # 拼接到 df_results
    #             results = pd.concat([results, res], ignore_index=True)

    #         time.sleep(0.5)
    #     return results

    def ansql_user_info(self, wear_user_id):
        sql_s = "SELECT BIRTHDAY, STATURE, WEIGHT, GENDER FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None

        # 把BIRTHDAY计算成 年龄
        res['BIRTHDAY'] = self.cal_age_by_birthday(res['BIRTHDAY'].tolist()[0])
        #
        res.rename(columns={'BIRTHDAY': 'Age', 'STATURE': 'Height', 'WEIGHT': 'Weight', 'GENDER': 'Gender'},
                   inplace=True)

        return res

    def ansql_user_age(self, wear_user_id):
        sql_s = "SELECT BIRTHDAY FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return self.cal_age_by_birthday(res['BIRTHDAY'][0])

    def ansql_user_height(self, wear_user_id):
        sql_s = "SELECT STATURE FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return int(res['STATURE'].values[0])

    def ansql_user_weight(self, wear_user_id):
        sql_s = "SELECT WEIGHT FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return int(res['WEIGHT'].values[0])

    def ansql_user_gender(self, wear_user_id):
        sql_s = "SELECT GENDER FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return int(res['GENDER'].values[0])

    def women_his(self,did,date=None):
        sql = f''' select * from andun_collection.women_data where device_id ='{did}' and menstruation != '4' '''
        if date:
            sql = f''' select * from andun_collection.women_data where device_id ='{did}' and create_time='{date}' '''
        res = self.ansql_read_mysql(sql)
        return res

    def uid_did(self,did) -> str:
        """
            获取uid
        """
        sql = f''' 
        SELECT\
            andun_app.t_wear_med_device.A_WEAR_USER_ID as id\
        FROM\
	        andun_app.t_wear_med_device
        WHERE\
	        andun_app.t_wear_med_device.A_DEVICE_ID = {did} '''
        res = self.ansql_read_mysql(sql)
        return res.id[0]

    def did_uid(self,uid) -> str:
        """
            获取did
        """
        sql = f''' 
        SELECT\
            andun_app.t_wear_med_device.A_DEVICE_ID as id\
        FROM\
	        andun_app.t_wear_med_device
        WHERE\
	        andun_app.t_wear_med_device.A_WEAR_USER_ID = '{uid}' '''
        res = self.ansql_read_mysql(sql)
        return res.id[0]

    def ppg_oneday(self,uid,date):
        db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port, charset='utf8')
        sql_str = f''' SELECT FROMPPG AS ppg FROM andun_watch.d_bp_feature WHERE DATE = '{date}' AND WEAR_USER_ID = '{uid}' LIMIT 1 '''
        cursor = db.cursor()
        if cursor.execute(sql_str):
            d =  cursor.fetchone()
            db.close()
            return d
        db.close()
        return None

    def hr_oneday(self,uid,date):
        db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port, charset='utf8')
        sql_str = f''' SELECT HEART_RATE AS hr FROM andun_watch.d_hr_data WHERE WEAR_USER_ID='{uid}' AND DATE='{date}' LIMIT 1 '''
        cursor = db.cursor()
        if cursor.execute(sql_str):
            d =  cursor.fetchone()
            db.close()
            return d
        db.close()
        return None

    def by_sql(self,sql):
        db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port, charset='utf8')
        cursor = db.cursor()
        if cursor.execute(sql):
            d =  cursor.fetchone()
            db.close()
            return d
        db.close()
        return None
    def list_by_sql(self,sql):
        db = pymysql.connect(host=self.host, user=self.account, password=self.pwd, port=self.port, charset='utf8')
        cursor = db.cursor()
        if cursor.execute(sql):
            d = cursor.fetchall()
            db.close()
            return d
        db.close()
        return None
    def h_prengnant_config_insert(self,id,men_start_lastest,men_keep,men_cycle,near_seven_data,ratio,min_ratio):
        sql = f''' insert into andun_health.h_prengnant_config (wear_user_id,men_start_lastest,men_keep,men_cycle,near_seven_data,ratio,min_ratio,create_time) values('{id}','{men_start_lastest}',{men_keep},{men_cycle},'{near_seven_data}',{ratio},{min_ratio},'{datetime.datetime.now()}') '''
        try:
            db = self.conn()
            cursor = db.cursor()
            cursor.execute(sql)
        except:
            db.close()
            self.h_prengnant_config_update(id,id,men_start_lastest,men_keep,men_cycle,near_seven_data)
            return
        db.commit()
        db.close()

    def h_prengnant_config_update(self,id,men_start_lastest,men_keep,men_cycle,near_seven_data):
        sql = f''' update andun_health.h_prengnant_config set men_start_lastest = '{men_start_lastest}',men_keep={men_keep},men_cycle={men_cycle},near_seven_data='{near_seven_data}' where wear_user_id='{id}' '''
        db = self.conn()
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        db.close()
    '''
    本方法对模型自身的数据进行插入或者更新
    '''
    def h_prengnant_data_insert(self,uid,did,date,status_res,ratio,pre_json,model_type,menses_duration,last_menstruation_date,menses_period):
        sql = f''' INSERT into andun_health.`h_pregnant_cycle_data` (wear_user_id,device_id,`date`,status,ratio,pre_json,`type`,menses_duration,last_menstruation_date,
        menses_period,create_time) VALUES('{uid}','{did}','{date}',{status_res},{ratio},'{pre_json}',{model_type},{menses_duration},'{last_menstruation_date}',{menses_period},now()) '''
        try:
            db = self.conn()
            cursor = db.cursor()
            cursor.execute(sql)
        except:
            db.close()
            self.h_prengnant_data_update(uid,date,status_res,last_menstruation_date,menses_duration,menses_period,ratio,pre_json)
            return
        db.commit()
        db.close()
    def h_prengnant_data_update(self,id,date,status,last_menstruation_date,menses_duration,menses_period,ratio,pre_json):
        sql = f''' update andun_health.h_pregnant_cycle_data set last_menstruation_date = '{last_menstruation_date}',status={status},menses_duration={menses_duration},menses_period={menses_period},ratio='{ratio}',pre_json='{pre_json}',update_time=now() where wear_user_id='{id}' and date='{date}' '''
        db = self.conn()
        cursor = db.cursor()
        cursor.execute(sql)
        db.commit()
        db.close()
    def get_all_prengnant_data(self):
        sql = 'select * from andun_health.h_pregnant_cycle_data  order by date desc limit 1'

class Mongo():
    def __init__(self,db_name='aliyun') -> None:
        '''
        db_name:'aliyun' / 'local_suanfa' / 'local'
        '''
        user = "suanfa"
        password = "LpcxQauP17BsVm8cUay&"
        host = "dds-2ze287e358ebf9441164-pub.mongodb.rds.aliyuncs.com:3717/andun_1"

        if db_name=='local_suanfa':
            user = 'andun001'
            password = 'andun1819++'
            host = '39.97.120.189:27017/admin'
            uri = "mongodb://%s:%s@%s" % (
                quote_plus(user),
                quote_plus(password),
                host
                )

            self.client = pymongo.MongoClient(uri)
            self.db = self.client.andun_1
            self.wx = self.db.wx_data_1
            # self.collection = self.db.device_collection_data
        if db_name=='local':

            self.client = pymongo.MongoClient('mongodb://localhost:27017/')
            self.db = self.client.women_2021
            self.wx = self.db.wx_data_2
        else:
            uri = "mongodb://%s:%s@%s" % (
                quote_plus(user),
                quote_plus(password),
                host
                )

            self.client = pymongo.MongoClient(uri)
            self.db = self.client.andun_1
            self.wx = self.db.wx_data
            self.collection = self.db.women_collection_data
            
    def __enter__(self):
        return  self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def ppg_by_wx(self,id,t0,t1):
        '''
            从wx_data获取PPG_Green数据\n
            By设备号&时间区间
        '''
        sql = {
            '$and': [
                {'deviceId': id},
                {'createTime': {'$gte': t0}},
                {'createTime': {'$lte': t1}},
            ]
        }
        key = {
            'PPG_Green',
        }
        return self.wx.find(sql, key)

    def data2show(self,id,tab='wx'):
        '''
        tab: 'wx' / 'collection'
        '''
        sql = {
            'dataId':id
        }
        key={
            'PPG_Green',
            'PPG_Red',
            'PPG_IR',
            'Gsensor_X',
            'Gsensor_Y',
            'Gsensor_Z',
            'createTime'
        }
        if tab == 'wx':
            return self.wx.find_one(sql,key)
        return self.collection.find_one(sql,key)

    def df_read(self,collection,query,key):
        df_read = pd.DataFrame(list(self.db[collection].find(query,key)))
        return df_read

    def women_ppg_did(self,did):
        query = {
            'deviceId':did
        }
        key = {
            'dataId',
            'createTime',
            'PPG_Green',
            'deviceId'
        }
        return self.df_read('women_collection_data',query,key)
    def women_all_ppg_did(self,did):
        query = {
            'deviceId':did
        }
        key = {
            'dataId',
            'createTime',
            'PPG_Green',
            'PPG_Red',
            'PPG_IR',
            'deviceId'
        }
        return self.df_read('women_collection_data',query,key)

    def women_ppg_did_date(self,did,t0,t1):
        query = {
            '$and':[
                {'deviceId':did},
                {'createTime':{'$gte':t0}},
                {'createTime':{'$lte':t1}}
            ]
        }
        key = {
            'dataId',
            'createTime',
            'PPG_Green',
            'deviceId'
        }
        return self.df_read('women_collection_data',query,key)
    '''
    根据时间进行获取ppg原始数据
    '''
    def women_all_ppg_did_date(self,did,t0,t1):
        query = {
            '$and':[
                {'deviceId':did},
                {'createTime':{'$gte':t0}},
                {'createTime':{'$lte':t1}}
            ]
        }
        key = {
            'dataId',
            'createTime',
            'PPG_Green',
            'PPG_Red',
            'PPG_IR',
            'deviceId'
        }
        return self.df_read('women_collection_data',query,key)