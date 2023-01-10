import datetime as dt
import os
import json
import pickle

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from utils.andun_database import Mongo, Sql

ansql = Sql()

class Women():

    def save_data(self,data,did,date):
        fp = f'''data/dump/{did}_{date}'''
        f = open(fp,'wb')
        pickle.dump(data,f)
        f.close()

    def load_data(self,did,date):
        fp = f'''data/dump/{did}_{date}'''
        with open(fp,'rb') as f:
            return pickle.load(f)

    def serv_his(self,uid):
        did = ansql.did_uid(uid)
        return ansql.women_his(did)

    def ExtractorById(self,uid,date):
        """
            did, date -> list
        """
        did = ansql.did_uid(uid)
        datas_wx = []
        t0 = dt.datetime(date.year,date.month,date.day)-dt.timedelta(hours=8)
        t1 = t0 +dt.timedelta(hours=5)
        with Mongo() as mongo:
            data = mongo.ppg_by_wx(did, t0, t1)
            data = list(data)
            data = [np.fromstring(data[i]['PPG_Green'][1:-1],int,sep=',') for i in range(len(data))]
            datas_wx.extend(data)
        return datas_wx

    def ExtractorByWomen(self,women_his):
        """
            填充datas_wx(需先定义)
        """
        datas_wx = []
        datas_path = 'F:/Datas/未怀孕'
        for i in tqdm(range(women_his.shape[0])):
            his = women_his.iloc[i]
            date = his.create_time
            try:
                datas_wx.extend(self.load_data(his['device_id'],date))
                continue
            except:pass
            status = his.menstruation
            t0 = dt.datetime(date.year,date.month,date.day)-dt.timedelta(hours=8)
            t1 = t0 +dt.timedelta(hours=5)
            if t1 <= dt.datetime(2021,12,31,16):
                data = []
                folder_date = f'''{datas_path}/{his.device_id}/{date.strftime('%Y-%m-%d')}/'''
                try:
                    fnames = os.listdir(folder_date)
                except:
                    continue
                for fname in fnames:
                    with open(folder_date+fname,encoding='utf-8') as f:
                        js = json.load(f)
                        try:
                            data.append(js['pPG_Green'])
                        except:
                            data.append(js['PPG_Green'])
                datas_wx.extend([[np.fromstring(data[i][1:-1],int,sep=','),status] for i in range(len(data))])
            else:
                with Mongo() as mongo:
                    data = mongo.ppg_by_wx(his.device_id,t0,t1)
                    data = list(data)
                data = [[np.fromstring(data[i]['PPG_Green'][1:-1],int,sep=','),status] for i in range(len(data))]
                datas_wx.extend(data)
                self.save_data(data,his['device_id'],date)
        return datas_wx

    def serv_devide_data(self,data, sample_size:int, step:int=0):
        """
            分割data,返回datas
        """
        if not step:
            step=sample_size

        data_list=[]

        for i in range(len(data)):

            try:
                begin=0
                datai = data[i]
                len_datai = len(datai)

                while begin+sample_size <= len_datai:
                    data_list.append(datai[begin:begin+sample_size])
                    begin += step

            except:continue

        return data_list

    def devide_data(self,data, sample_size:int, step:int=0):
        """
            分割data,返回datas,labels
        """
        if not step:
            step=sample_size

        data_list=[]
        label_list=[]

        for i in range(len(data)):

            try:
                begin=0
                datai = data[i]
                len_datai = len(data[i][0])

                if (len(data[i][0]) > 5000):

                    while begin+sample_size <= len_datai:
                        data_list.append(datai[0][begin:begin+sample_size])
                        begin += step
                        label_list.append(datai[1])

            except:continue

        return data_list,label_list

    def just_data(self,data,limit_len = True):
        '''
        数据质量
        '''
        try:

            # if len(data) <9000 and limit_len:
            #     return False

            just = [1 for i in data if (i >= 2e6 or i <= 1e5)]
            n = len(just)
            if n / len(data) >= 0.33:
                return False

            just = signal.medfilt(data,5)
            x = data-just
            x = np.abs(x)
            std1 = np.std(x)
            mean1 = np.mean(x)

            if mean1 >= 400 or mean1 <=40:
                return False
            if std1 >= 300 or std1<= 30:
                return False

            return True
        except:return False

    def filt_data(self,data):
        x = -data
        # x = x[750:]
        x = signal.medfilt(x,5)
        x = x-x[0]
        
        b,a = signal.butter(2,[0.00664,0.01064],'bandpass')
        filt =signal.filtfilt(b,a,x)
        return filt

    def balance_data(self,datas,labels):
        df_labels = pd.DataFrame(labels,columns=['status'])
        inds_lab1 = df_labels[df_labels.status!=0].index.tolist()
        for ind in inds_lab1:
            labels[ind] = 1
        return datas,labels