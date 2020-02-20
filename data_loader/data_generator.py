import numpy as np
import pandas as pd
import datetime
import os
from sklearn.preprocessing import MinMaxScaler
def csv_read(csv_name,encoding=None):
    raw_data=pd.read_csv(csv_name,encoding=encoding,thousands=',')
    return raw_data[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']]
#    return raw_data[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']]

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        inn=csv_read(r'data_loader/groupby_floor_seoul.csv',encoding='Ansi')
        
        inn = inn.values
        y=[[1 for _ in range(30)] for i in range(len(inn))]
        for i in range(len(inn)):
            cnt=0
            sum=0
            for j in range(30):
                if(inn[i][j]==0):
                    y[i][j]=0
                else:
                    sum+=inn[i][j]
                    cnt+=1
            if(cnt!=0):
                for j in range(30):
                    inn[i][j]=inn[i][j]/(sum/cnt)
        inn= np.array(inn)
        y= np.array(y)
        self.input = inn
        self.y = y
        print(self.input)



    def next_batch(self, batch_size):
#        idx = np.random.choice(500, batch_size)
        idx = np.random.choice(len(self.input), batch_size)
        yield self.input[idx], self.y[idx]

    def get_full_dataset(self):
        return self.input, self.y
