import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    data=pd.read_csv(data)
    data=data.dropna()

    q1=data['question1']
    q2=data['question2']
    labels=data['is_duplicate']

    combined=q1 + "[SEP]" + q2

    return  train_test_split(combined,labels,q1,q2,test_size=0.2,random_state=42)
    