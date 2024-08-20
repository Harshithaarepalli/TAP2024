import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from collections import Counter
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")
import time

def transform_data(data):
    global df
    df=data.copy()
    df['trans_date_trans_time']=pd.to_datetime(df['trans_date_trans_time'])
    df['trans_date']=df['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
    df['trans_date']=pd.to_datetime(df['trans_date'])
    df['dob']=pd.to_datetime(df['dob'])
    df['trans_day'] = df['trans_date_trans_time'].dt.day_name()
    df['trans_month'] = df['trans_date_trans_time'].dt.month_name()
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_date'] = df['trans_date_trans_time'].dt.date
    df['cust_age'] = np.round((df['trans_date_trans_time'] - df['dob'])/np.timedelta64(365,'D')).astype('int')
    df['cust_age_group'] = pd.cut(df['cust_age'],bins=[10,20,30,40,50,60,1000],labels=['10-20', '20-30', '30-40', '40-50', '50-60', '60 - Above'])
    return df


def encoding(data):
    le=LabelEncoder()
    data['merchant']=le.fit_transform(data['merchant'])
    data['job']=le.fit_transform(data['job'])
    data['category']=le.fit_transform(data['category'])
    data['gender']=le.fit_transform(data['gender'])
    data['trans_day']=le.fit_transform(data['trans_day'])
    data['state']=le.fit_transform(data['state'])
    return data

def drop_cols(data):
    extra_cols=['trans_date_trans_time','dob','cc_num', 'city', 'cust_age_group', 'dob', 'first', 'last', 'lat', 'long', 'merch_lat',
                'merch_long', 'street', 'trans_date', 'trans_date_trans_time', 'trans_month', 'trans_num', 'unix_time', 'zip']
    data.drop(extra_cols,axis=1,inplace=True)
    return data

def scaling(X):
    scaler=StandardScaler()
    scaler.fit(X)
    return X



def scores(y,data):
    print('Random Forest\nAccuracy Score:',accuracy_score(y,data))
    print('\nPrecision Score:',precision_score(y,data))
    print('\nRecall Score:',recall_score(y,data))
    print('\nF1 Score:',f1_score(y,data))
    print('\nConfusion Matrix:\n',confusion_matrix(y,data))
    print('\nClassification report:\n',classification_report(y,data))
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(y,data)
    auc_score_lr=roc_auc_score(y,data)
    def plot(fpr,tpr,auc):
        plt.figure(figsize=(5,3))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
    plot(fpr_lr,tpr_lr,auc_score_lr)