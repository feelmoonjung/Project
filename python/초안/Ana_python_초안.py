import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import seaborn as sns
from sklearn.metrics import f1_score

import warnings; warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import f1_score
import pymysql

X_train = pd.read_csv(r'./trainX.csv')
y_train = pd.read_csv(r'./trainY.csv')

# # 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size = 0.2, random_state = 2)

# 데이터 스케일링
# 변형 객체 생성
scaler = StandardScaler()

# 학습 데이터 스케일링
X_train_scaled = scaler.fit_transform(X_train)

# 모델 생성 및 학습
model_xgbcl = XGBClassifier(n_estimators=20, learning_rate=0.05, max_depth=4, eval_metric = 'error')

model_xgbcl.fit(X_train, y_train)

# db 연동
host_name = '127.0.0.1'
host_port = 3306
username = 'root'
password = '1234'
database_name = 'algo'

db = pymysql.connect(
    host = host_name, 
    port = host_port,
    user = username,
    passwd = password,
    db = database_name
)
# select-path
SQL_data_sel = 'select file_path from analysis order by num DESC LIMIT 1;'

file_path = pd.read_sql_query(SQL_data_sel, db)

SQL_data_num = 'select num from analysis order by num DESC LIMIT 1;'

file_num = pd.read_sql_query(SQL_data_num, db)

X_test = pd.read_csv(file_path['file_path'].to_string(index = False))
X_test

# 테스트 데이터의 스케일링
X_test_scaled =scaler.transform(X_test)

# 모델 예측
y_pred_xc = model_xgbcl.predict(X_test_scaled)
score_xc = f1_score(y_train, y_pred_xc, average = 'macro')
# print(f'불량탐지 XGBOOST f1_score:', score_xc)
print()

# 예측결과 상세정보 확인
from sklearn.metrics import classification_report
y_pred_xc = model_xgbcl.predict(X_test)

report = classification_report(y_train, y_pred_xc, target_names = ['safe', 'danger'])

print(classification_report(y_train, y_pred_xc, target_names = ['safe', 'danger']))

SQL_insert_data = 'insert into analysis_result (result) values (%s);'

cursor = db.cursor()
cursor.execute(SQL_insert_data, report)

db.commit()
db.close()
