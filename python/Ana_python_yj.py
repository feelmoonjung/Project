import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pymysql
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

# ������ �ε� �� ��ó��

# select-path
SQL_data_sel = 'select file_path from analysis order by num DESC LIMIT 1;'

file_path = pd.read_sql_query(SQL_data_sel, db)

SQL_data_num = 'select num from analysis order by num DESC LIMIT 1;'

file_num = pd.read_sql_query(SQL_data_num, db)

dc_data = pd.read_csv(file_path['file_path'].to_string(index = False), header=[0, 1])

# dc_data = pd.read_csv("data/DieCasting_Raw_Data.csv", encoding='EUC-KR')  # 'EUC-KR' �Ǵ� 'CP949' �õ�

# ���ʿ��� �� ���� �� ������ ó��
dc_data_drop = dc_data.drop(['Shot', '_id'], axis=1)
dc_data_drop = dc_data_drop.dropna().reset_index(drop=True)

# ���������� ���Ӻ��� ����
X = dc_data_drop.drop(columns=['Machine_Status'])
Y = dc_data_drop['Machine_Status']

# ������ �����ϸ�
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# �н�/�׽�Ʈ ������ ����
train_x, test_x, train_y, test_y = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# SMOTE�� ������ ����
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(train_x, train_y)

# ���� ������Ʈ �� ����
rfc = RandomForestClassifier(n_estimators=101, max_depth=25, max_leaf_nodes=25,
                             min_samples_leaf=1, min_samples_split=6, random_state=0)

# ���� �� �Լ�
def get_clf_eval(y_test, pred):
    # print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print('Accuracy:', accuracy_score(y_test, pred))
    print('F1 Score:', f1_score(y_test, pred))
    print('Precision:', precision_score(y_test, pred))
    print('Recall:', recall_score(y_test, pred))

# �� �Ʒ� �� �� �Լ�
def get_model_train_eval(model, ftr_train, ftr_test, tgt_train, tgt_test):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    get_clf_eval(tgt_test, pred)
    return pred

# �� �Ʒ� �� ��
pred = get_model_train_eval(rfc, X_train_over, test_x, y_train_over, test_y)

ana_model = "RandomForestClassifier"
accuracy_s = accuracy_score(test_y, pred)
f1_s = f1_score(test_y, pred, average = 'weighted')
precision_s = precision_score(test_y, pred, average = 'weighted')
recall_s = recall_score(test_y, pred, average = 'weighted')

# 분석시간 추가
now = datetime.now()
ana_time = str(now.microsecond)
save_dir = "C:\\Users\\jungf\\OneDrive\\바탕 화면\\Study\\workspace\\Algo\\result_image\\"
save_path = save_dir + ana_time + '.png'

java_dir = "./result_image/"
java_path =  ana_time +'.png'



val = (accuracy_s, f1_s, recall_s, precision_s, ana_model, java_path)
# val = ("1234", "1234", "1234", "1234")

print(val)

SQL_insert_data = 'insert into analysis_result (accuracy_score, f1_score, recall_score, precision_score, ana_model, result_path) values (%s, %s, %s, %s, %s, %s)'

cursor = db.cursor()
cursor.execute(SQL_insert_data, val)

db.commit()
db.close()
cm = confusion_matrix(test_y, pred)
sns.heatmap(cm, annot = True, cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.plot()
plt.savefig(save_path, format = 'png', dpi = 300, transparent = True)




