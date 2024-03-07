from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd



# 假设你的数据在一个名为 df 的 DataFrame 中，目标变量为 'aki'
# 你需要准备好自变量 (features) 和目标变量 (target)
# 在这个例子中，假设 X 是特征，y 是目标变量
data2 = pd.read_csv(r'C:\Users\apple\Desktop\sph6004_assignment1_data.csv',)
data2 = pd.DataFrame(data2)
data2 = data2.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)

'''feature_choose=['weight_admit', 'bun_max', 'admission_age', 'bun_min', 'glucose_max',
       'ptt_max', 'pt_max', 'glucose_mean', 'ptt_min', 'temperature_mean',
       'pt_min', 'glucose_min', 'resp_rate_mean', 'spo2_mean', 'sbp_min',
       'sbp_mean', 'dbp_mean', 'glucose_max.2', 'heart_rate_mean', 'mbp_mean']'''
feature_choose=['weight_admit', 'admission_age', 'bun_min', 'bun_max', 'ptt_max',
       'pao2fio2ratio_min', 'pt_max', 'glucose_max', 'sbp_min', 'glucose_mean',
       'height', 'glucose_min', 'ptt_min', 'pt_min', 'mbp_min',
       'resp_rate_mean', 'glucose_max.2', 'dbp_min', 'temperature_mean',
       'dbp_mean']

X = data2[feature_choose]
y = data2['aki'].apply(lambda x: 0 if x < 2 else 1) 

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 在训练集上训练模型
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 打印模型性能指标
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
