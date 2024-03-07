# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 从CSV文件加载数据
data2 = pd.read_csv(r'C:\Users\apple\Desktop\sph6004_assignment1_data.csv',)
data2 = pd.DataFrame(data2)
data2 = data2.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)



# 假设目标变量在 'target' 列中
feature_choose=['weight_admit', 'admission_age', 'bun_min', 'bun_max', 'ptt_max',
       'pao2fio2ratio_min', 'pt_max', 'glucose_max', 'sbp_min', 'glucose_mean',
       'height', 'glucose_min', 'ptt_min', 'pt_min', 'mbp_min',
       'resp_rate_mean', 'glucose_max.2', 'dbp_min', 'temperature_mean',
       'dbp_mean']

X = data2[feature_choose]
y = data2['aki'].apply(lambda x: 0 if x < 2 else 1)

# 数据预处理：标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建 Logistic Regression 模型
logistic_model = LogisticRegression()

# 训练模型
logistic_model.fit(X_train, y_train)

# 进行预测
y_pred = logistic_model.predict(X_test)

# 模型评估
print("Accuracy:", accuracy_score(y_test, y_pred))
# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
print("Classification Report:\n", classification_report(y_test, y_pred))
