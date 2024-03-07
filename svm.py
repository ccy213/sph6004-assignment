# 导入所需的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.model_selection import GridSearchCV

from sklearn.datasets import load_iris


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

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建 SVM 分类器
# 定义SVM模型
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

'''# 定义超参数网格
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'C': [0.1, 1, 10, 100],
              'gamma': [0.1, 0.01, 0.001]}

# 使用GridSearchCV进行超参数调优
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 输出最优参数
print("Best Parameters: ", grid_search.best_params_)

# 获取最优模型
best_model = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_model.predict(X_test_scaled)

# 输出分类报告
print("Classification Report:\n", classification_report(y_test, y_pred))'''

# 在训练集上训练模型
svm_classifier.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = svm_classifier.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# 打印分类报告
class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}')
