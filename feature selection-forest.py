import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np


n=20

# 读取CSV文件
df = pd.read_csv(r'C:\Users\apple\Desktop\sph6004_assignment1_data.csv')


df['aki'] = df['aki'].apply(lambda x: 0 if x < 2 else 1)


# 假设第二列是目标变量，第三列往后是特征
y = df.iloc[:, 1]   # 目标变量
X = df.iloc[:, 2:]  # 特征

# 创建随机森林分类器
rf_classifier = RandomForestClassifier()

# 训练模型
rf_classifier.fit(X, y)

# 获取特征重要性
feature_importances = rf_classifier.feature_importances_

# 获取特征重要性排名前几的特征
top_n_features = np.argsort(feature_importances)[::-1][:n]

# 打印排名前几的特征
selected_features = X.columns[top_n_features]
print(f"Selected Features: {selected_features}")

# 使用排名前几的特征重新构建 X
X_selected = X[selected_features]

# 重新训练模型
rf_classifier.fit(X_selected, y)

# 将特征重要性与特征名字对应起来
feature_importance_dict = dict(zip(X.columns, feature_importances))

# 打印特征重要性
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")

# 将特征重要性可视化
feature_importance_series = pd.Series(feature_importance_dict)
feature_importance_series.sort_values(ascending=False).plot(kind='barh')
plt.title('Feature Importance')
plt.show()
