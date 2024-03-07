import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso,LassoCV
import seaborn as sns
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#=================================读取数据============================
class Solution():
    def __init__(self):
        feature = ['aki','admission_age','race',
                   'heart_rate_min','heart_rate_max','heart_rate_mean','sbp_min','sbp_max','sbp_mean','dbp_min','dbp_max','dbp_mean','mbp_min','mbp_max','mbp_mean','resp_rate_min','resp_rate_max','resp_rate_mean','temperature_min','temperature_max','temperature_mean','spo2_min','spo2_max','spo2_mean','glucose_min','glucose_max','glucose_mean','lactate_min','lactate_max',
'ph_min','ph_max','so2_min','so2_max','po2_min','po2_max','pco2_min','pco2_max','aado2_min','aado2_max','aado2_calc_min','aado2_calc_max','pao2fio2ratio_min','pao2fio2ratio_max','baseexcess_min','baseexcess_max','bicarbonate_min','bicarbonate_max','totalco2_min','totalco2_max','hematocrit_min','hematocrit_max','hemoglobin_min','hemoglobin_max','carboxyhemoglobin_min','carboxyhemoglobin_max','methemoglobin_min','methemoglobin_max','temperature_min.1','temperature_max.1','chloride_min','chloride_max','calcium_min','calcium_max',
'glucose_min.1','glucose_max.1','potassium_min','potassium_max','sodium_min','sodium_max','hematocrit_min.1','hematocrit_max.1','hemoglobin_min.1','hemoglobin_max.1','platelets_min','platelets_max','wbc_min','wbc_max','albumin_min','albumin_max','globulin_min','globulin_max','total_protein_min','total_protein_max','aniongap_min','aniongap_max','bicarbonate_min.1','bicarbonate_max.1','bun_min','bun_max','calcium_min.1','calcium_max.1','chloride_min.1','chloride_max.1',
'glucose_min.2','glucose_max.2','sodium_min.1','sodium_max.1','potassium_min.1','potassium_max.1','abs_basophils_min','abs_basophils_max','abs_eosinophils_min','abs_eosinophils_max','abs_lymphocytes_min','abs_lymphocytes_max','abs_monocytes_min','abs_monocytes_max','abs_neutrophils_min','abs_neutrophils_max','atyps_min','atyps_max','bands_min','bands_max','imm_granulocytes_min','imm_granulocytes_max','metas_min','metas_max','nrbc_min','nrbc_max','d_dimer_min','d_dimer_max','fibrinogen_min','fibrinogen_max','thrombin_min','thrombin_max','inr_min','inr_max',
'pt_min','pt_max','ptt_min','ptt_max','alt_min','alt_max','alp_min','alp_max','ast_min','ast_max','amylase_min','amylase_max','bilirubin_total_min','bilirubin_total_max','bilirubin_direct_min','bilirubin_direct_max','bilirubin_indirect_min','bilirubin_indirect_max','ck_cpk_min','ck_cpk_max','ck_mb_min','ck_mb_max',
'ggt_min','ggt_max','ld_ldh_min','ld_ldh_max','gcs_min','gcs_motor','gcs_verbal','gcs_eyes','gcs_unable','height','weight_admit']
        self.feature=feature
    def Data_sort(self,file):

        data = pd.read_csv('C:/Users/apple/Desktop/sph6004_assignment2_data .csv',)


# 创建 LabelEncoder 对象
        label_encoder = LabelEncoder()

# 对 'race' 列进行标签编码并替换原有列的内容
        data['race'] = label_encoder.fit_transform(data['race'])
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        #data = pd.DataFrame(data)
        random_state_value = 90  # 随机种子
        sample_number = 82  # 欠采样数目

        data1 = data[self.feature]
        data1 = data1.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)#缺失值补充
        data1['aki'] = data1['aki'].apply(lambda x: 0 if x < 3 else 1)
        
        print(len(data1))
        dataset=data1
        train_dataset = dataset.sample(frac=0.7, random_state=random_state_value)
        test_dataset = dataset.drop(train_dataset.index)
        print(len(test_dataset))
        train_dataset[train_dataset['aki'].isin([1])]=\
            train_dataset[train_dataset['aki'].isin([1])].iloc[:sample_number]
        train_NRA=train_dataset[train_dataset['aki'].isin([0])]
        train_RA=train_dataset[train_dataset['aki'].isin([1])]
        

        train_dataset = pd.concat([train_NRA, train_RA], ignore_index=True)

        train_dataset=train_dataset.sample(frac=1,random_state=0)
        print(len(train_dataset))
        train_labels =train_dataset.pop('aki')
        test_labels =test_dataset.pop('aki')
        return train_dataset,train_labels,test_dataset,test_labels


#=======================Lasso变量筛===============
    def optimal_lambda_value(self):
        Lambdas = np.logspace(-5, 2, 200)    #10的-5到10的2次方
        # 构造空列表，用于存储模型的偏回归系数
        lasso_cofficients = []
        for Lambda in Lambdas:
            lasso = Lasso(alpha = Lambda,max_iter=10000)
            lasso.fit(train_dataset, train_labels)
            lasso_cofficients.append(lasso.coef_)
        # 绘制Lambda与回归系数的关系
        plt.plot(Lambdas, lasso_cofficients)
        # 对x轴作对数变换
        plt.xscale('log')
        # 设置折线图x轴和y轴标签
        plt.xlabel('Lambda')
        plt.ylabel('Cofficients')
        # 显示图形
        plt.show()
        # LASSO回归模型的交叉验证
        lasso_cv = LassoCV(alphas = Lambdas,cv = 10, max_iter=10000)
        lasso_cv.fit(train_dataset, train_labels)
        # 输出最佳的lambda值
        lasso_best_alpha = lasso_cv.alpha_
        print(lasso_best_alpha)
        return lasso_best_alpha

    # 基于最佳的lambda值建模
    def model(self,train_dataset, train_labels,lasso_best_alpha):
        lasso = Lasso(alpha = lasso_best_alpha, max_iter=10000)
        lasso.fit(train_dataset, train_labels)
        return lasso

    def feature_importance(self,lasso):
        # 返回LASSO回归的系数
        dic={'特征':train_dataset.columns,'系数':lasso.coef_}
        df=pd.DataFrame(dic)
        df1=df[df['系数']!=0]
        print(df1)
        coef = pd.Series(lasso.coef_, index=train_dataset.columns)
        imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
        sns.set(font_scale=1.2)
        # plt.rc('font', family='Times New Roman')
        plt.rc('font', family='simsun')
        imp_coef.plot(kind="barh")
        plt.title("Lasso回归模型")
        plt.show()
        return df1

    def prediction(self,lasso):
        # lasso_predict = lasso.predict(test_dataset)
        lasso_predict = np.round(lasso.predict(test_dataset))
        print(sum(lasso_predict==test_labels))
        print(metrics.classification_report(test_labels,lasso_predict))
        print(metrics.confusion_matrix(test_labels, lasso_predict))
        RMSE = np.sqrt(mean_squared_error(test_labels,lasso_predict))
        print(RMSE)
        return RMSE

if __name__=="__main__":
    Object1=Solution()
    train_dataset, train_labels, test_dataset, test_labels=\
        Object1.Data_sort('C:/Users/apple/Desktop/sph6004_assignment2_data .csv')
    lasso_best_alpha = Object1.optimal_lambda_value()
    lasso=Object1.model(train_dataset, train_labels,lasso_best_alpha)
    feature_choose=Object1.feature_importance(lasso)
    RMSE=Object1.prediction(lasso)

