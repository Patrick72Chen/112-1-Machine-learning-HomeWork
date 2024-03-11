#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 00:03:07 2023

@author: chenmaige
"""

#1 intall sklearn
#conda install scikit-learn=0.24.2



#2 import套件
import pandas as pd
import numpy as np
import re
import sklearn
from sklearn.utils import resample

#for normalization
from sklearn.preprocessing import StandardScaler

#For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import itertools

#For Metrics evaluation 
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



#3 導入資料
df = pd.read_csv('/Users/chenmaige/Desktop/成大/機器學習/HW1/archive/train.csv')#更改為資料位置路徑



#4 檢查遺失值、丟掉policy_id colum
df.isna().sum() 
df = df.drop('policy_id', axis=1)



#5 畫圖檢視數據是否平衡
fig = plt.figure(facecolor='white')

ax = fig.add_subplot(1, 1, 1, facecolor='white')

plt.pie(df['is_claim'].value_counts(), labels=['No Claim', 'Claim'], radius=1, colors=['green', 'orange'],
        autopct='%1.1f%%', explode=[0.1, 0.15], labeldistance=1.15, startangle=45,
        textprops={'fontsize': 15, 'fontweight': 'bold'})

plt.legend(title='Outcome:', loc='upper right', bbox_to_anchor=(1.6, 1))

fig.patch.set_facecolor('white')

plt.show()



#6 換成torque/rpm、換成power/rpm
df['torque'] = df['max_torque'].apply(lambda x: re.findall(r'\d+\.?\d*(?=Nm)', x)[0])
df['rpm'] = df['max_torque'].apply(lambda x: re.findall(r'\d+\.?\d*(?=rpm)', x)[0])

df['torque'] = pd.to_numeric(df['torque'])
df['rpm'] = pd.to_numeric(df['rpm'])

df['torque_to_rpm_ratio'] = df['torque'] / df['rpm']

df.drop('max_torque', axis=1,inplace=True)
df.drop('rpm',axis=1,inplace=True)
df.drop('torque',axis=1,inplace=True)


df['power'] = df['max_power'].apply(lambda x: re.findall(r'\d+\.?\d*(?=bhp)', x)[0])
df['rpm'] = df['max_power'].apply(lambda x: re.findall(r'\d+', x)[-1])

df['power'] = pd.to_numeric(df['power'])
df['rpm'] = pd.to_numeric(df['rpm'])

df['power_to_rpm_ratio'] = df['power'] / df['rpm']

df.drop('power', axis=1,inplace=True)
df.drop('rpm',axis=1,inplace=True)
df.drop('max_power',axis=1,inplace=True)



#7 把yes/no換成0,1、檢查數字行變數及物件變數
is_cols=[col for col in df.columns if col.startswith("is") and col!="is_claim"]
print(is_cols)

df = df.replace({ "No" : 0 , "Yes" : 1 })


#數字型變數
dataset_num_col = df.select_dtypes(include=['int', 'float']).columns
print(" Data Set Numerical columns:")
print(dataset_num_col.nunique())
print(dataset_num_col)

#物件變數
dataset_cat_cols = df.select_dtypes(include=['object']).columns
print("Data Set categorical columns:")
print(dataset_cat_cols.nunique())
print(dataset_cat_cols)

#將物件變數轉換為二進制形式
df= pd.get_dummies(df, columns=dataset_cat_cols,drop_first=True)




#8處理數據不平衡
majority_class = df[df['is_claim'] == 0]
minority_class = df[df['is_claim'] == 1]

undersampled_majority = resample(
    majority_class,
    replace=False,  
    n_samples=len(minority_class) * 2,  
    random_state=42  
)

# Combine the undersampled majority class with the minority class
df_final = pd.concat([undersampled_majority, minority_class])



#9 標準化連續型變數
#標準化
#population_density
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['population_density']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['population_density'] = column_normalized

#標準化
#displacement
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['displacement']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['displacement'] = column_normalized

#標準化
#length
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['length']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['length'] = column_normalized

#標準化
#width
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['width']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['width'] = column_normalized

#標準化
#height
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['height']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['height'] = column_normalized

#標準化
#gross_weight
scaler = StandardScaler()
# 提取要标准化的列为一个单独的 Series
column_to_normalize = df_final['gross_weight']
# 使用标准化器进行标准化
column_normalized = scaler.fit_transform(column_to_normalize.values.reshape(-1, 1))
#存回來column
df_final['gross_weight'] = column_normalized



#10 分好X, y，並將資料切分
X = df_final.drop('is_claim', axis = 1)
y = df_final['is_claim']

#train=8:test=2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train.value_counts()
y_test.value_counts()



#11 畫出處理完數據不平衡後的圖形
plt.pie(y.value_counts(),labels=['No Claim','Claim'],radius=1.5,colors = ['#FFFACD','#ADD8E6'],
        autopct='%1.1f%%',labeldistance=1.15,startangle =0)

plt.show()


#前處理結束










############################################第一大題#####################################################################

###第一題

#percentron

class PerceptronClassifier:
    def __init__(self, learning_rate=0.1, epochs=100, shuffle=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.shuffle = shuffle
    
    def fit(self, X_train, y_train):
        num_features = X_train.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            if self.shuffle:
                # 如果設置了shuffle，將數據洗牌
                shuffled_indices = np.random.permutation(len(X_train))
                X_train = X_train.iloc[shuffled_indices]
                y_train = y_train.iloc[shuffled_indices]

            for i in range(len(X_train)):
                prediction = np.dot(X_train.iloc[i], self.weights) + self.bias
                y_pred = 1 if prediction >= 0 else 0
                
                if y_pred != y_train.iloc[i]:
                    self.weights += self.learning_rate * (y_train.iloc[i] - y_pred) * X_train.iloc[i]
                    self.bias += self.learning_rate * (y_train.iloc[i] - y_pred)
            
            #print每個epoch之後的訓練準確度
            train_accuracy = np.mean(self.predict(X_train) == y_train)
            print(f"Epoch {epoch + 1}/{self.epochs} - Training Accuracy: {train_accuracy}")

    def predict(self, X_test):
        y_pred_perceptron = (np.dot(X_test, self.weights) + self.bias >= 0).astype(int)
        return y_pred_perceptron

    def evaluate(self, X_test, y_test):
        y_pred_perceptron = self.predict(X_test)
        accuracy_perceptron = np.mean(y_pred_perceptron == y_test)
        return accuracy_perceptron, y_pred_perceptron, self.weights, self.bias



# 宣告模型
perceptron_model = PerceptronClassifier(learning_rate=0.1, epochs=100)

# 訓練模型
perceptron_model.fit(X_train, y_train)

# 評估模型
accuracy_perceptron, y_pred_perceptron, weights, bias = perceptron_model.evaluate(X_test, y_test)
print("Accuracy of perceptron is:", accuracy_perceptron)



########第二大_第一題########################################################
                                                                          # 
# feature importance                                                      # 
feature_importance_percentron = np.abs(weights)                           #  
                                                                          # 
print("Feature Importance of perceptron:", feature_importance_percentron) #            
                                                                          # 
###########################################################################










###第二題


#k-NN(euclidean_distance)

class KNN_euclidean_Classifier:
    def __init__(self, k=3):
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []

        for test_point in X_test:
            distances = [self.euclidean_distance(test_point, train_point) for train_point in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            predicted_label = 1 if sum(nearest_labels) >= self.k / 2 else 0
            y_pred.append(predicted_label)

        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred



# 宣告模型
knn_euclidean_model = KNN_euclidean_Classifier(k=3)

# 訓練模型
knn_euclidean_model.fit(X_train.values, y_train.values)

# 評估模型
accuracy_knn_euclidean, y_pred_knn_euclidean = knn_euclidean_model.evaluate(X_test.values, y_test.values)
print("Accuracy of knn_euclidean:", accuracy_knn_euclidean)






#k-NN(manhattan_distance)

class KNN_manhattan_Classifier:
    def __init__(self, k=3):
        self.k = k

    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []

        for test_point in X_test:
            distances = [self.manhattan_distance(test_point, train_point) for train_point in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            predicted_label = 1 if sum(nearest_labels) >= self.k / 2 else 0
            y_pred.append(predicted_label)

        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred



# 宣告模型
knn_manhattan_model = KNN_manhattan_Classifier(k=3)

# 訓練模型
knn_manhattan_model.fit(X_train.values, y_train.values)

# 評估模型
accuracy_knn_manhattan, y_pred_knn_manhattan = knn_manhattan_model.evaluate(X_test.values, y_test.values)
print("Accuracy of knn_manhattan:", accuracy_knn_manhattan)






#k-NN(chebyshev_distance)

class KNN_chebyshev_Classifier:
    def __init__(self, k=3):
        self.k = k

    def chebyshev_distance(self, x1, x2):
        return np.max(np.abs(x1 - x2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []

        for test_point in X_test:
            distances = [self.chebyshev_distance(test_point, train_point) for train_point in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            predicted_label = 1 if sum(nearest_labels) >= self.k / 2 else 0
            y_pred.append(predicted_label)

        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred



# 宣告模型
knn_chebyshev_model = KNN_chebyshev_Classifier(k=3)

# 訓練模型
knn_chebyshev_model.fit(X_train.values, y_train.values)

# 評估模型
accuracy_knn_chebyshev, y_pred_knn_chebyshev = knn_chebyshev_model.evaluate(X_test.values, y_test.values)
print("Accuracy of knn_chebyshev:", accuracy_knn_chebyshev)










###第三題

#Naïve Decision Tree Classifier

class NaiveDecisionTreeClassifier:
    def __init__(self):
        self.tree = None
        self.feature_importance = None  # 新增特徵重要性屬性
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        # 訓練完模型後，計算特徵重要性
        self.feature_importance = self.calculate_feature_importance(X, y)
    
    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return y.iloc[0]
        if len(X.columns) == 0:
            return y.value_counts().idxmax()
        
        best_feature, best_threshold, best_gini = None, None, 1.0
        
        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                left_indices = X[feature] <= threshold
                right_indices = X[feature] > threshold
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                #檢查索引一致性
                if not all(left_indices.index == y.index) or not all(right_indices.index == y.index):
                    continue
                
                
                left_gini = self.calculate_gini(y[left_indices])
                right_gini = self.calculate_gini(y[right_indices])
                weighted_gini = (len(y[left_indices]) / len(y)) * left_gini + \
                                (len(y[right_indices]) / len(y)) * right_gini
                if weighted_gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = weighted_gini
        
        if best_gini == 1.0:
            return y.value_counts().idxmax()
        
        tree = {'feature': best_feature, 'threshold': best_threshold}
        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold
        
        if len(left_indices) == 0:
            return y[right_indices].value_counts().idxmax()
        if len(right_indices) == 0:
            return y[left_indices].value_counts().idxmax()
        
        tree['left'] = self.build_tree(X[left_indices], y[left_indices])
        tree['right'] = self.build_tree(X[right_indices], y[right_indices])
        
        return tree
    
    def calculate_gini(self, y):
        if len(y) == 0:
            return 1.0
        unique_classes = y.unique()
        gini = 1.0
        total_samples = len(y)
        for c in unique_classes:
            p_i = len(y[y == c]) / total_samples
            gini -= p_i ** 2
        return gini
    
    def calculate_feature_importance(self, X, y):
        num_features = X.shape[1]
        feature_importance = np.zeros(num_features)
        
        for feature in range(num_features):
            thresholds = np.unique(X.iloc[:, feature])
            for threshold in thresholds:
                left_indices = X.iloc[:, feature] <= threshold
                right_indices = X.iloc[:, feature] > threshold
                
                left_gini = self.calculate_gini(y[left_indices])
                right_gini = self.calculate_gini(y[right_indices])
                
                # 計算特徵的基尼重要性
                feature_importance[feature] += (len(y[left_indices]) / len(y)) * left_gini + \
                                               (len(y[right_indices]) / len(y)) * right_gini
        
        return feature_importance


    
    def get_feature_importance(self):
        # 将特征重要性返回为 Pandas Series
        return pd.Series(self.feature_importance, index=X.columns)
    
    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(self.traverse_tree(row, self.tree))
        return pd.Series(y_pred)
    
    def traverse_tree(self, x, tree):
        if type(tree) is not dict:
            return tree
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] <= threshold:
            return self.traverse_tree(x, tree['left'])
        else:
            return self.traverse_tree(x, tree['right'])



# 宣告模型
decision_tree_model = NaiveDecisionTreeClassifier()

# 訓練模型
decision_tree_model.fit(X_train, y_train)

# 預測測試數據集
y_pred_decision_tree = decision_tree_model.predict(X_test)

# 評估模型
accuracy_decision_tree = (y_pred_decision_tree.values == y_test.values).mean()
print(f'Accuracy of Decision Tree: {accuracy_decision_tree:.2f}')



########第二大題_第一題#################################################################
#計算Gini importance。                                                                #                                             
feature_importance_decision_tree = decision_tree_model.get_feature_importance()      #
                                                                                     #
print("Feature Importance:", feature_importance_decision_tree)                       #
                                                                                     #
######################################################################################



########第二大題_第二題####################################################
#計算SHAP值                                                              #      
#conda install  -c conda-forge shap                                     #
import shap                                                             #
                                                                        #
explainer = shap.Explainer(decision_tree_model.predict, X_train)        #
shap_values = explainer(X_train)                                        #
                                                                        #
shap.summary_plot(shap_values, X_train)                                 #
                                                                        #
shap.plots.bar(shap_values, max_display=12)                             #
                                                                        #
#########################################################################










###第四題

#Decision Tree with Pruning

class DecisionTreeWithPruning:
    def __init__(self, max_depth=5):
        self.tree = None
        self.max_depth = max_depth
    
    def fit(self, X, y, depth=0):
        self.tree = self.build_tree(X, y, depth)
    
    def build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return y.iloc[0]
        if len(X.columns) == 0:
            return y.value_counts().idxmax()
        
        best_feature, best_threshold, best_gini = None, None, 1.0
        
        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                left_indices = X[feature] <= threshold
                right_indices = X[feature] > threshold
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                left_gini = self.calculate_gini(y[left_indices])
                right_gini = self.calculate_gini(y[right_indices])
                weighted_gini = (len(y[left_indices]) / len(y)) * left_gini + \
                                (len(y[right_indices]) / len(y)) * right_gini
                if weighted_gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = weighted_gini
        
        if best_gini == 1.0:
            return y.value_counts().idxmax()
        
        tree = {'feature': best_feature, 'threshold': best_threshold}
        left_indices = X[best_feature] <= best_threshold
        right_indices = X[best_feature] > best_threshold
        
        if len(left_indices) == 0:
            return y[right_indices].value_counts().idxmax()
        if len(right_indices) == 0:
            return y[left_indices].value_counts().idxmax()
        
        tree['left'] = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        tree['right'] = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return tree
    
    def calculate_gini(self, y):
        if len(y) == 0:
            return 1.0
        unique_classes = y.unique()
        gini = 1.0
        total_samples = len(y)
        for c in unique_classes:
            p_i = len(y[y == c]) / total_samples
            gini -= p_i ** 2
        return gini
    
    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(self.traverse_tree(row, self.tree))
        return pd.Series(y_pred)
    
    def traverse_tree(self, x, tree):
        if type(tree) is not dict:
            return tree
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] <= threshold:
            return self.traverse_tree(x, tree['left'])
        else:
            return self.traverse_tree(x, tree['right'])



# 宣告模型
decision_tree_pruned_model = DecisionTreeWithPruning(max_depth=5)

# 訓練模型
decision_tree_pruned_model.fit(X_train, y_train)

# 預測測試數據集
y_pred_tree_pruned = decision_tree_pruned_model.predict(X_test)

# 評估模型
accuracy_tree_pruned = (y_pred_tree_pruned.values == y_test.values).mean()
print(f'Accuracy of Tree with Pruning: {accuracy_tree_pruned:.2f}')










##########################################第二大題##########################################################


###第一題

####################################################################
#求取feature importance的方法在前面perceptron & decision tree模型下   #
                                                                  #
###################################################################


# 取最高的12個值
top_values1 = feature_importance_percentron.nlargest(12)
top_values2 = feature_importance_decision_tree.nlargest(12)

# 畫直方圖
# percentron
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=top_values1.index, y=top_values1.values, palette="viridis")
plt.title("Feature importance of percentron")
plt.xlabel("Features")
plt.ylabel("Importance")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()


# decision_tree
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=top_values2.index, y=top_values2.values, palette="viridis")
plt.title("Feature importance of decision tree")
plt.xlabel("Features")
plt.ylabel("Importance")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.show()








###第二題

######################################
#SHAP方法在前面decision tree模型下     #
                                    #
#####################################







 
###第三題

#使用Stacking方法

#再跑一次微更改的KNN-Classifier（因為資料型態緣故修改）
class KNN_euclidean_Classifier_２:
    def __init__(self, k=3):
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []

        for test_point in X_test:
            distances = [self.euclidean_distance(test_point, train_point) for train_point in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = np.array([self.y_train.iloc[i] for i in nearest_indices if i < len(self.y_train)])
            predicted_label = 1 if np.sum(nearest_labels) >= self.k / 2 else 0
            y_pred.append(predicted_label)

        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred



#將Array、Series轉為list    
y_pred_decision_tree_list = y_pred_decision_tree.tolist()
y_pred_tree_pruned_list = y_pred_tree_pruned.tolist()
y_pred_perceptron_list = y_pred_perceptron.tolist()


#製造New Feature X_stacked
X_stacked = np.array([y_pred_knn_euclidean, y_pred_perceptron_list, y_pred_decision_tree_list, y_pred_tree_pruned_list]).T

# 宣告模型
meta_model = KNN_euclidean_Classifier_２(k=3)

# 訓練模型
meta_model.fit(X_stacked, y_test)

# 預測新的predicter
stacked_pred = meta_model.predict(X_stacked)

# 印出新、舊feature的 accuracy 
print ("The Accuracy of our new feature is:", sklearn.metrics.accuracy_score(y_test, stacked_pred))

print ("The Accuracy of our old feature is:", sklearn.metrics.accuracy_score(y_test, y_pred_knn_euclidean))

#結論
print("After stacking, the accuracy did indeed increase compared to the original knn_euclidean model, but the improvement was not very substantial.")










##########################################第三大題############################################################3

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold



###第一題

#k-fold fo perceptron

k_values=[3, 5, 10]

for k in k_values:
    # 宣告模型
    perceptron_classifier = PerceptronClassifier(learning_rate=0.1, epochs=100)
    
    # 初始化 K-fold cross-validation
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize a list to store accuracies for each fold
    accuracies = []
    
    # K-fold cross-validation
    for fold, (train_index, test_index) in enumerate(k_fold.split(X, y), 1):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        # 訓練模型
        perceptron_classifier.fit(X_train_fold, y_train_fold)
        
        # 評估模型
        accuracy, y_pred, _, _ = perceptron_classifier.evaluate(X_test_fold, y_test_fold)
        accuracies.append(accuracy)
        
        # 印出每個fold的Accuracy
        print(f'Fold {fold} - Accuracy for k={k}: {accuracy}')
    
    # Calculate and print the average accuracy for the current k
    avg_accuracy = np.mean(accuracies)
    print(f'Average Accuracy for k={k}: {avg_accuracy}\n')




#f-fold for KNN

k_values = [3, 5, 10]

for k in k_values:
    # 宣告模型
    knn_classifier = KNN_euclidean_Classifier(3)

    # 初始化 K-fold cross-validation
    k_fold = StratifiedKFold(n_splits = k, shuffle=True, random_state=42)

    # Initialize a list to store accuracies for each fold
    accuracies = []

    # K-fold cross-validation
    for fold, (train_index, test_index) in enumerate(k_fold.split(X_train, y_train), 1):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # 訓練模型
        knn_classifier.fit(X_train_fold.values, y_train_fold.values)

        # 評估模型
        accuracy, y_pred = knn_classifier.evaluate(X_test_fold.values, y_test_fold.values)
        accuracies.append(accuracy)

        # 印出每個fold的Accuracy
        print(f'Fold {fold} - Accuracy for k={k}: {accuracy}')

    # Calculate and print the average accuracy for the current k
    avg_accuracy = np.mean(accuracies)
    print(f'Average Accuracy for k={k}: {avg_accuracy}\n')




#k-fold for decision tree

k_values = [3, 5, 10]

for k in k_values:
    # 宣告模型
    decision_tree_classifier = NaiveDecisionTreeClassifier()
    
    # 初始化 K-fold cross-validation
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize a list to store accuracies for each fold
    accuracies = []
    
    # K-fold cross-validation
    for fold, (train_index, test_index) in enumerate(k_fold.split(X_train, y_train), 1):
        X_train_fold, X_test_fold = X_train.iloc[train_index].reset_index(drop=True), X_train.iloc[test_index].reset_index(drop=True)
        y_train_fold, y_test_fold = y_train.iloc[train_index].reset_index(drop=True), y_train.iloc[test_index].reset_index(drop=True)
        
        # 訓練模型
        decision_tree_classifier.fit(X_train_fold, y_train_fold)
        
        # 用trained model做預測
        y_pred = decision_tree_classifier.predict(X_test_fold)
        
        # 計算Accuracy
        accuracy = np.mean(y_pred == y_test_fold)
        accuracies.append(accuracy)
        
        # 印出每個fold的Accuracy
        print(f'Fold {fold} - Accuracy for k={k}: {accuracy}')
    
    # Calculate and print the average accuracy for the current k
    avg_accuracy = np.mean(accuracies)
    print(f'Average Accuracy for k={k}: {avg_accuracy}\n')




#k-fold for decision tree with pruning

k_values = [3, 5, 10]

for k in k_values:
    # 宣告模型
    decision_tree_with_pruning_classifier = DecisionTreeWithPruning(max_depth=5)
    
    # 初始化 K-fold cross-validation
    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize a list to store accuracies for each fold
    accuracies = []
    
    # K-fold cross-validation
    for fold, (train_index, test_index) in enumerate(k_fold.split(X_train, y_train), 1):
        X_train_fold, X_test_fold = X_train.iloc[train_index].reset_index(drop=True), X_train.iloc[test_index].reset_index(drop=True)
        y_train_fold, y_test_fold = y_train.iloc[train_index].reset_index(drop=True), y_train.iloc[test_index].reset_index(drop=True)
        
        # 訓練模型
        decision_tree_with_pruning_classifier.fit(X_train_fold, y_train_fold)
        
        # 用trained model做預測
        y_pred = decision_tree_with_pruning_classifier.predict(X_test_fold)
        
        # 計算Accuracy
        accuracy = np.mean(y_pred == y_test_fold)
        accuracies.append(accuracy)
        
        # 印出每個fold的Accuracy
        print(f'Fold {fold} - Accuracy for k={k}: {accuracy}')
    
    # Calculate and print the average accuracy for the current k
    avg_accuracy = np.mean(accuracies)
    print(f'Average Accuracy for k={k}: {avg_accuracy}\n')










###第二題

# 將X, y轉為 NumPy arrays
X = X.values
y = y.values

# 因為X, y都轉為array了所以以下模型都做了細節修改，麻煩再跑一次！

class PerceptronClassifier:
    def __init__(self, learning_rate=0.1, epochs=100, shuffle=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.shuffle = shuffle
    
    def fit(self, X_train, y_train):
        num_features = X_train.shape[1]
        self.weights = np.zeros(num_features, dtype=float)  
        self.bias = 0.0
        
        for epoch in range(self.epochs):
            if self.shuffle:
                # 創建副本，以免修改原始數據
                indices = np.random.permutation(len(X_train))
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]
            else:
                X_train_shuffled = X_train
                y_train_shuffled = y_train

            for i in range(len(X_train_shuffled)):
                prediction = np.dot(X_train_shuffled[i], self.weights) + self.bias
                y_pred = 1 if prediction >= 0 else 0
                
                if y_pred != y_train_shuffled[i]:
                    self.weights = self.weights + self.learning_rate * (y_train_shuffled[i] - y_pred) * X_train_shuffled[i]
                    self.bias = self.bias + self.learning_rate * (y_train_shuffled[i] - y_pred)
            
            train_accuracy = np.mean(self.predict(X_train_shuffled) == y_train_shuffled)
            #print(f"Epoch {epoch + 1}/{self.epochs} - Training Accuracy: {train_accuracy}")


    def predict(self, X_test):
        y_pred_perceptron = (np.dot(X_test, self.weights) + self.bias >= 0).astype(int)
        return y_pred_perceptron

    def evaluate(self, X_test, y_test):
        y_pred_perceptron = self.predict(X_test)
        accuracy_perceptron = np.mean(y_pred_perceptron == y_test)
        return accuracy_perceptron, y_pred_perceptron, self.weights, self.bias


class KNN_euclidean_Classifier:
    def __init__(self, k=3):
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []

        for test_point in X_test:
            distances = [self.euclidean_distance(test_point, train_point) for train_point in self.X_train]
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            
            # 使用np.bincount来处理平票情况
            predicted_label = np.argmax(np.bincount(nearest_labels))
            
            y_pred.append(predicted_label)

        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred


class NaiveDecisionTreeClassifier:
    def __init__(self):
        self.tree = None
        self.feature_importance = None  # 新增特徵重要性屬性
    
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        self.feature_importance = self.calculate_feature_importance(X, y)
    
    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return y[0]
        if len(X[0]) == 0:
            return np.argmax(np.bincount(y))
        
        best_feature, best_threshold, best_gini = None, None, 1.0
        
        for feature in range(len(X[0])):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                left_gini = self.calculate_gini(y[left_indices])
                right_gini = self.calculate_gini(y[right_indices])
                weighted_gini = (len(y[left_indices]) / len(y)) * left_gini + \
                                (len(y[right_indices]) / len(y)) * right_gini
                if weighted_gini < best_gini:
                    best_feature = feature
                    best_threshold = threshold
                    best_gini = weighted_gini
        
        if best_gini == 1.0:
            return np.argmax(np.bincount(y))
        
        tree = {'feature': best_feature, 'threshold': best_threshold}
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        if len(left_indices) == 0:
            return np.argmax(np.bincount(y[right_indices]))
        if len(right_indices) == 0:
            return np.argmax(np.bincount(y[left_indices]))
        
        tree['left'] = self.build_tree(X[left_indices], y[left_indices])
        tree['right'] = self.build_tree(X[right_indices], y[right_indices])
        
        return tree
    
    def calculate_gini(self, y):
        if len(y) == 0:
            return 1.0
        unique_classes = np.unique(y)
        gini = 1.0
        total_samples = len(y)
        for c in unique_classes:
            p_i = len(y[y == c]) / total_samples
            gini -= p_i ** 2
        return gini
    
    def calculate_feature_importance(self, X, y):
        num_features = len(X[0])
        feature_importance = np.zeros(num_features)
        for feature in range(num_features):
            left_indices = X[:, feature] <= np.median(X[:, feature])
            right_indices = X[:, feature] > np.median(X[:, feature])
            weighted_gini = (len(y[left_indices]) / len(y)) * self.calculate_gini(y[left_indices]) + \
                            (len(y[right_indices]) / len(y)) * self.calculate_gini(y[right_indices])
            feature_importance[feature] = weighted_gini
        return feature_importance
    
    def get_feature_importance(self):
        return self.feature_importance
    
    def predict(self, X):
        y_pred = []
        for row in X:
            # 進行迭代得到預測值
            prediction = self.predict_single_row(self.tree, row)
            y_pred.append(prediction)
        return y_pred
    
    def predict_single_row(self, tree, row):
        if isinstance(tree, np.int64):
            return tree
        if row[tree['feature']] <= tree['threshold']:
            return self.predict_single_row(tree['left'], row)
        else:
            return self.predict_single_row(tree['right'], row)




# 進行三種模型的 merge/aggregate 預測

k_folds = 5  

# 宣告模型
perceptron_model = PerceptronClassifier()
knn_model = KNN_euclidean_Classifier()
tree_model = NaiveDecisionTreeClassifier()

# 創建K-fold分割器
kf = KFold(n_splits=k_folds)

# 初始化數據組來存取預測
perceptron_predictions = np.zeros(len(X))
knn_predictions = np.zeros(len(X))
tree_predictions = np.zeros(len(X))

# K-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 訓練 Perceptron 模型
    perceptron_model.fit(X_train, y_train)
    perceptron_predictions[test_index] = perceptron_model.predict(X_test)

    # 訓練 KNN 模型
    knn_model.fit(X_train, y_train)
    knn_predictions[test_index] = knn_model.predict(X_test)

    # 訓練 Naive Decision Tree 模型
    tree_model.fit(X_train, y_train)
    tree_predictions[test_index] = tree_model.predict(X_test)

# merge/aggregate K=5個分類器的預測结果
final_predictions = np.vstack((perceptron_predictions, knn_predictions, tree_predictions))

# 每一列取投票结果
ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=final_predictions)

# 計算並印出Accuracy
ensemble_accuracy = np.mean(ensemble_predictions == y)
print("Ensemble Accuracy:", ensemble_accuracy)


# 生成classification_report
classification_rep = classification_report(y, ensemble_predictions)
print("Ensemble Classification Report:")
print(classification_rep)









###第三題
#############################
#在Document裡有詳細比較說明

#############################


