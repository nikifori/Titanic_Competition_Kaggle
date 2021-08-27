"""
Created on Mon Jul 26 23:09:02 2021

@author: Konstantinos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_X = train_data.drop(columns=["PassengerId", "Ticket"])

test_X = test_data.drop(columns=["PassengerId", "Ticket"])
test_PassengerId = test_data.loc[:, ["PassengerId"]]


for i in enumerate(train_X["Name"]):
    if "Mrs" in i[1]:
        train_X.iloc[i[0], 2] = "Mrs"
    elif "Mr" in i[1]:
        train_X.iloc[i[0], 2] = "Mr"
    elif "Miss" in i[1]:
        train_X.iloc[i[0], 2] = "Miss"
    elif "Master" in i[1]:
        train_X.iloc[i[0], 2] = "Master"
    else:
        train_X.iloc[i[0], 2] = "None"

for i in enumerate(test_X["Name"]):
    if "Mrs" in i[1]:
        test_X.iloc[i[0], 1] = "Mrs"
    elif "Mr" in i[1]:
        test_X.iloc[i[0], 1] = "Mr"
    elif "Miss" in i[1]:
        test_X.iloc[i[0], 1] = "Miss"
    elif "Master" in i[1]:
        test_X.iloc[i[0], 1] = "Master"
    else:
        test_X.iloc[i[0], 1] = "None"

train_X = train_X.replace("female", 0)
train_X = train_X.replace("male", 1)

test_X = test_X.replace("female", 0)
test_X = test_X.replace("male", 1)

age_imputer = IterativeImputer(missing_values=np.nan, min_value=0, max_value=80, random_state=42)
train_X["Age"] = age_imputer.fit_transform(
    train_X["Age"].to_numpy().reshape(-1, 1)).round()
test_X["Age"] = age_imputer.transform(
    test_X["Age"].to_numpy().reshape(-1, 1)).round()


train_X = train_X[~train_X["Age"].isna()]
train_X = train_X.reset_index(drop=True)

train_X = train_X[~train_X["Embarked"].isna()]
train_X = train_X.reset_index(drop=True)

train_X["Cabin"] = train_X["Cabin"].replace(np.nan,0)
train_X["Cabin"] = train_X["Cabin"].replace(".*",1,regex=True)

test_X["Cabin"] = test_X["Cabin"].replace(np.nan,0)
test_X["Cabin"] = test_X["Cabin"].replace(".*",1,regex=True)

fare_imputer = IterativeImputer(random_state=42)
test_X["Fare"] = fare_imputer.fit_transform(test_X["Fare"].to_numpy().reshape(-1,1)).round()

# train_X.dtypes
train_Y = train_X["Survived"]
train_X = train_X.drop(columns=["Survived"])

final_train_X = train_X.select_dtypes(exclude=[object])
final_test_X = test_X.select_dtypes(exclude=[object])

object_columns = train_X.select_dtypes(include=[object]).columns

for i in object_columns:
    encoder = OrdinalEncoder()
    label_encoded = encoder.fit_transform(pd.DataFrame(train_X[i]))
    encoder_1hot = OneHotEncoder()
    label_encoded_1hot = encoder_1hot.fit_transform(label_encoded.reshape(-1,1))
    columns = encoder.categories_[0]
    temp_df = pd.DataFrame(data=label_encoded_1hot.toarray(), columns=columns)
    final_train_X = pd.concat([final_train_X, temp_df], axis=1)
    
    label_encoded = encoder.transform(pd.DataFrame(test_X[i]))
    label_encoded_1hot = encoder_1hot.transform(label_encoded.reshape(-1,1))
    temp_df = pd.DataFrame(data=label_encoded_1hot.toarray(), columns=columns)
    final_test_X = pd.concat([final_test_X, temp_df], axis=1)

final_train_X.describe()

# Age and Fare segmentation

# age_data_0 = final_train_X[train_Y==0]["Age"]
# plt.subplot(1,2,1)
# plt.title("Age distribution Survived=0")
# plt.axis([0,100,0,400])
# plt.xlabel("Age")
# plt.ylabel("Number of humans")
# histo = plt.hist(age_data_0, bins=[0, 16, 32, 48, 64, 80], color="blue")


# age_data_1 = final_train_X[train_Y==1]["Age"]
# plt.subplot(1,2,2)
# plt.title("Age distribution Survived=1")
# plt.axis([0,100,0,400])
# plt.xlabel("Age")
# plt.ylabel("Number of humans")
# histo2 = plt.hist(age_data_1, bins=[0, 16, 32, 48, 64, 80], color="red")

# fare_data_0 = final_train_X[train_Y==0]["Fare"]
# plt.subplot(1,2,1)
# plt.title("Fare distribution Survived=0")
# plt.axis([0,600,0,600])
# plt.xlabel("Fare")
# plt.ylabel("Number of humans")
# histo3 = plt.hist(fare_data_0, bins=[0, 128.082, 256.165, 384.247, 513], color="blue")


# fare_data_1 = final_train_X[train_Y==1]["Fare"]
# plt.subplot(1,2,2)
# plt.title("Fare distribution Survived=1")
# plt.axis([0,600,0,600])
# plt.xlabel("Fare")
# plt.ylabel("Number of humans")
# histo4 = plt.hist(fare_data_1, bins=[0, 128.082, 256.165, 384.247, 513], color="red")

for i in enumerate(final_train_X["Age"]):
    if i[1]>=0 and i[1]<16:
        final_train_X.iloc[i[0], 2] = 0
    elif i[1]>=16 and i[1]<32:
        final_train_X.iloc[i[0], 2] = 1
    elif i[1]>=32 and i[1]<48:
        final_train_X.iloc[i[0], 2] = 2
    elif i[1]>=48 and i[1]<64:
        final_train_X.iloc[i[0], 2] = 3
    else:
        final_train_X.iloc[i[0], 2] = 4

for i in enumerate(final_test_X["Age"]):
    if i[1]>=0 and i[1]<16:
        final_test_X.iloc[i[0], 2] = 0
    elif i[1]>=16 and i[1]<32:
        final_test_X.iloc[i[0], 2] = 1
    elif i[1]>=32 and i[1]<48:
        final_test_X.iloc[i[0], 2] = 2
    elif i[1]>=48 and i[1]<64:
        final_test_X.iloc[i[0], 2] = 3
    else:
        final_test_X.iloc[i[0], 2] = 4

for i in enumerate(final_train_X["Fare"]):
    if i[1]>=0 and i[1]<128.082:
        final_train_X.iloc[i[0], 5] = 0
    elif i[1]>=128.082 and i[1]<256.165:
        final_train_X.iloc[i[0], 5] = 1
    elif i[1]>=256.165 and i[1]<384.247:
        final_train_X.iloc[i[0], 5] = 2
    else:
        final_train_X.iloc[i[0], 5] = 3
        
for i in enumerate(final_test_X["Fare"]):
    if i[1]>=0 and i[1]<128.082:
        final_test_X.iloc[i[0], 5] = 0
    elif i[1]>=128.082 and i[1]<256.165:
        final_test_X.iloc[i[0], 5] = 1
    elif i[1]>=256.165 and i[1]<384.247:
        final_test_X.iloc[i[0], 5] = 2
    else:
        final_test_X.iloc[i[0], 5] = 3

# check correlation
# corel_df = pd.concat([final_train_X, train_Y], axis=1)
# for i in corel_df.columns.drop(["Survived"]):
#     print("\n")
#     print(corel_df[[i, "Survived"]].groupby([i], as_index=False).mean(). \
#     sort_values(by="Survived", ascending=False))