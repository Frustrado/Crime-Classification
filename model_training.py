import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('data/train.csv')

train_df.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

for district in train_df['PdDistrict'].unique():
    train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']] = imp_mean.fit_transform(
        train_df.loc[train_df['PdDistrict'] == district, ['X', 'Y']])
train_df.drop_duplicates(inplace=True)
train_df['Dates'] = pd.to_datetime(train_df['Dates'])

part_Of_Day_dic = {}
for i in range(2,23,4):
    part_Of_Day_dic["hour_"+str(i)] = pd.to_datetime(str(i)+':00:00').time()

def get_part_of_day(time):
    return (
        "Dawn" if part_Of_Day_dic.get('hour_2') <= time < part_Of_Day_dic.get('hour_6')
        else
        "Morning" if part_Of_Day_dic.get('hour_6') <= time < part_Of_Day_dic.get('hour_10')
        else
        "Noon" if part_Of_Day_dic.get('hour_10') <= time < part_Of_Day_dic.get('hour_14')
        else
        "After Noon" if part_Of_Day_dic.get('hour_14') <= time < part_Of_Day_dic.get('hour_18')
        else
        "Evening" if part_Of_Day_dic.get('hour_18') <= time < part_Of_Day_dic.get('hour_22')
        else
        "Night"
    )

train_df['PartOfDay'] = train_df['Dates'].dt.time.apply(get_part_of_day)


def extract_feature(dataFrame):
    dataFrame['Year'] = dataFrame['Dates'].dt.year
    dataFrame['Month'] = dataFrame['Dates'].dt.month
    dataFrame['Day'] = dataFrame['Dates'].dt.day
    dataFrame['Hour'] = dataFrame['Dates'].dt.hour
    dataFrame['Minute'] = dataFrame['Dates'].dt.minute
    dataFrame['Block'] = dataFrame['Address'].str.contains('block', case=False)

    dataFrame.drop(columns=['Dates', 'Address'], inplace=True)
    return dataFrame


train_df = extract_feature(train_df)
train_df.drop(columns=['Descript','Resolution'], inplace=True)

train_df.drop(columns=['Day','Minute'], inplace=True)

train_df_encoded = pd.get_dummies(train_df, columns=["PdDistrict", "DayOfWeek", "PartOfDay"])

train_df_encoded["X_Hour"] = np.cos((train_df_encoded["Hour"]/24)*360)
train_df_encoded["Y_Hour"] = np.sin((train_df_encoded["Hour"]/24)*360)


train_df_encoded["Block"] = train_df_encoded["Block"].astype(int)


le  = LabelEncoder()
y = le.fit_transform(train_df_encoded.pop('Category'))

train_df_encoded.drop(columns=['Hour'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train_df_encoded, y, test_size=0.3, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

models = []
models.append(('LR',LogisticRegression(),{'C':[0.01,0.1,1]}))
models.append(('KNN',KNeighborsClassifier(),{'n_neighbors':[2,3,4,5,6,7,8,9,10,15,20,30,40]}))
models.append(('RFC',RandomForestClassifier(),{'max_depth':[2,5,6,7,8,9,10,15,20,30,40],'n_estimators':[50,100,200]}))

column_names = ["Name", "TrainAccuracy", "TestAccuracy", "BestParameters"]
models_df = pd.DataFrame(columns = column_names)
i = 0
for name,model,parameters in models:
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train_std, y_train)
    models_df.loc[0 if pd.isnull(models_df.index.max()) else models_df.index.max() + 1] = [name,accuracy_score(y_train, clf.predict(X_train_std)), accuracy_score(y_test, clf.predict(X_test_std)), clf.best_params_]
print(models_df.head())
