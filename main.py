import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,recall_score,classification_report,confusion_matrix
df=pd.read_csv("heart.csv")
df.fillna(df.median(),inplace=True)
for col in df.columns:
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    df[col]=df[col].where(df[col].between(lower,upper),df[col].median())
x=df.drop('target',axis=1)
y=df['target']
X_train,X_temp,y_train,y_temp=train_test_split(x,y,test_size=0.4,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)
X_test=scaler.transform(X_test)
pca=PCA(n_components=0.95)
X_train=pca.fit_transform(X_train)
X_val=pca.transform(X_val)
X_test=pca.transform(X_test)
ks=[3,5,7,9,11]
max_recall=0
best=1
for k in ks:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_val)
    accu=accuracy_score(y_val,y_pred)
    recal=recall_score(y_val,y_pred)
    if(recal>max_recall):
        max_recall=recal
        best=k
knn1=KNeighborsClassifier(n_neighbors=best)
knn1.fit(X_train,y_train)
y_test_pred=knn1.predict(X_test)
print(accuracy_score(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))
print(classification_report(y_test,y_test_pred))