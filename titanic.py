"""

Kaggle
Titanic - Machine Learning from Disaster
Predict survival on the Titanic and get familiar with ML basics
https://www.kaggle.com/competitions/titanic

Predecir la supervivencia del Titanic
con scikit-learn

"""

import numpy as np
import pandas as pd

#algoritmos

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#archivos de datos y destino

url_test = 'titanic/test.csv'
url_train = 'titanic/train.csv'

dir_test = 'titanic/titanic_test.csv'
dir_train = 'titanic/titanic_train.csv'

df_test = pd.read_csv(url_test)
df_train = pd.read_csv(url_train)

#análisis de datos 

print('Datos cargados train/test: ')
print(df_train.shape)
print(df_test.shape)

#tipo de datos

print('Tipos de datos train/test:')
print(df_train.info())
print(df_test.info())

#datos inválidos/faltantes

print('Datos faltantes:')
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

#estadísticas de los dataset

print('Estadísticas de los dataset:')
print(df_train.describe())
print(df_test.describe())

#datos de columna sexo a números
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)

#datos de origen de embarque en números
df_train['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
df_test['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)

#medias de edad
print(df_train["Age"].mean())
print(df_test["Age"].mean())

#Reemplazo de datos faltantes en la edad por la media calculada en 30
promedio = 30
df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

#Agrupamiento en bandas por edades: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)

#Se elimina la columna de "Cabin"
df_train.drop(['Cabin'], axis = 1, inplace=True)
df_test.drop(['Cabin'], axis = 1, inplace=True)

#Elimino las columnas que considero que no son necesarias para el análisis
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'], axis=1)

#Se elimina las filas con los datos perdidos#Se elimina las filas con los datos perdidos
df_train.dropna(axis=0, how='any', inplace=True)
df_test.dropna(axis=0, how='any', inplace=True)

#Verificación de los datos inconsistentes a 0
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())

print(df_train.shape)
print(df_test.shape)

print(df_test.head(10))
print(df_train.head(10))

#Separar la columna con la información de los sobrevivientes
X = np.array(df_train.drop(['Survived'], 1))
y = np.array(df_train['Survived'])

#Separar los datos de "train" en entrenamiento y prueba para probar algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Algoritmo 1: Regresión logística
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print('Precisión de la Regresión Logística:')
print(logreg.score(X_train, y_train))

#Algoritmo 2: Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print('Precisión Soporte de Vectores:')
print(svc.score(X_train, y_train))

#Algoritmo 3: K neighbors (vecino más cercano)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print('Precisión Vecinos más Cercanos:')
print(knn.score(X_train, y_train))


#realizar la predicción con los modelos creados
ids = df_test['PassengerId']

#Support Vector Machines
prediccion_svc = svc.predict(df_test.drop('PassengerId', axis=1))
out_svc = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_svc })
print('Predicción Soporte de Vectores:')
print(out_svc.head())

#K neighbors
prediccion_knn = knn.predict(df_test.drop('PassengerId', axis=1))
out_knn = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_knn })
print('Predicción Vecinos más Cercanos:')
print(out_knn[['PassengerId', 'Survived']])
print(out_knn.describe())

#Guarda resultado con mayor precisión
out_svc.to_csv(dir_test,index=False)
print(out_svc.shape)

#IMPORTANTE: La preparación de datos elimina el pasajero 1044 por tener una FARE como NA, así que hay que asignar un valor, un promedio de tarifa
print(df_test.iloc[150:155])

