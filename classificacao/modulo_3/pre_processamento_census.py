# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:27:57 2020

@author: igort
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import 


base = pd.read_csv('D:\código\python\machine_learning\modulo_3\census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
# Transforma os dados do categóricos para dummies
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
