# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:32:34 2020

@author: igort
"""

import pandas as pd
base  = pd.read_csv('D:\código\python\machine_learning\classificacao\modulo_8\credito_risco2.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])


from sklearn.linear_model import LogisticRegression
classificador = LogisticRegression()
classificador.fit(previsores, classe)
print (classificador.intercept_)
print (classificador.coef_)
# teste1: historico bom, divida alta, garantias nenhuma, renda > 35
# teste2 (correcao laplaciana): historico ruim, divida alta, garantias adequadas, renda < 15
resultado = classificador.predict([ [0,0,1,2], [3,0,0,0] ])
print(resultado)
resultado_proba = classificador.predict_proba([ [0,0,1,2], [3,0,0,0] ])
print(resultado_proba)
# propriedades do naive bayes
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_) 