# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:22:38 2020

@author: igort
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('D:\código\python\machine_learning\modulo_3\original.csv')
base.describe()
base.loc[base['age'] < 0]

# TÉCNICAS DE TRATAMENTO DE VALORES INCONSISTENTES
# no caso, há três idades negativas

# apagar a coluna de idades
base.drop('age', 1, inplace=True) # 1 = apagar a coluna inteira;

# apagar os registros de clientes com idades negativas
base.drop(base[base.age < 0].index, inplace=True) 

# preencher os valores manualmente (entrar em contato com os clientes)
# entrar em contato com os clientes (inviável quase sempre)
# substituir os valores com a média

base.mean()
base['age'].mean()
base['age'][base.age>0].mean()
base.loc[base.age < 0, 'age'] = 40.92

# TÉCNICAS DE TRATAMENTO DE VALORES FALTANTES
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

# separa os campos entre previsores e o resultado da análise (campo default)
# excluímos o clienteid pois não é um previsor

previsores = base.iloc[:, 1:4].values # do campo 1 até e inclusive o 3
classe = base.iloc[:, 4].values # apenas o campo de classe
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # cria um 
# objeto que corrige NaN utilizando a média 
imputer = imputer.fit(previsores[:, 0:3]) # faz a correção
previsores[:, 0:3] = imputer.transform(previsores[:,0:3]) # aplica a correção


# ESCALONAMENTO DE ATRIBUTOS (padronização dos dados em uma mesma escala)
    # Padronização = x - média / desvio padrão (mais robusta)
    # Normalizacação = x - minimo / maximo - minimo
scaler = StandardScaler ()
previsores = scaler.fit_transform(previsores)