# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:27:57 2020

@author: igort
"""

import pandas as pd

base = pd.read_csv('D:\código\python\machine_learning\classificacao\modulo_3\census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Transforma os dados categóricos em discretos
labelencoder_previsores = LabelEncoder()
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])

"""    
Transforma os dados categóricos para dummies
  
Alguns algoritmos de previsão não funcionam bem com dados categóricos trans-
formados em discretos. A transformação dummy cria uma nova matriz para cada va-
riável categórica. 
Exemplo: Cor dos olhos -> castanho, azul e verde. A transformação para discreto 
resultaria em 0, 1 e 2. Porém, alguns algoritmos classificariam 0 < 1 < 2, o que
não é verdade nesse exemplo. A abordagem dummy funciona assim: 
    pessoa 1 = olhos castanhos -> CASTANHO, AZUL, VERDE 
                                    1        0      0
    pessoa 2 = olhos verdes - >     0        0      1                               
"""
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(),
                                [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# Transforma a classe em dados discretos
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# Escalonamento dos dados 
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão em dados de treinamento e de testes
from sklearn.model_selection import train_test_split
previsores_treinameto, previsores_teste, classe_treinamento, classe_teste = train_test_split(
                        previsores, classe, test_size=0.15, random_state=0)

# APLICAÇÃO DO MÉTODO SVM

from sklearn.svm import SVC
classificador = SVC(kernel='linear', random_state = 1)

# aprendizado 
classificador.fit(previsores_treinameto, classe_treinamento)
# testes
previsoes = classificador.predict(previsores_teste)

# Análise estatística dos resultados
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

# Verificação do Base line
from collections import Counter
Counter(classe_teste)


