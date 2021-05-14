# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:14:41 2020

@author: igort
"""

import pandas as pd
base  = pd.read_csv('D:\código\python\machine_learning\modulo_3\\risco.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])


from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion="entropy")
classificador.fit(previsores, classe)
print(classificador.feature_importances_)

export.export_graphviz(classificador,
                       out_file='arvore.dot',
                       feature_names=['historia', 'divida',' garantias', 'renda'],
                       class_names=['alto', 'moderado'',baixo'], filled=True,
                       leaves_parallel=True)

export.export_graphviz(classificador,
                       out_file = 'arvore.dot',
                       feature_names = ['historia', 'divida', 'garantias', 'renda'],
                       class_names = ['alto', 'moderado', 'baixo'],
                       filled = True,
                       leaves_parallel=True)


resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])
# propriedades do naive bayes
print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_) 

