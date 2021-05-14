# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:32:34 2020

@author: igort
"""

import Orange as og

base  = og.data.Table('D:\código\python\machine_learning\modulo_6\\credito.csv')
base.domain

# Base Line Classifier
# Simplesmente aplica o resultado da maioria a todos
# Exemplo: classe[0,1] -> [0]:430, [1]: 170 -> A saída é 0 pra qualquer entrada
# Serve como um valor mínimo pro uso de um classificador

base_dividida = og.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]
len(base_treinamento)
len(base_teste)

classificador = og.classification.MajorityLearner()
resultados = og.evaluation.testing.TestOnTestData(base_treinamento,
                                base_teste, [lambda testdata: classificador])
print(og.evaluation.CA(resultados))

from collections import Counter
print(Counter(str(d.get_class()) for d in base_teste))