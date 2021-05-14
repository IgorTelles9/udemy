# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:32:34 2020

@author: igort
"""

import Orange as og

base  = og.data.Table('D:\código\python\machine_learning\modulo_6\\credito.csv')
base.domain
# Aceita atributos categóricos

base_dividida = og.evaluation.testing.sample(base, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

cn2_learner = og.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)

# testes

resultados = og.evaluation.testing.TestOnTestData(base_treinamento,
                                base_teste, [lambda testdata: classificador])
print(og.evaluation.CA(resultados))
