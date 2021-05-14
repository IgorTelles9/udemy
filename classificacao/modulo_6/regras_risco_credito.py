# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:32:34 2020

@author: igort
"""

import Orange as og

base  = og.data.Table('D:\código\python\machine_learning\modulo_6\\risco.csv')
base.domain
# Aceita atributos categóricos
cn2_learner = og.classification.rules.CN2Learner()
classificador = cn2_learner(base)

for regras in classificador.rule_list:
    print(regras)

# testes

resultados = classificador([['boa', 'alta','nenhuma', 'acima_35'], 
                           ['ruim', 'alta','adequada', '0_15']])

for resultado in resultados:
    print(base.domain.class_var.values[resultado])