#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 sw=4 ts=4 et:

from __future__ import (unicode_literals, absolute_import, print_function, division)

import re
import gzip
import pickle

import flask
from flask import request


with gzip.open('model-budget.pkl.gz', 'rb') as model_dump:
    model_budget = pickle.load(model_dump)


def transform_budget_data(data, categ_words):
    # Transformation de la date
    result = dict(zip(['jour', 'mois', 'année'], [ int(d) for d in data['date'][:10].split('/') ]))

    # Calcul du montant de l'opération
    result['montant'] = float('0'+data['credit'].replace(',', '.')) - float('0'+data['debit'].replace(',', '.'))

    # Traitement du texte de description (selon les mots utilisés par le modèle)
    words = [ s.lower() for s in re.split(r'\W+', data["nature"]) ]
    for w in categ_words:
        if w in words:
            result[w] = 1
        else:
            result[w] = 0

    return result

from app import app

@app.route("/budget/", methods = ('GET', ))
def predict_categ_accounting():
    if all([ c in request.args for c in ('date', 'debit', 'credit', 'nature')]):
        tmp = transform_budget_data(request.args, model_budget.categ_words)
        result = model_budget.predict([ [ tmp[c] for c in model_budget.features_list ] ])
        return ("%s" % (result[0], ), 200)
    else:
        return ('Bad request', 400, )


#if __name__ == '__main__':

    #app.run(port=5999, debug=True, host='localhost')

    #http://localhost:5000/budget/?date=01/09/2017&credit=0&debit=34.23&nature=Bioplaisir
    #http://localhost:5000/budget/?date=05/09/2017&credit=14.2&debit=0&nature=Remboursement+CPAM
    #http://localhost:5000/budget/?date=03/09/2017&credit=0&debit=334.23&nature=Mur+de+Lyon

    #data = [
    #        { "date": "01/09/2017", "nature": "Bioplaisir",         "debit":  "34,23", "credit":  "0" },
    #        { "date": "03/09/2017", "nature": "Mur de Lyon",        "debit": "334,23", "credit":  "0" },
    #        { "date": "05/03/2017", "nature": "Remboursement CPAM", "debit":   "0",    "credit": "14.2" },
    #       ]
    #for a in data:
    #    tmp = transform_budget_data(a, model_budget.categ_words)
    #    result = model_budget.predict_proba([ [ tmp[c] for c in model_budget.features_list ] ])
    #    print("%s => %s" % (a, dict(zip(model_budget.classes_, result[0]))))

