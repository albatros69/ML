#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 sw=4 ts=4 et:

from __future__ import (unicode_literals, absolute_import, print_function, division)

import re
import gzip
import cPickle as pickle

#import flask


with gzip.open('model-budget.pkl.gz', 'rb') as model_dump:
    model_budget = pickle.load(model_dump)


def transform_budget_data(data, categ_words):
    # Transformation de la date
    result = dict(zip([u'Année', u'Mois', u'Jour'], map(int, data['Date'][:10].split('-'))))

    # Calcul du montant de l'opération
    result['Montant'] = data['Crédit'] - data['Débit']

    # Traitement du texte de description (selon les mots utilisés par le modèle)
    words = map(lambda s: s.lower(), re.split('\W+', data["Nature de l'opération"]))
    for w in categ_words:
        if w in words:
            result[w] = 1
        else:
            result[w] = 0

    return result


if __name__ == '__main__':

    data = [
            { "Date": "2017-09-01", "Nature de l'opération": "Bioplaisir",         "Débit":  34.23, "Crédit":  0 },
            { "Date": "2017-09-03", "Nature de l'opération": "Mur de Lyon",        "Débit": 334.23, "Crédit":  0 },
            { "Date": "2017-09-05", "Nature de l'opération": "Remboursement CPAM", "Débit":   0,    "Crédit": 14.2 },
           ]
    for a in data:
        tmp = transform_budget_data(a, model_budget.categ_words)
        result = model_budget.predict([ [ tmp[c] for c in model_budget.features_list ] ])
        print("%s => %s" % (a, result))

