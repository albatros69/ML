#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 sw=4 ts=4 et:

from __future__ import (unicode_literals, absolute_import, print_function, division)

from argparse import ArgumentParser

import re

import pandas
import numpy
import ezodf

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score, train_test_split


stop_words = stopwords.words('french')
tknzr = RegexpTokenizer('\w+')


#def transform_text(t):
#    result = [ a.lower() for a in tknzr.tokenize(t)
#            if (len(a)>1 and a.lower() not in stop_words) ]
#    #result.sort()
#    return result


if __name__ == '__main__':

    Parser = ArgumentParser()
    Parser.add_argument("file", action="store", metavar='FILE',
                        help="Load data from this file.")
    Args = Parser.parse_args()

    doc = ezodf.opendoc(Args.file)
    sheet = doc.sheets[1]

    #print("Spreadsheet contains %d sheet(s)." % len(doc.sheets))
    #for sheet in doc.sheets:
    #    print("-"*40)
    #    print("   Sheet name : '%s'" % sheet.name)
    #    print("Size of Sheet : (rows=%d, cols=%d)" % (sheet.nrows(), sheet.ncols()) )

    # convert the first sheet to a pandas.DataFrame
    sheet = doc.sheets[1]
    df_dict = {}
    for i, row in enumerate(sheet.rows()):
        # row is a list of cells
        # the header is on the third row
        if i < 2:
            continue
        elif i == 2:
            # columns as lists in a dictionary
            df_dict = {cell.value:[] for cell in row if cell.value}
            # create index for the column headers
            col_index = {j:cell.value for j, cell in enumerate(row)}
            continue
        for j, cell in enumerate(row):
            # use header instead of column index
            if col_index[j]:
                df_dict[col_index[j]].append(cell.value)
    # and convert to a DataFrame
    df = pandas.DataFrame(df_dict)

    # On combine Crédit et Débit en une seule colonne
    df['Crédit'].fillna(value=0, inplace=True)
    df['Débit'].fillna(value=0, inplace=True)
    df['Montant'] = df['Crédit'] - df['Débit']

    for col in [ 'Retour', 'Réel', 'Montant LB', 'Conversion F', 'Débit', 'Crédit', ]:
        df.drop(col, axis=1, inplace=True)
    # Suppression de cette catégorie
    df = df[df['Nature'] != 'Revenus divers']
    df.dropna(axis=0, inplace=True)

    # Transformation de la date en trois colonnes
    df_date = df['Date'].map(lambda x: x[:10].split('-')).apply(pandas.Series).astype(int)\
            .rename(columns=lambda x: {0:'Année', 1:'Mois', 2:'Jour'}[x] )

    # Traitement de la colonne de texte
    #df_OpDescr = df["Nature de l'opération"].map(transform_text).apply(pandas.Series).rename(columns = lambda x : 'mot_' + str(x))
    vctzr = CountVectorizer(min_df=10, stop_words=stop_words, lowercase=True)
    OpDescr = vctzr.fit_transform(df["Nature de l'opération"].tolist())
    df_OpDescr = pandas.DataFrame(OpDescr.A, columns=vctzr.get_feature_names(), index=df.index)
    categ_words = vctzr.get_feature_names()

    for col in [ 'Date', "Nature de l'opération", ]:
        df.drop(col, axis=1, inplace=True)

    # Traduction des catégories en index
    #categories = df['Nature'].unique()
    #categories = dict(zip(range(len(categories)), categories))

    X = df.drop('Nature', axis=1)
    X = pandas.concat([X, df_OpDescr, df_date], axis=1)
    #Y = df['Nature'].map(lambda x: categories.values().index(x)).astype(int)
    #le = LabelEncoder()
    #Y = le.fit_transform(df['Nature'])
    Y = df['Nature']

    model = RandomForestClassifier()
    #X_train, X_test, Y_train, Y_true = train_test_split(X, Y, test_size=0.1)
    #model.fit(X_train, Y_train)
    #Y_pred = pandas.DataFrame(model.predict(X_test), index=Y_true.index)
    #print(X_test.head())
    #print("Accuracy: %0.3f" % (accuracy_score(Y_true, Y_pred, normalize=True), ))

    #scores = cross_val_score(XGBClassifier(n_estimators=20), X, Y, scoring='accuracy', cv=20, n_jobs=-1, verbose=1)
    #print("XGB CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #scores = cross_val_score(RandomForestClassifier(n_estimators=20), X, Y, scoring='accuracy', cv=20, n_jobs=4, verbose=1)
    #print("RF CV Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    model.fit(X, Y)

    data = [
            { "Date": "2017-09-01", "Nature de l'opération": "Bioplaisir",         "Débit":  34.23, "Crédit":  0 },
            { "Date": "2017-09-03", "Nature de l'opération": "Mur de Lyon",        "Débit": 334.23, "Crédit":  0 },
            { "Date": "2017-09-05", "Nature de l'opération": "Remboursement CPAM", "Débit":   0,    "Crédit": 14.2 },
           ]
    for a in data:
        tmp = dict(zip([u'Année', u'Mois', u'Jour'], map(int, a['Date'][:10].split('-'))))
        tmp['Montant'] = a['Crédit'] - a['Débit']
        #tmp['Montant'] = getattr(a, 'Crédit', 0.0) - getattr(a, 'Débit', 0.0)
        words = map(lambda s: s.lower(), re.split('\W+', a["Nature de l'opération"]))
        for w in categ_words:
            if w in words:
                tmp[w] = 1
            else:
                tmp[w] = 0
        result = model.predict([ [ tmp[c] for c in list(X) ] ])
        print("%s => %s" % (a, result))
