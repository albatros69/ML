#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 sw=4 ts=4 et:

from __future__ import (unicode_literals, absolute_import, print_function, division)

# import re
import gzip
from argparse import ArgumentParser
import pickle

import pandas
import ezodf

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


stop_words = stopwords.words('french')


if __name__ == '__main__':

    Parser = ArgumentParser()
    Parser.add_argument("file", action="store", metavar='FILE',
                        help="Load data from this file.")
    Parser.add_argument("-d", "--dump", dest="dump", action="store", nargs='?', const="model-budget.pkl.gz",
                        help="Dump the model as a Pickle file.")
    Args = Parser.parse_args()

    doc = ezodf.opendoc(Args.file)

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
    sheet = doc.sheets[4] # on ajoute les données Bourso
    for i, row in enumerate(sheet.rows()):
        if i <= 2:
            continue
        for j, cell in enumerate(row):
            if col_index[j]:
                df_dict[col_index[j]].append(cell.value)
    # and convert to a DataFrame
    df = pandas.DataFrame(df_dict)

    # On combine Crédit et Débit en une seule colonne
    df.fillna({'Crédit': 0, 'Débit': 0}, inplace=True)
    df['montant'] = df['Crédit'] - df['Débit']

    for col in [ 'Retour', 'Réel', 'Montant LB', 'Conversion F', 'Débit', 'Crédit', ]:
        df.drop(col, axis=1, inplace=True)
    # Suppression de cette catégorie
    df = df[df['Nature'] != 'Revenus divers']
    df.dropna(axis=0, inplace=True)

    # Transformation de la date en trois colonnes
    df_date = df['Date'].map(lambda x: x[:10].split('-')).apply(pandas.Series).astype(int)\
            .rename(columns=lambda x: {0:'année', 1:'mois', 2:'jour'}[x] )

    # Traitement de la colonne de texte
    vctzr = CountVectorizer(min_df=10, stop_words=stop_words, lowercase=True)
    OpDescr = vctzr.fit_transform(df["Nature de l'opération"].tolist())
    df_OpDescr = pandas.DataFrame(OpDescr.toarray(), columns=vctzr.get_feature_names_out(), index=df.index)
    categ_words = vctzr.get_feature_names_out()

    for col in [ 'Date', "Nature de l'opération", ]:
        df.drop(col, axis=1, inplace=True)

    X = df.drop('Nature', axis=1)
    X = pandas.concat([X, df_OpDescr, df_date], axis=1)
    # print(X.describe())
    Y = df['Nature']

    n_estimators = 50
    if Args.dump:
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        model.fit(X.values, Y.values)

        # from sklearn.metrics import accuracy_score
        # Y_pred = model.predict(X)
        # print("Accuracy (training error): %0.3f" % (accuracy_score(Y, Y_pred, normalize=True), ))

        model.categ_words = categ_words
        model.features_list = list(X)

        fic_name = Args.dump
        if not fic_name.endswith('.gz'):
            fic_name += '.gz'
        with gzip.open(fic_name, 'wb') as dump_file:
            pickle.dump(model, dump_file)
    else:
        # from sklearn.metrics import accuracy_score
        # from sklearn.model_selection import train_test_split
        #X_train, X_test, Y_train, Y_true = train_test_split(X, Y, test_size=0.1, shuffle=True)
        #model.fit(X_train, Y_train)
        #Y_pred = pandas.DataFrame(model.predict(X_test), index=Y_true.index)
        #print(X_test.head())
        #print("Accuracy (testing error): %0.3f" % (accuracy_score(Y_true, Y_pred, normalize=True), ))

        # print(f"# valeurs : {Y.size}", Y.value_counts(), sep='\n')

        from sklearn.model_selection import cross_val_score, ShuffleSplit
        cv = ShuffleSplit(n_splits=20, test_size=0.1)
        scores = cross_val_score(RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1), X, Y,
            scoring='accuracy', cv=cv, n_jobs=-1, verbose=1)
        print("RandomForest CV Accuracy (testing error): %0.3f (± %0.3f)" % (scores.mean(), scores.std()*2))

        # from sklearn.model_selection import validation_curve
        # n_estimators = range(1,100,5) #range(1,16) #range(1,30,2)
        # train_scores, test_scores = validation_curve(
        #     RandomForestClassifier(), X, Y, param_name="n_estimators", param_range=n_estimators,
        #     cv=cv, scoring="accuracy", n_jobs=4, verbose=1)

        # import matplotlib.pyplot as plt
        # plt.plot(n_estimators, train_scores.mean(axis=1), label="Training score")
        # plt.plot(n_estimators, test_scores.mean(axis=1), label="Testing score")
        # plt.legend()

        # plt.xlabel("Number of estimators")
        # plt.ylabel("Accuracy")
        # _ = plt.title("Validation curve for Random Forest")
        # plt.show()

