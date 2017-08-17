#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 sw=4 ts=4 et:

from __future__ import (unicode_literals, absolute_import, print_function, division)

from argparse import ArgumentParser

import pandas
import ezodf

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


