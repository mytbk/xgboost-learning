#!/usr/bin/env python2

import pandas
import sys
from types import *

def getDataFromCSV(csv):
        table = pandas.read_csv(csv)
        column_names = table.columns
        working = pandas.DataFrame()

        for col in column_names:
	        print(col)
	        if type(table[col][0]) is not StringType:
		        working = pandas.concat([working, table[col]], axis=1)
	        else:
		        dummy = pandas.get_dummies(table[col], prefix=col)
		        working = pandas.concat([working, dummy], axis=1)

        return working

if __name__ == '__main__':
        working = getDataFromCSV(sys.argv[1])
        fo = open(sys.argv[2], 'w')
        for row in working.values:
	        fo.write('%d' % row[1])
	        data = row[2:]
	        for i in range(0,len(data)):
		        if data[i] != 0:
			        fo.write(' %d:%d' % (i+1, data[i]))
	        fo.write('\n')

        fo.close()

