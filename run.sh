#!/bin/bash

./convert_data.py train.csv out.txt
./mknfold.py out.txt 20 40
./train.py
