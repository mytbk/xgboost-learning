#!/usr/bin/python
import sys
import random

if len(sys.argv) < 3:
    print ('Usage:<filename> <k-test> <k-validate> [nfold = 100]')
    exit(0)

random.seed( 10 )

k1 = int( sys.argv[2] )
k2 = int( sys.argv[3] )
if len(sys.argv) > 4:
    nfold = int( sys.argv[4] )
else:
    nfold = 100

fi = open( sys.argv[1], 'r' )
ftr = open( sys.argv[1]+'.train', 'w' )
ftv = open( sys.argv[1]+'.validate', 'w' )
fte = open( sys.argv[1]+'.test', 'w' )
for l in fi:
	rnd = random.randint(1, nfold)
	if rnd <= k1:
		fte.write( l )
	elif rnd <= k2:
		ftv.write( l )
	else:
		ftr.write( l )

fi.close()
ftr.close()
ftv.close()
fte.close()

