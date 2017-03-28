'''
  Myanna Harris
  Jasmine Jans
  11-18-16
  asgn6_test.py

  Probabilistic part-of-speech tagger using Brown Corpus
  Makes A and B matrices

  To run:
  python asgn6_test.py Sentence to tag
'''

import nltk
from nltk.corpus import brown
from random import randint
import csv
import sys
import numpy as np

def readFromFile():
	matA = []
	matB = []
	reverseIndices = {}
	wordIndices = {}

	# read it
	with open('matA.csv', 'r') as csvfile:
   		reader = csv.reader(csvfile)
    		matA = [[float(e) for e in r] for r in reader]

	# read it
	with open('matB.csv', 'r') as csvfile:
   		reader = csv.reader(csvfile)
    		matB = [[float(e) for e in r] for r in reader]

    	# read it
    	rows = []
    	with open('reverse.txt', 'r') as file:
                rows = filter(None, file.read().split("\n"))
        
        for r in rows:
                key = ""
                value = ""
                key, value = r.split("=+=+=")
                reverseIndices[int(key)] = value

        # read it
    	rows = []
    	with open('word.txt', 'r') as file:
                rows = filter(None, file.read().split("\n"))
        for r in rows:
                key = ""
                value = ""
                key, value = r.split("=+=+=")
                wordIndices[key] = int(value)
                

	return (matA, matB, reverseIndices, wordIndices)

def viterbi(observation, A, B, posDict, wordDict):
	tags = ""
	obsv = observation
	
	viterbi = [[0 for t in range(0, len(obsv))] for s in range(0, len(A) + 2)]
	backpointer =[[0 for t in range(0, len(obsv))] for s in range(0, len(A) + 2)]
	
	for s in range(0, len(A)):
		viterbi[s][0] = A[0][s] * B[s][ wordDict[obsv[0]] ]
		
	for t in range(1, len(obsv)):
                if obsv[t] == '':
                        continue
		for s in range(0, len(A)):
			viterbi[s][t] = max([ 
				viterbi[sp][t-1] * A[sp][s] * B[s][wordDict[obsv[t]]] for sp in range(0, len(A)) ])
				
			backpointer[s][t] = np.argmax([ 
				viterbi[sp][t-1] * A[sp][s] for sp in range(0, len(A)) ])

	z = [0 for t in range(0, len(obsv))]
	x = ["" for t in range(0, len(obsv))]
	
	z[len(obsv)-1] = np.argmax([ viterbi[sp][len(obsv)-1] for sp in range(0, len(A)) ])
	x[len(obsv)-1] = posDict[z[len(obsv)-1]]
	
	for i in range(len(obsv)-2,0, -1):
		z[i-1] = backpointer[z[i]][i]
		x[i-1] = posDict[z[i-1]]
	
	tags = " ".join(x)
	return tags

def main(argv):
	# Make matrix A and B
	matA = []
	matB = []
	reverseIndices = {}
	wordIndices = {}

	matA, matB, reverseIndices, wordIndices = readFromFile()

	observation = argv

	tags = viterbi(observation, matA, matB, reverseIndices, wordIndices)
	print tags

if __name__ == '__main__':
    main(sys.argv[1:])
