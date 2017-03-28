'''
  Myanna Harris
  Jasmine Jans
  11-18-16
  asgn6_train.py

  Probabilistic part-of-speech tagger using Brown Corpus
  Makes A and B matrices

  To run:
  python asgn6_train.py 
'''

import nltk
from nltk.corpus import brown
import random
from random import randint
import csv
import numpy as np

def getTestAndTrainingSets(corpus):
	test = []
	train = []
      	numTest = int(len(corpus) * .1)
        testNums = []

        testNums = random.sample(range(0,len(corpus)-1), numTest)
	for i in range(0, len(corpus)):
                if i in testNums:
			test.append(corpus[i])
		else:
			train.append(corpus[i])
	return (test, train)



def getTagFreqs(train):
        tagFreq = {}
        for tup in train:
                 if not tagFreq.has_key(tup[1]):
                        tagFreq[tup[1]] = 1
                 else:
                        tagFreq[tup[1]] += 1
        return tagFreq

def getTagIndices(train):
        posIdxDict = {}
        currDictIdx = 0
        
        for tup in train:
                if not posIdxDict.has_key(tup[1]):
                        posIdxDict[tup[1]] = currDictIdx
                        currDictIdx += 1
                        
        return posIdxDict

def makeReverseDict(posIdxDict):
        newPosIdxDict = {}
        for key, value in posIdxDict.items():
                newPosIdxDict[value] = key

        return newPosIdxDict
                
def makeMatA(train):
        tagFreq = getTagFreqs(train)
        posIdxDict = getTagIndices(train)

        #previous POS tags across the row, current POS tag down the column (pos, prevpos)
        matA = [[0 for i in range(0, len(posIdxDict.keys()))] for k in range(0, len(posIdxDict.keys()))]
                 
        for i in range(1, len(train)-1):
            prevPos = train[i-1][1]
            pos = train[i][1]
            prevPosIdx = posIdxDict[prevPos]
            posIdx = posIdxDict[pos]

            #add freq of the pos transition prob to correlating matrices position
            matA[posIdx][prevPosIdx] += 1

        for i in range(0,len(matA)):
                for k in range(0, len(matA[i])):
                    for key, value in posIdxDict.items():
                        if value == k:
                            posTrans = key
                            break
                    matA[i][k] /= float(tagFreq[posTrans])

        return matA

def getWordIndices(train):
        wordIdxDict = {}
        currDictIdx = 0
        
        for tup in train:
                if not wordIdxDict.has_key(tup[0]):
                        wordIdxDict[tup[0]] = currDictIdx
                        currDictIdx += 1
                        
        return wordIdxDict

def makeMatB(train):
        tagFreq = getTagFreqs(train)
        posIdxDict = getTagIndices(train)
        wordIdxDict = getWordIndices(train)
        
        #words across the row, POS tags down the column (pos, word)
        matB = [[0 for i in range(0, len(wordIdxDict.keys()))] for k in range(0, len(posIdxDict.keys()))]
                 
        for i in range(0, len(train)):
            word = train[i][0]
            pos = train[i][1]
            posIdx = posIdxDict[pos]
            wordIdx = wordIdxDict[word]

            #add freq of the pos transition prob to correlating matrices position
            matB[posIdx][wordIdx] += 1
        
        posTrans = ""
        for i in range(0,len(matB)):
                #get the POS tag
                for key, value in posIdxDict.items():
                        if value == i:
                            posTrans = key
                            break
                #go through each word with said POS tag
                for k in range(0, len(matB[i])):
                    matB[i][k] /= float(tagFreq[posTrans])
        
        return matB

def writeToFile(matA, matB, reverseIndices, wordIndices):
	# write it
	with open('matA.csv', 'w') as csvfile:
    		writer = csv.writer(csvfile)
    		[writer.writerow(r) for r in matA]

	with open('matB.csv', 'w') as csvfile:
    		writer = csv.writer(csvfile)
    		[writer.writerow(r) for r in matB]

    	with open('reverse.txt', 'w') as file:
                out = []
                for key, value in reverseIndices.items():
                        out.append(str(key)+"=+=+="+str(value))
                file.write("\n".join(out))

        with open('word.txt', 'w') as file:
                out = []
                for key, value in wordIndices.items():
                        out.append(str(key)+"=+=+="+str(value))
                file.write("\n".join(out))

def viterbi(observation, A, B, posDict, wordDict):
	tags = ""
	obsv = observation.split(' ')
	
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

#def extractSentences(k, test):
        

def main():
	brown_tag = nltk.corpus.brown.tagged_words()
	if not len(brown_tag) == 1161192:
		# Check corpus size
		print "Error loading corpus - wrong size"
	elif  not [brown_tag[i] for i in range(2)] ==  [(u'The', u'AT'), (u'Fulton', u'NP-TL')]:
		# Check corpus content
		print "Error loading corpus - wrong content"
	else:

		#the program runs slow (we have yet to experience it complete) when run on the whole corpus
		#we use this to shorten the corpus (takes about 5-10 minutes at 100 lines)
		brown_tag = [brown_tag[i] for i in range(0, int(.1*len(brown_tag)))]
		brown_tag_new = [(
                        tup[0].encode('ascii'),tup[1].encode('ascii')) for tup in brown_tag if tup[1] is not u'']

                obs = ""
                tagsObs = ""
                for i in range(30):
                        print brown_tag_new[i]
                        if brown_tag_new[i][0] == '.':
                                break
                        else:
                                obs += brown_tag_new[i][0] + " "
                                tagsObs += brown_tag_new[i][1] + " "
                                #print obs
                                #print tagsObs
                                #print "=============="
                print obs
                print tagsObs
                print "======================================================================================================"
		
		# Check ascii content
		if [brown_tag_new[i] for i in range(2)] ==  [('The', 'AT'), ('Fulton','NP-TL')]:
			# Get random test and training sets
			brown_test, brown_train = getTestAndTrainingSets(brown_tag_new)

                        wordIndices = getWordIndices(brown_train)
			posIndices = getTagIndices(brown_train)
			reverseIndices = makeReverseDict(posIndices)

			# Make matrix A and B
			matA = makeMatA(brown_train)
                        
			matB = makeMatB(brown_train)
			#print matB

			print viterbi(obs, matA, matB, reverseIndices, wordIndices)

			writeToFile(matA, matB, reverseIndices, wordIndices)

if __name__ == '__main__':
    main()
