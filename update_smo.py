# *-* coding:utf-8 *-* 
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat
   
def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def calcEk(oS, k):
	fXk = float(multiply(oS.alphas, oS.labelMat).T*\
		(oS.X * oS.X[k, :].T)) + oS.b
	Ek = fXk - float( oS.labelMat[k] )
	return Ek

def selectJ(i, oS, Ei):
	maxK = -1; maxDeltaE = 0; Ej = 0
	oS.eCache[i] = [1 ,Ei]
	validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
	if(len(validEcacheList) > 1):
		for k in validEcacheList:
			if k == i:
				continue
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)
			if(deltaE > maxDeltaE):
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej
	else:
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	return j, Ej

def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1, Ek]

def selectJrand(i, m):
	j = i
	while(j == i):
		j = int( random.uniform(0, m) )
	return j

def innerL(i ,oS):
	Ei = calcEk(oS, i)
	if ( (oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C) ) or \
	( (oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0) ):
		j, Ej = selectJ(i ,oS, Ei)
		alphasIold = oS.alphas[i].copy(); alphasJold = oS.alphas[j].copy()
		if( oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j]+oS.alphas[i]-oS.C)
			H = min(oS.C, oS.alphas[j]+oS.alphas[i])
		if L == H:
			print  'L == H'
			return 0
		eta = 2.0*oS.X[i, :]*oS.X[j, :].T-oS.X[i,:]*oS.X[i, :].T- \
		oS.X[j, :]*oS.X[j, :].T
		if eta >= 0:
			print 'eta>=0'
			return 0
		oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		updateEk(oS, j)
		if( abs(oS.alphas[j] - alphasJold) < 0.00001 ):
			print "j not moving enough"
			return 0
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphasJold-oS.alphas[j])
		updateEk(oS, i)
		b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*\
		oS.X[i, :]*oS.X[i, :].T-oS.labelMat[j]*\
		(oS.alphas[j]-alphasJold)*oS.X[i, :]*oS.X[j, :].T
		b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphasIold)*\
		oS.X[i, :]*oS.X[j, :].T-oS.labelMat[j]*\
		(oS.alphas[j]-alphasJold)*oS.X[j, :]*oS.X[j, :].T

		if (0 < oS.alphas[i] ) and (oS.alphas[i] < oS.C):
			oS.b = b1
		elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
			oS.b = b2
		else:
			oS.b = (b1+b2)/2.0
		return 1
	else:
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0) ):
	oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)
	iters = 0
	entireSet = True; alphaPairsChanged = 0
	while (iters < maxIter) and ((alphaPairsChanged>0) or (entireSet)):
		alphaPairsChanged = 0
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)
			iters += 1
		else:
			nonBoundIs = nonzero( (oS.alphas.A > 0) * (oS.alphas.A < C) )[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i, oS)
			iters += 1
		if entireSet:
			entireSet = False
		elif(alphaPairsChanged == 0):
			entireSet = True
	return oS.b, oS.alphas

def plotSupportVector(supportVector, dataArr):
	dataMat = array(dataArr)
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(100):
		if supportVector[i] == 1:
			xcord1.append(dataMat[i, 0]); ycord1.append(dataMat[i, 1])
		else:
			xcord2.append(dataMat[i, 0]); ycord2.append(dataMat[i, 1])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	plt.show()

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag

dataArr, labelArr = loadDataSet('testSet.txt')
b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)



supportVector = zeros(100)
#打印出支持向量
for i in range(100):
	if alphas[i] > 0.0:
		supportVector[i] = 1
		print dataArr[i], labelArr[i]

plotSupportVector(supportVector, dataArr)