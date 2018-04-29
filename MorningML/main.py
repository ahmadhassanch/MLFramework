import numpy as np
from NNetwork import NNetwork
from NNData import NNData
from NNInput import NNInput
from NNInnerProduct import NNInnerProduct
from NNRelu import NNRelu
from NNSigmoid import NNSigmoid
import scipy.io as sio


def genAndSaveData(numIn, m):
	refX = (np.random.rand(numIn,m)*10).round();
	refY = (np.random.rand(1,m)*1).round();
	sio.savemat("refData", {'refX':refX, 'refY':refY})
	return refX, refY

def loadData():
	mDict = sio.loadmat("refData")
	return mDict['refX'], mDict['refY']

def main():
	m = 3;
	numInputs = 2;

	#refX, refY = genAndSaveData(numInputs, m)	
	refX, refY = loadData()	

	refDataX = NNData(numInputs, m);
	refDataY = NNData(1, m);
	refDataX.data = refX;
	refDataY.data = refY;

	refDataX.mPrint()
	refDataY.mPrint()

	net    = NNetwork();
	layer0 = NNInput(net, 'Input0', 2);
	
	layer1 = NNInnerProduct(net, 'InnerProduct1', 4);
	layer2 = NNRelu(net, 'NNRelu1', 4);
	
	layer3 = NNInnerProduct(net, 'InnerProduct2', 1);
	layer4 = NNSigmoid(net, 'NNSigmoid1', 1);
	
	net.initWeights();
	net.forward(refDataX);
	print '==============================forward done=============================='
	net.computeLoss(refDataY)
	net.backprop(refDataY);

	#layer0.outData.mPrint();
	#layer1.outData.mPrint();


main();


