import numpy as np
import matplotlib.pyplot as plt
import h5py
from NNetwork import NNetwork
from NNData import NNData
from NNLayer import NNInput
from NNInnerProduct import NNInnerProduct
from NNActivation import NNRelu
from NNActivation import NNSigmoid
import scipy.io as sio


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def genAndSaveData(numIn, m):
	refX = (np.random.rand(numIn,m)*10).round();
	refY = (np.random.rand(1,m)*1).round();
	sio.savemat("refData", {'refX':refX, 'refY':refY})
	return refX, refY

def loadData():
	mDict = sio.loadmat("refData")
	return mDict['refX'], mDict['refY']

def trainNetwork(net, train_set_x, train_set_y):
	numInputs = train_set_x.shape[0];
	m = train_set_x.shape[1];
	refDataX = NNData(numInputs, m);
	refDataY = NNData(1, m);
	refDataX.data = train_set_x;
	refDataY.data = train_set_y;
	'''
	J =================================================== J =  0.627797307324
	J =================================================== J =  1.01246567508
	================restoring Pivot===========>>>>>>
	================restoring Pivot===========>>>>>>
	J =================================================== J =  0.649202240636

	'''
	net.initWeights();
	JArr = [];
	alpha = 0.0075;


	for i in range(3000):
		print '==============================forward====== ', i
		net.forward(refDataX);
		y, yHat, loss, J = net.computeLoss(refDataY)
		
		JArr.append(J);
		net.backprop(y, yHat);
		net.gradientDescent(alpha);


	plt.plot(JArr)
	plt.show()
	#net.saveWeights();

def testNetwork(net, test_set_x, test_set_y):
	numInputs = test_set_x.shape[0];
	m = test_set_x.shape[1];
	print "M = ", m
	nnTestX = NNData(numInputs, m);
	nnTestY = NNData(1, m);
	nnTestX.data = test_set_x;
	nnTestY.data = test_set_y;
	
	#net.loadWeights();

	y = test_set_y;
	yHat = net.forward(nnTestX);
	
	#y, yHat, loss = net.computeLoss(refDataY)

	yHat[yHat>0.5] = 1.0;
	yHat[yHat<=0.5] = 0.0;
	err = np.sum(np.abs(y-yHat))
	print "NumErrors", err, err/m*100



def main():

	np.random.seed(2)
	# Example of a picture
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
	train_set_x = train_set_x_flatten / 255.
	test_set_x = test_set_x_flatten / 255.

	index = 25
	#plt.imshow(train_set_x_orig[index])
	print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")
	#plt.show()
	#pause


	#refX, refY = genAndSaveData(numInputs, m)	
	#refX, refY = loadData()	

	net    = NNetwork();
	layer0 = NNInput(net, 'Input0', 12288);
	
	layer1 = NNInnerProduct(net, 'InnerProduct1', 7);
	layer2 = NNRelu(net, 'NNRelu1', 7);
	
	layer3 = NNInnerProduct(net, 'InnerProduct2', 1);
	layer4 = NNSigmoid(net, 'NNSigmoid1', 1);
	
	trainNetwork(net, train_set_x, train_set_y);
	testNetwork(net, train_set_x, train_set_y);
	testNetwork(net, test_set_x, test_set_y);

	
	#net.forward(refDataX);
	#y, yHat, loss = net.computeLoss(refDataY)
	#layer0.outData.mPrint();
	#layer1.outData.mPrint();


main();


