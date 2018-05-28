import numpy as np
import matplotlib.pyplot as plt
from NNetwork import NNetwork
from NNData import NNData
from NNLayer import NNInput
from NNInnerProduct import NNInnerProduct
from NNActivation import NNRelu
from NNActivation import NNSigmoid
import scipy.io as sio
from ioUtils import * 

import sys
import select

def heardEnter(alpha):
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
			print "Alpha = ", alpha, " Enter new Alpha";
			input2 = sys.stdin.readline()
			scale = raw_input("Enter NOw:");
			scale = float(scale)
			alpha = alpha * scale;
			print "heardEnter", input2, scale, alpha
			
    return alpha

def trainNetwork(net, nIterations, alpha, train_set_x, train_set_y):
	numInputs = train_set_x.shape[0];
	m = train_set_x.shape[1];
	refDataX = NNData(numInputs, m);
	refDataY = NNData(1, m);
	refDataX.data = train_set_x;
	refDataY.data = train_set_y;

	net.initWeights();

	net.gradientCheck(refDataX, refDataY)

	JArr = [];
	prevJ = 1e6;
	for i in range(nIterations):
		net.forward(refDataX);
		y, yHat, loss, J = net.computeLoss(refDataY)
		alpha = heardEnter(alpha)
		if(i%100 == 0):
			print '==============================forward====== ', i, J
			#print yHat
			net.debugInfo()
		#if(i==1200):
		# 	alpha = alpha/4
		#if(i>300):
		JArr.append(J);

		net.backprop(y, yHat);
		net.gradientDescent(alpha);
		#exit()
			#if(prevJ - J)/prevJ *100 < 5:
		 	#	alpha = alpha/2
			#	print '>>> changing alpa ', i
			#prevJ = J


	plt.plot(JArr)
	plt.show()
	net.saveWeights();


def testNetwork(net, test_set_x, test_set_y):
	numInputs = test_set_x.shape[0];
	m = test_set_x.shape[1];
	print "M = ", m
	nnTestX = NNData(numInputs, m);
	nnTestY = NNData(1, m);
	nnTestX.data = test_set_x;
	nnTestY.data = test_set_y;
	
	net.loadWeights();

	y = test_set_y;
	yHat = net.forward(nnTestX);
	
	yHat[yHat>0.5] = 1.0;
	yHat[yHat<=0.5] = 0.0;
	err = np.sum(np.abs(y-yHat))
	print "NumErrors", err, err/m*100, " Correct Pred Percent = ", 100 - err/m*100



def main1():
	alpha = 0.0075;
	#
	#alpha = 0.0001;
	#==============================forward======  0 0.69304973566
	#==============================forward======  100 0.646432095343

	np.random.seed(1)
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
	#[7,5,1]
	net    = NNetwork();
	layer0 = NNInput(net, 'Input0', 12288);
		
	layer1 = NNInnerProduct(net, 'InnerProduct1', 7);
	layer2 = NNRelu(net, 'NNRelu1', 7);

	layer3 = NNInnerProduct(net, 'InnerProductFinal', 1);
	layer4 = NNSigmoid(net, 'NNSigmoidFinal', 1);
	
	trainNetwork(net, 3500, alpha, train_set_x, train_set_y);
	testNetwork(net, train_set_x, train_set_y);
	testNetwork(net, test_set_x, test_set_y);

	
	#net.forward(refDataX);
	#y, yHat, loss = net.computeLoss(refDataY)
	#layer0.outData.mPrint();
	#layer1.outData.mPrint();


def main2():
	#Cost after iteration 0: 0.771749
	#Cost after iteration 100: 0.672053
	#==============================forward======  0 0.771749328424
	#==============================forward======  100 0.672053440082

	alpha = 0.0075;
	np.random.seed(1)
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
	#[7,5,1]
	net    = NNetwork();
	layer0 = NNInput(net, 'Input0', 12288);

	layer1 = NNInnerProduct(net, 'InnerProduct1', 20);
	layer2 = NNRelu(net, 'NNRelu1', 20);

	layer3 = NNInnerProduct(net, 'InnerProduct2', 7);
	layer4 = NNRelu(net, 'NNRelu2', 7);

	layer5 = NNInnerProduct(net, 'InnerProduct3', 5);
	layer6 = NNRelu(net, 'NNRelu3', 5);

	layer7 = NNInnerProduct(net, 'InnerProductFinal', 1);
	layer8 = NNSigmoid(net, 'NNSigmoidFinal', 1);
	
	trainNetwork(net, 2500, alpha, train_set_x, train_set_y);
	testNetwork(net, train_set_x, train_set_y);
	testNetwork(net, test_set_x, test_set_y);

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
main2();


