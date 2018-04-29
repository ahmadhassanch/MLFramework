import numpy as np
from NNetwork import NNetwork
from NNData import NNData
from NNInput import NNInput
from NNInnerProduct import NNInnerProduct
from NNRelu import NNRelu
from NNSigmoid import NNSigmoid
import scipy.io as sio


def genData():
	refX = (np.random.rand(2,m)*10).round();
	refY = (np.random.rand(1,m)*1).round();
	refDataX.data = refX;
	refDataY.data = refY;
	sio.savemat("refData", {'refX':refX, 'refY':refY})


def main():
	m = 3;

	refDataX = NNData(2, m);
	refDataY = NNData(1, m);
	
	genData(refDataX, refDataY)	
		
	refDataY.mPrint()
	exit()
	
	net    = NNetwork();
	layer0 = NNInput(net, 'Input0', 2);
	
	layer1 = NNInnerProduct(net, 'InnerProduct1', 4);
	layer2 = NNRelu(net, 'NNRelu1', 4);
	
	layer3 = NNInnerProduct(net, 'InnerProduct2', 1);
	layer4 = NNSigmoid(net, 'NNSigmoid1', 1);
	
	net.initWeights();

	net.forward(refDataX);

	net.computeLoss(refDataY)

	net.backprop(refDataY);

	#layer0.outData.mPrint();
	#layer1.outData.mPrint();


main();


