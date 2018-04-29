import numpy as np
from NNetwork import NNetwork
from NNData import NNData
from NNInput import NNInput
from NNInnerProduct import NNInnerProduct
from NNRelu import NNRelu
from NNSigmoid import NNSigmoid

def main():
	print '\n'

	m = 3;

	refDataX = NNData(2, m);
	refDataY = NNData(1, m);
	
	v = (np.random.rand(2,m)*10).round();
	refDataX.data = v;
	refDataX.mPrint()
	
	net    = NNetwork();
	layer0 = NNInput(net, 'Input0', 2);
	
	layer1 = NNInnerProduct(net, 'InnerProduct1', 4);
	layer2 = NNRelu(net, 'NNRelu1', 4);
	
	layer3 = NNInnerProduct(net, 'InnerProduct2', 1);
	layer4 = NNSigmoid(net, 'NNSigmoid1', 1);
	
	net.initWeights();

	net.forward(refDataX);
	#layer0.outData.mPrint();
	#layer1.outData.mPrint();


main();


