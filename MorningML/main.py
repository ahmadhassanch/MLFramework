import numpy as np
from NNetwork import NNetwork
from NNData import NNData
from NNInnerProduct import NNInnerProduct
from NNRelu import NNRelu
from NNSigmoid import NNSigmoid

def main():
	print 'test'
	inputDim = 2;
	outputDim = 1;
	m = 5;

	refX = np.zeros((inputDim, m));
	refY = np.zeros((outputDim, m));

	net = NNetwork();

	layer = NNInnerProduct('InnerProduct1', 4, inputDim);
	d = net.addLayer(layer);

	'''
	layer = NNRelu('NNRelu1', d, d);
	d = net.addLayer(layer);

	layer = NNInnerProduct('InnerProduct1', 1, d);
	d = net.addLayer(layer);

	layer = NNSigmoid('NNSigmoid1', d, d);
	net.addLayer(layer);
	'''
	#net.createOutputData(refX);
	net.forward(refX);

main();


