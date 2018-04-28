import numpy as np
from NNetwork import NNetwork
from NNData import NNData
from NNInnerProduct import NNInnerProduct
from NNRelu import NNRelu
from NNSigmoid import NNSigmoid

def main():
	print '\n'
	inputDim = 2;
	outputDim = 1;
	m = 5;

	refDataX = NNData(inputDim, m);
	refDataY = NNData(outputDim, m);

	net = NNetwork();

	layer = NNInnerProduct('InnerProduct1', 4, inputDim);
	d = net.addLayer(layer);

	layer = NNRelu('NNRelu1', d, d);
	d = net.addLayer(layer);

	layer = NNInnerProduct('InnerProduct1', 1, d);
	d = net.addLayer(layer);

	layer = NNSigmoid('NNSigmoid1', d, d);
	net.addLayer(layer);
	

	net.initWeights();

	net.forward(refDataX);

main();


