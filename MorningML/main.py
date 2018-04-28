import numpy as np
from NNetwork import NNetwork
from NNData import NNData
from NNInput import NNInput
from NNInnerProduct import NNInnerProduct
from NNRelu import NNRelu
from NNSigmoid import NNSigmoid

def main():
	print '\n'
	inputDim = 2;
	outputDim = 1;
	m = 3;

	refDataX = NNData(inputDim, m);
	refDataY = NNData(outputDim, m);

	refDataX.data

	refDataX.mPrint()
#	exit()

	net = NNetwork();

	layer = NNInput(net, 'Input0', inputDim, -1);
	d = net.addLayer(layer);

	layer = NNInnerProduct(net, 'InnerProduct1', 3, inputDim);
	d = net.addLayer(layer);

	layer = NNRelu(net, 'NNRelu1', d, d);
	d = net.addLayer(layer);

	layer = NNInnerProduct(net, 'InnerProduct2', 1, d);
	d = net.addLayer(layer);

	layer = NNSigmoid(net, 'NNSigmoid1', d, d);
	net.addLayer(layer);
	
	net.initWeights();

	net.forward(refDataX);

main();


