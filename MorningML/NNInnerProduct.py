import numpy as np
from NNLayer import NNLayer
from NNData import NNData
import scipy.io as sio

class NNInnerProduct(NNLayer):
	def __init__(self, network, name, nlOut):
		NNLayer.__init__(self, network, name, nlOut)
		self.type = "InnerProduct"

	def initWeights(self):
		print self.name, "Initializing Weights, biases"
		self.W = NNData(self.nlOut, self.nlIn);
		self.B = NNData(self.nlOut, 1);
		self.W.rand();
		#	self.B.rand();
		#self.W.ones();
		#self.B.data[0,0]=10;
		#print self.W.data.shape;
		self.dW = NNData(self.nlOut, self.nlIn);
		self.dB = NNData(self.nlOut, 1);

		self.pW = NNData(self.nlOut, self.nlIn);
		self.pB = NNData(self.nlOut, 1);
		
		#self.W.mPrint();
		#self.B.mPrint();
		#exit()

	def saveWeights(self):
		sio.savemat(self.name+".mat", {'W':self.W.data, 'B':self.B.data})
		#print self.name, ": No Weights, biases"
		

	def loadWeights(self):
		self.initWeights()

		mDict = sio.loadmat(self.name+".mat")
		#exit()
		self.W.data = mDict['W']
		self.B.data = mDict['B']
		#self.W.mPrint();
		#exit()

	def forward(self, X):		
		self.outData = self.W * X + self.B;
		#self.W.mPrint()

		self.X = X.data;
		return self.outData

	def backprop(self, dGlobal):
		m = dGlobal.shape[1];
		W = self.W.data;
		dGlobalNew = np.matmul(W.T, dGlobal);
		self.dW.data = (1.0/m)*np.matmul(dGlobal, self.X.T)
		#print self.name
		#print self.dW.data.shape
		#print self.dW.data;

		self.dB.data = (1.0/m)*np.sum(dGlobal, axis=1, keepdims=True);
		#print type(self.dB.data)
		#exit()
		return dGlobalNew

	def restorePivot(self):
		print "================restoring Pivot===========>>>>>>"
		#self.W.mPrint()
		self.W.data = np.copy(self.pW.data)
		self.B.data = np.copy(self.pB.data)
		#self.W.mPrint()
		#exit()

	def gradientDescent(self, alpha):
		W = self.W.data;
		B = self.B.data;
		#print self.dW.data;
		#print type(self.dW.data)
		
		self.pW.data = np.copy(self.W.data)
		self.pB.data = np.copy(self.B.data)

		#exit()
		#W = W - np.clip(np.sign(self.dW.data),-alpha,alpha);
		#B = B - np.clip(np.sign(self.dB.data), -alpha, alpha);
	
		W = W - alpha * self.dW.data;
		B = B - alpha * self.dB.data;
		self.W.data = W;
		self.B.data = B;


	def debugInfo(self):
		print self.name
		#self.W.mPrintSTD()
		#if(self.name == "InnerProductFinal"):
		#	self.outData.mPrint();

		self.W.mPrintSTD()
		#self.B.mPrintSTD()
		self.outData.mPrintSTD();
