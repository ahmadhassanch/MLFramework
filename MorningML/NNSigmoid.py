
from NNLayer import NNLayer
from NNActivation import NNActivation

class NNSigmoid(NNActivation):

	def forward(self, inData):
		self.outData = inData.sigmoid();
		return self.outData

