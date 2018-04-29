from NNLayer import NNLayer
from NNActivation import NNActivation

class NNRelu(NNActivation):
			
	def forward(self, inData):
		self.outData = inData.relu();
		return self.outData
