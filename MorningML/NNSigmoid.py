
from NNLayer import NNLayer

class NNSigmoid(NNLayer):
	def __init__(self, name, nlOut, nlIn):
		NNLayer.__init__(self, name, nlOut, nlIn)

	def forward(self):
		print 'Forward Prop Layer: ', self.name;
