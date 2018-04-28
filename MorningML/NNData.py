import numpy as np

class NNData():
	def __init__(self, rows, columns, values = "ZEROS"):
		self.data = 'UNINITIALIZED ... ';
		if   values == 'ZEROS':
			self.data = np.zeros((rows, columns));
		elif values == 'ONES':
			self.data = np.ones((rows, columns));
		elif values == 'RANDOM':
			self.data = np.random.randn(rows, columns);
		else:
			print 'unknow NNData type'
		
	def mPrint(self):
		print self.data