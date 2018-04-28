import numpy as np

class NNData():
	def __init__(self, rows, columns):
		self.rows = rows
		self.columns = columns
		self.data = np.zeros((rows, columns));
		
	def rand(self):
		self.data = np.random.rand(self.rows, self.columns)


	def mPrint(self):
		print self.data