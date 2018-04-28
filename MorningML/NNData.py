import numpy as np

class NNData():
	def __init__(self, rows, columns):
		self.rows = rows
		self.columns = columns
		self.data = np.zeros((rows, columns));
		
	def rand(self):
		self.data = np.random.rand(self.rows, self.columns)

	def ones(self):
		self.data = np.ones((self.rows, self.columns))

	def mPrint(self):
		print 'test12'
		print self.data
		print 'test22'
		#exit()

