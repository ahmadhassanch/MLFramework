import numpy as np

class NNData():
	def __init__(self, rows, columns):
		self.rows = rows
		self.columns = columns
		self.data = np.zeros((rows, columns));
		
	def rand(self):
		self.data = np.random.rand(self.rows, self.columns) - 0.5;

	def ones(self):
		self.data = np.ones((self.rows, self.columns))

	def __mul__(self, B):
		c = np.matmul(self.data, B.data);
		tempObj = NNData(c.shape[0], c.shape[1]);
		tempObj.data = c;
		return tempObj;

	def __add__(self, B):
		c = np.add(self.data, B.data);
		tempObj = NNData(c.shape[0], c.shape[1]);
		tempObj.data = c;
		return tempObj;

	def sigmoid(self):
		ez = np.exp(-self.data);
		tempObj = NNData(self.data.shape[0], self.data.shape[1]);
		tempObj.data = 1.0 / (1.0 + ez);
		return tempObj;

	def relu(self):
		a = self.data;
		a[a < 0] = 0;
		tempObj = NNData(self.data.shape[0], self.data.shape[1]);
		tempObj.data = a;
		return tempObj;

	def mPrint(self):
		#print 'test12'
		print "Data Dimensions:", self.data.shape
		print self.data
		#print 'test22'
		#exit()

