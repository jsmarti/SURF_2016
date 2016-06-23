import numpy as np

class Obj:
	def __init__(self):
		self.data = np.array([])
		self.y = np.array([])
		
	def add(self, value):
		if not isinstance(value,int):
			raise Exception('Invalid value')
		else:
			self.data = np.append(self.data,value)
			self.update_y()
			
	def update_y(self):
		self.y = np.sin(2*np.pi*100*self.data)
	
	def get_data(self):
		return self.data, self.y
