# ANNCodeGen-Project
 An adaptive ANN (CNN or FNN) code generator using python
 
 - Generates a standard Neural Network Class in Python
     -- Right Now done only in nn
 - Format for text file having description is 
     -- Start file with 'StartNetworkDesc' and end with 'EndNetworkDesc'
     -- Anything outside this is ignored
     -- Then within that just give the layers one by one with parameters (same order as in python nn documentation)
     -- The generator builds a nn class file for you!
     
 - Eg. For text file, 
     -- StartNetworkDesc
        Linear <-><8>
        Sigmoid <>
        Linear <8><16>
        Sigmoid <>
        Linear <16><->
        Softmax <>
        EndNetworkDesc

   Output class file is, 
     -- 
 
class BasicNN(nn.Module):
	def __init__(self , n_inputs, n_outputs):
		import torch
		import torch.nn as nn
		from collections import OrderedDict
		super().__init__()
		torch.manual_seed(0)
		self.net = nn.Sequential(
		nn.Linear(n_inputs, 8), 
		nn.Sigmoid(), 
		nn.Linear(8, 16), 
		nn.Sigmoid(), 
		nn.Linear(16, n_outputs), 
		nn.Softmax()  
		)
	def forward(self, X):
		return self.net(X)
	def fit(self, x, y, opt, loss_fn, epochs, display_loss=True):
		from torch import optim
		import matplotlib.pyplot as plt
		import matplotlib.colors
		loss_arr = []
		for epoch in range(epochs):
			loss = self.loss_fn(self.forward(x), y)
			loss_temp = loss.item()
			loss_arr.append(loss_temp)
			loss.backward()
			opt.step()
			opt.zero_grad()
		if display_loss:
			plt.plot(loss_arr)
			plt.xlabel('Epochs')
			plt.ylabel('CE')
			plt.show()
		return loss.item()
	def predict(self, X):
		import numpy as np
		Y_pred = self.net(X)
		Y_pred = Y_pred.detach().numpy()
		return np.array(Y_pred).squeeze()
