import torch
import math

def grad(x,y,create_graph=False,retain_graph=False):
	"""
	compute gradients of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	:return: dx/dy
	"""
	return torch.autograd.grad(torch.sum(x),y,create_graph=create_graph,retain_graph=retain_graph)[0]

def rot(x,y,create_graph=False,retain_graph=False):
	"""
	compute curl of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	"""
	result = grad(x,y,create_graph,retain_graph)[:,[1,0]]
	result[:,1] = -result[:,1]
	return result

def laplace(x, y, create_graph = False,retain_graph=False):
	"""
	compute laplacian of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	"""
	return div(grad(x,y,create_graph=True,retain_graph=True), y,create_graph=create_graph,retain_graph=retain_graph)

def div(x, y,create_graph=False,retain_graph=False):
	"""
	compute divergence of x = f(y)
	:x: function outputs
	:y: parameters (expected shape: [batchsize x 3] - 3 for x,y,t)
	"""
	div = grad(x[:,0],y,create_graph=create_graph,retain_graph=True)[:,0:1]+grad(x[:,1],y,create_graph=create_graph,retain_graph=retain_graph)[:,1:2]
	return div

def vector2HSV(vector,plot_sqrt=False):
	"""
	transform vector field into hsv color wheel
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = torch.ones(values.shape)
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	#values = norm*torch.log(values+1)
	values = values/torch.max(values)
	if plot_sqrt:
		values = torch.sqrt(values)
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).numpy()
