import torch
from torch import nn
import torch.nn.functional as F
from get_param import toCuda,toCpu
from operators import rot,grad,div
from unet_parts import *
import numpy as np
import os,pickle

def get_Net(params):
	if params.net == "Fluid_model":
		net = fluid_model(orders_v=[params.orders_v,params.orders_v],orders_p=[params.orders_p,params.orders_p],hidden_size=params.hidden_size,input_size=3)
	elif params.net == "Wave_model":
		params.orders_p = params.orders_v = params.orders_z
		net = wave_model(orders_v=[params.orders_v,params.orders_v],orders_p=[params.orders_p,params.orders_p],hidden_size=params.hidden_size,input_size=2,residuals=True)
	return net

class fluid_model(nn.Module):
	# inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, orders_v,orders_p,hidden_size=64,interpolation_size=5, bilinear=True,input_size=3,residuals=False):
		"""
		:orders_v: order of spline for velocity potential (should be at least 2)
		:orders_p: order of spline for pressure field
		:hidden_size: hidden size of neural net
		:interpolation_size: size of first interpolation layer for v_cond and v_mask
		"""
		super(fluid_model, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear
		self.input_size = input_size
		
		self.orders_v = orders_v
		self.orders_p = orders_p
		self.v_size = np.prod([i+1 for i in orders_v])
		self.p_size = np.prod([i+1 for i in orders_p])
		self.hidden_state_size = self.v_size + self.p_size
		self.residuals = residuals
		
		self.interpol = nn.Conv2d(input_size,interpolation_size,kernel_size=2) # interpolate v_cond (2) and v_mask (1) from 4 surrounding fields
		self.inc = DoubleConv(self.hidden_state_size+interpolation_size, hidden_size) # input: hidden_state + interpolation of v_cond and v_mask
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, self.hidden_state_size)
		self.output_scaler = toCuda(torch.ones(1,self.v_size+self.p_size,1,1)*2)
		self.output_scaler[:,0:1,:,:] = 400
		self.output_scaler[:,(self.v_size):(self.v_size+1),:,:] = 400
		
	
	def forward(self,hidden_state,v_cond,v_mask):
		"""
		:hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		:v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
		:v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
		:return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		"""
		x = torch.cat([v_cond,v_mask],dim=1)
		
		x = self.interpol(x)
		
		x = torch.cat([hidden_state,x],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		out = x
		
		# residual connections
		out[:,:,:,:] = self.output_scaler*torch.tanh((out[:,:,:,:]+hidden_state[:,:,:,:])/self.output_scaler)
		
		#substract mean of a_z and p
		out[:,0:1,:,:] = out[:,0:1,:,:]-torch.mean(out[:,0:1,:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		out[:,(self.v_size):(self.v_size+1),:,:] = out[:,(self.v_size):(self.v_size+1),:,:]-torch.mean(out[:,(self.v_size):(self.v_size+1),:,:],dim=(2,3)).unsqueeze(2).unsqueeze(3)
		return out

class wave_model(nn.Module):
	
	def __init__(self, orders_v,orders_p,hidden_size=64,interpolation_size=5, bilinear=True,input_size=3,residuals=False):
		"""
		:orders_v: order of spline for velocity potential (should be at least 2)
		:orders_p: order of spline for pressure field
		:hidden_size: hidden size of neural net
		:interpolation_size: size of first interpolation layer for v_cond and v_mask
		"""
		super(wave_model, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear
		self.input_size = input_size
		
		self.orders_v = orders_v
		self.orders_p = orders_p
		self.v_size = np.prod([i+1 for i in orders_v])
		self.p_size = np.prod([i+1 for i in orders_p])
		self.hidden_state_size = self.v_size + self.p_size
		self.residuals = residuals
		
		self.interpol = nn.Conv2d(input_size,interpolation_size,kernel_size=2) # interpolate v_cond (2) and v_mask (1) from 4 surrounding fields
		self.conv1 = nn.Conv2d(self.hidden_state_size+interpolation_size, self.hidden_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
		self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
		self.conv3 = nn.Conv2d(self.hidden_size, self.hidden_state_size,kernel_size=3,padding=1) # input: hidden_state + interpolation of v_cond and v_mask
		
		if self.hidden_state_size == 18: # if orders_z = 2
			self.output_scaler_wave = toCuda(torch.Tensor([5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05, 5,0.5,0.05,0.5, 0.05,0.05,0.05,0.05,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
		elif self.hidden_state_size == 8: # if orders_z = 1
			self.output_scaler_wave = toCuda(torch.Tensor([5,0.5,0.5,0.05, 5,0.5,0.5,0.05]).unsqueeze(0).unsqueeze(2).unsqueeze(3))
		
	
	def forward(self,hidden_state,v_cond,v_mask):
		"""
		:hidden_state: old hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		:v_cond: velocity (dirichlet) conditions on boundaries (average value within cell): bs x 2 x w x h
		:v_mask: mask for boundary conditions (average value within cell): bs x 1 x w x h
		:return: new hidden state of size: bs x hidden_state_size x (w-1) x (h-1)
		"""
		x = torch.cat([v_cond,v_mask],dim=1)
		
		x = self.interpol(x)
		
		x = torch.cat([hidden_state,x],dim=1)
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		out = self.conv3(x)
		
		# residual connections
		out[:,:,:,:] = self.output_scaler_wave*torch.tanh((out[:,:,:,:]+hidden_state[:,:,:,:]/self.output_scaler_wave))
		
		return out

def sign(x):
	s = torch.sign(x)
	s[s==0]=1
	return s

def heaviside(x):
	return (torch.sign(x)+1)/2

# 1st order splines
def p1_1(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)

p1 = [p1_1] # list of 1st order basis splines

# 2nd order splines
def p2_1(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)**2*(1+2*offsets)

def p2_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(1-abs_offsets)**2*(abs_offsets)

# derivatives
def dp2_1(offsets):#first derivative (needs to be devided by dt)
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(6*abs_offsets**2-6*abs_offsets)

def dp2_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return 3*abs_offsets**2 - 4*abs_offsets + 1

def d2p2_1(offsets):#2nd derivative (needs to be devided by dt**2)
	abs_offsets = offsets*sign(offsets)
	return 12*abs_offsets - 6

def d2p2_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(6*abs_offsets-4)

p2 = [p2_1,p2_2] # list of 2nd order basis splines

# 3rd order splines
def p3_1(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)**3*(1+3*offsets+6*offsets**2)

def p3_2(offsets):
	abs_offsets = offsets*sign(offsets)
	return sign(offsets)*(1-abs_offsets)**3*(abs_offsets+3*abs_offsets**2)*2

def p3_3(offsets):
	offsets = offsets*sign(offsets)
	return (1-offsets)**3*(0.5*offsets**2)*16

p3 = [p3_1,p3_2,p3_3] # list of 3rd order basis splines

# 4th order splines
def p4_1(offsets):
	return (offsets-1)**4*(1+4*offsets+10*offsets**2+20*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(1-4*offsets+10*offsets**2-20*offsets**3)*heaviside(-offsets)

def p4_2(offsets):
	return ((offsets-1)**4*(1*offsets+4*offsets**2+10*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(1*offsets-4*offsets**2+10*offsets**3)*heaviside(-offsets))*4

def p4_3(offsets):
	return ((offsets-1)**4*(0.5*offsets**2+2*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(0.5*offsets**2-2*offsets**3)*heaviside(-offsets))*32

def p4_4(offsets):
	return ((offsets-1)**4*(1.0/6.0*offsets**3)*heaviside(offsets)+(-offsets-1)**4*(1.0/6.0*offsets**3)*heaviside(-offsets))*512

p4 = [p4_1,p4_2,p4_3,p4_4]

# 5th order splines
def p5_1(offsets):
	return ((offsets-1)**5*(-1-5*offsets-15*offsets**2-35*offsets**3-70*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-1+5*offsets-15*offsets**2+35*offsets**3-70*offsets**4)*heaviside(-offsets))

def p5_2(offsets):
	return ((offsets-1)**5*(-1*offsets-5*offsets**2-15*offsets**3-35*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-1*offsets+5*offsets**2-15*offsets**3+35*offsets**4)*heaviside(-offsets))*4

def p5_3(offsets):
	return ((offsets-1)**5*(-0.5*offsets**2-2.5*offsets**3-7.5*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-0.5*offsets**2+2.5*offsets**3-7.5*offsets**4)*heaviside(-offsets))*32

def p5_4(offsets):
	return ((offsets-1)**5*(-0.5/3.0*offsets**3-2.5/3.0*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-0.5/3.0*offsets**3+2.5/3.0*offsets**4)*heaviside(-offsets))*512

def p5_5(offsets):
	return ((offsets-1)**5*(-2.5/6.0*offsets**4)*heaviside(offsets)+(-offsets-1)**5*(-2.5/6.0*offsets**4)*heaviside(-offsets))*1024

p5 = [p5_1,p5_2,p5_3,p5_4,p5_5]

pi = [p1,p2,p3,p4,p5] # list of lists of basis splines for different orders

def p_multidim(offsets,orders,indices):
	"""
	multidimensional basis spline of specified orders and indices
	:offsets: offsets of size: bs x n_dims x ...
	:orders: orders of spline for each dimension (note: counting starts at 0 => 0 ~ 1st order, 1 ~ 2nd order, 2 ~ 3rd order)
	:indices: indices of spline for each dimension (note: counting starts at 0)
	"""
	return torch.prod(torch.cat([pi[orders[i]][indices[i]](offsets[:,i:(i+1)]).unsqueeze(0) for i in range(len(orders))]),dim=0)

# buffering interpolation kernels significantly speeds up computations
offset_summary = toCuda(torch.tensor([[[0,0],[1,0]],[[0,1],[1,1]]]).unsqueeze(0).permute(0,3,2,1))
kernel_buffer_velocity = {}
kernel_buffer_velocity_superres = {}
kernel_buffer_pressure = {}
kernel_buffer_pressure_superres = {}
kernel_buffer_wave = {}
kernel_buffer_wave_superres = {}

def save_buffers():
	os.makedirs('Logger/buffers',exist_ok=True)
	path = 'Logger/buffers/buffers.dic'
	with open(path,"wb") as f:
		pickle.dump({"vel":kernel_buffer_velocity,"vel_superres":kernel_buffer_velocity_superres,"pres":kernel_buffer_pressure,"pres_superres":kernel_buffer_pressure_superres,"wave":kernel_buffer_wave,"wave_superres":kernel_buffer_wave_superres},f)

def load_buffers():
	global kernel_buffer_velocity,kernel_buffer_velocity_superres,kernel_buffer_pressure,kernel_buffer_pressure_superres,kernel_buffer_wave,kernel_buffer_wave_superres
	path = 'Logger/buffers/buffers.dic'
	with open(path,"rb") as f:
		buffers = pickle.load(f)
		kernel_buffer_velocity = buffers["vel"]
		kernel_buffer_velocity_superres = buffers["vel_superres"]
		kernel_buffer_pressure = buffers["pres"]
		kernel_buffer_pressure_superres = buffers["pres_superres"]
		kernel_buffer_wave = buffers["wave"]
		kernel_buffer_wave_superres = buffers["wave_superres"]

try:
	load_buffers()
	print("loaded buffers")
except:
	print("no buffers available")

# interpolate hidden_states
def interpolate_states(old_hidden_states,new_hidden_states,offset,dt=1,orders_v=[2,2],orders_p=[0,0]):
	"""
	:old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
	:new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
	:offset: offset in x / y / t direction (vector of size 3 containing values between 0 and 1)
	:dt: delta time between old and new hidden states
	:orders_v: spline orders for vector potential of velocity field in x and y direction
	:orders_p: spline orders for pressure field in x and y direction
	:return: interpolated fields for:
		:a_z: vector potential
		:v: veloctiy field
		:dv_dt: time derivative of velocity field
		:grad_v: gradient of velocity field
		:laplace_v: laplacian of velocity field
		:p: pressure field
		:grad_p: gradient of pressure field
	"""
	v_size = np.prod([i+1 for i in orders_v])
	
	old_a_z,old_v,old_grad_v,old_laplace_v = interpolate_2d_velocity(old_hidden_states[:,:v_size],offset[0:2],orders_v)
	new_a_z,new_v,new_grad_v,new_laplace_v = interpolate_2d_velocity(new_hidden_states[:,:v_size],offset[0:2],orders_v)
	
	old_p,old_grad_p = interpolate_2d_pressure(old_hidden_states[:,v_size:],offset[0:2],orders_p)
	new_p,new_grad_p = interpolate_2d_pressure(new_hidden_states[:,v_size:],offset[0:2],orders_p)
	
	# time is interpolated linearly
	a_z = (1-offset[2])*old_a_z + offset[2]*new_a_z
	v = (1-offset[2])*old_v + offset[2]*new_v
	grad_v = (1-offset[2])*old_grad_v + offset[2]*new_grad_v
	laplace_v = (1-offset[2])*old_laplace_v + offset[2]*new_laplace_v
	p = (1-offset[2])*old_p + offset[2]*new_p
	grad_p = (1-offset[2])*old_grad_p + offset[2]*new_grad_p
	
	dv_dt = (new_v - old_v)/dt
	
	return a_z,v,dv_dt,grad_v,laplace_v,p,grad_p

# interpolate hidden_states for wave equation (second order spline in time)
def interpolate_wave_states(old_hidden_states,new_hidden_states,offset,dt=1,orders_z=[1,1]):
	"""
	:old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
	:new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
	:offset: offset in x / y / t direction (vector of size 3 containing values between 0 and 1)
	:dt: delta time between old and new hidden states
	:orders_z: spline orders for z values of membrane in x and y direction
	:return: interpolated fields for:
		:z: z field
		:grad(z): gradient of z field
		:laplace(z): laplacian of z field
		:dz/dt: velocity of z field
		:dz^2/dt^2: acceleration of z field
	"""
	z_size = np.prod([i+1 for i in orders_z])
	
	# first order in time
	old_z1,old_grad_z1,old_laplace_z1 = interpolate_2d_wave(old_hidden_states[:,:z_size],offset[0:2],orders_z)
	new_z1,new_grad_z1,new_laplace_z1 = interpolate_2d_wave(new_hidden_states[:,:z_size],offset[0:2],orders_z)
	
	# second order in time
	old_z2,old_grad_z2,old_laplace_z2 = interpolate_2d_wave(old_hidden_states[:,z_size:],offset[0:2],orders_z)
	new_z2,new_grad_z2,new_laplace_z2 = interpolate_2d_wave(new_hidden_states[:,z_size:],offset[0:2],orders_z)
	
	# 2nd order interpolation of time
	z = p2_1(offset[2])*old_z1 + p2_1(offset[2]-1)*new_z1 + p2_2(offset[2])*old_z2 + p2_2(offset[2]-1)*new_z2
	grad_z = p2_1(offset[2])*old_grad_z1 + p2_1(offset[2]-1)*new_grad_z1 + p2_2(offset[2])*old_grad_z2 + p2_2(offset[2]-1)*new_grad_z2
	laplace_z = p2_1(offset[2])*old_laplace_z1 + p2_1(offset[2]-1)*new_laplace_z1 + p2_2(offset[2])*old_laplace_z2 + p2_2(offset[2]-1)*new_laplace_z2
	v = (dp2_1(offset[2])*old_z1 + dp2_1(offset[2]-1)*new_z1 + dp2_2(offset[2])*old_z2 + dp2_2(offset[2]-1)*new_z2)/dt
	a = (d2p2_1(offset[2])*old_z1 + d2p2_1(offset[2]-1)*new_z1 + d2p2_2(offset[2])*old_z2 + d2p2_2(offset[2]-1)*new_z2)/(dt**2)
	
	return z,grad_z,laplace_z,v,a

# interpolate hidden_states for wave equation (separate z and v fields)
def interpolate_wave_states_2(old_hidden_states,new_hidden_states,offset,dt=1,orders_z=[1,1]):
	"""
	:old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
	:new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
	:offset: offset in x / y / t direction (vector of size 3 containing values between 0 and 1)
	:dt: delta time between old and new hidden states
	:orders_z: spline orders for z values of membrane in x and y direction
	:return: interpolated fields for:
		:z: z field
		:grad(z): gradient of z field
		:laplace(z): laplacian of z field
		:dz/dt: velocity of z field
		:dz^2/dt^2: acceleration of z field
	"""
	z_size = np.prod([i+1 for i in orders_z])
	
	# z field
	old_z,old_grad_z,old_laplace_z = interpolate_2d_wave(old_hidden_states[:,:z_size],offset[0:2],orders_z)
	new_z,new_grad_z,new_laplace_z = interpolate_2d_wave(new_hidden_states[:,:z_size],offset[0:2],orders_z)
	
	# v field
	old_v,old_grad_v,old_laplace_v = interpolate_2d_wave(old_hidden_states[:,z_size:],offset[0:2],orders_z)
	new_v,new_grad_V,new_laplace_v = interpolate_2d_wave(new_hidden_states[:,z_size:],offset[0:2],orders_z)
	
	# first order interpolation of z and v fields
	z = (1-offset[2])*old_z + offset[2]*new_z
	grad_z = (1-offset[2])*old_grad_z + offset[2]*new_grad_z
	laplace_z = (1-offset[2])*old_laplace_z + offset[2]*new_laplace_z
	dz_dt = (new_z-old_z)/dt
	v = (1-offset[2])*old_v + offset[2]*new_v # dzdt and v should be the same -> add residual loss!
	a = (new_v-old_v)/dt
	
	return z,grad_z,laplace_z,dz_dt,v,a

def superres_states(old_hidden_states,new_hidden_states,offset,dt=1,orders_v=[2,2],orders_p=[0,0],resolution_factor=1):#is not used... => remove
	"""
	:old_hidden_states: old hidden states (size: bs x (v_size+p_size) x w x h)
	:new_hidden_states: new hidden states (size: bs x (v_size+p_size) x w x h)
	:offset: offset in time direction (value between 0 and 1)
	:dt: delta time between old and new hidden states
	:orders_v: spline orders for vector potential of velocity field in x and y direction
	:orders_p: spline orders for pressure field in x and y direction
	:resolution_factor: superresolution factor
	:return: superresolution fields for:
		:a_z: vector potential
		:v: veloctiy field
		:dv_dt: time derivative of velocity field
		:grad_v: gradient of velocity field
		:laplace_v: laplacian of velocity field
		:p: pressure field
		:grad_p: gradient of pressure field
	"""
	v_size = np.prod([i+1 for i in orders_v])
	
	old_a_z,old_v,old_grad_v,old_laplace_v = superres_2d_velocity(old_hidden_states[:,:v_size],orders_v,resolution_factor)
	new_a_z,new_v,new_grad_v,new_laplace_v = superres_2d_velocity(new_hidden_states[:,:v_size],orders_v,resolution_factor)
	
	old_p,old_grad_p = superres_2d_pressure(old_hidden_states[:,v_size:],orders_p,resolution_factor)
	new_p,new_grad_p = superres_2d_pressure(new_hidden_states[:,v_size:],orders_p,resolution_factor)
	
	# time is interpolated linearly
	a_z = (1-offset)*old_a_z + offset*new_a_z
	v = (1-offset)*old_v + offset*new_v
	grad_v = (1-offset)*old_grad_v + offset*new_grad_v
	laplace_v = (1-offset)*old_laplace_v + offset*new_laplace_v
	p = (1-offset)*old_p + offset*new_p
	grad_p = (1-offset)*old_grad_p + offset*new_grad_p
	
	dv_dt = (new_v - old_v)/dt
	
	return a_z,v,dv_dt,grad_v,laplace_v,p,grad_p

def interpolate_2d_velocity(weights,offsets,orders=[2,2]):
	"""
	Idea: return derivatives of splines directly, implement with convolutions
	:weights: size: bs x (orders[0]+1) * (orders[1]+1) x w x h
	:offsets: offsets to interpolate in between weights, size: 2
	:orders: orders of spline for each dimension (note: counting starts at 0 => 0 ~ 1st order, 1 ~ 2nd order, 2 ~ 3rd order)
	:return: a_z,v,grad_v,laplace_v - note that, width / height is decreased by 1, because we only interpolate in between support points (weights)
		:a_z: vector potential of velocity field, size: bs x 1 x (w-1) x (h-1)
		:rot(a_z): velocity field, size: bs x 2 x (w-1) x (h-1)
		:grad(rot(a_z)): gradient (jacobian) of velocity field (dvx/dx dvx/dy dvy/dx dvy/dy), size: bs x 4 x (w-1) x (h-1)
		:laplace(rot(a_z)): laplacian of velocity field (laplace(vx) laplace(vy)), size:  bs x 2 x (w-1) x (h-1)
	"""
	# construct kernel matrix for 2x2 convolution based on offset:
	# => number of input channels = (orders[0]+1) * (orders[1]+1)
	# => number of output channels = 1 + 2 + 4 + 2 (a_z,v=rot(a_z),grad(v_x),grad(v_y),laplace(v_x),laplace(v_y)
	offset_key = f"{offsets[0]} {offsets[1]}, orders: {orders}"
	if offset_key in kernel_buffer_velocity.keys():
		kernels = kernel_buffer_velocity[offset_key]
	else:
		offsets = (offsets.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-offset_summary)
		offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1).detach().requires_grad_(True)
		
		kernels = toCuda(torch.zeros(1,1+2+4+2,(orders[0]+1),(orders[1]+1),2,2))
		for l in range(orders[0]+1):
			for m in range(orders[1]+1):
				kernels[0:1,0:1,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
		
		# velocity
		kernels[0:1,1:3,:,:,:,:] = rot(kernels[:,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)
		# gradients of velocity
		kernels[0:1,3:5] = grad(kernels[0:1,1:2,:,:,:,:],offsets,create_graph=True,retain_graph=True)
		kernels[0:1,5:7] = grad(kernels[0:1,2:3,:,:,:,:],offsets,create_graph=True,retain_graph=True)
		# laplace of velocity
		kernels[0:1,7:8] = div(kernels[0:1,3:5],offsets,retain_graph=True)
		kernels[0:1,8:9] = div(kernels[0:1,5:7],offsets,retain_graph=False)
		
		kernels = kernels.reshape(1,1+2+4+2,(orders[0]+1)*(orders[1]+1),2,2).detach()
		
		# buffer kernels
		kernel_buffer_velocity[offset_key] = kernels
		save_buffers()
	
	output = F.conv2d(weights,kernels[0],padding=0)
	
	# CODO: to be even more efficient, we could separate interpolation in x/y direction
	return output[:,0:1],output[:,1:3],output[:,3:7],output[:,7:9]

#  superresolution with strided convolution
def superres_2d_velocity(weights,orders=[2,2],resolution_factor=1):
	res_key = f"{resolution_factor}, orders: {orders}"
	if res_key in kernel_buffer_velocity_superres.keys():
		superres_kernels = kernel_buffer_velocity_superres[res_key]
	else:
		superres_kernels = toCuda(torch.zeros(1,1+2+4+2,(orders[0]+1)*(orders[1]+1),2*resolution_factor,2*resolution_factor))
		
		for i in range(resolution_factor):
			for j in range(resolution_factor):
				offsets = (toCuda(torch.tensor([i/resolution_factor,j/resolution_factor])).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-1+offset_summary)
				offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1).detach().requires_grad_(True)
				
				kernels = toCuda(torch.zeros(1,1+2+4+2,(orders[0]+1),(orders[1]+1),2,2))
				for l in range(orders[0]+1):
					for m in range(orders[1]+1):
						kernels[0:1,0:1,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
				
				# velocity
				kernels[0:1,1:3,:,:,:,:] = rot(kernels[0:1,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)
				# gradients of velocity
				kernels[0:1,3:5] = grad(kernels[0:1,1:2,:,:,:,:],offsets,create_graph=True,retain_graph=True)
				kernels[0:1,5:7] = grad(kernels[0:1,2:3,:,:,:,:],offsets,create_graph=True,retain_graph=True)
				# laplace of velocity
				kernels[0:1,7:8] = div(kernels[0:1,3:5],offsets,retain_graph=True)
				kernels[0:1,8:9] = div(kernels[0:1,5:7],offsets,retain_graph=False)
				
				kernels = kernels.reshape(1,1+2+4+2,(orders[0]+1)*(orders[1]+1),2,2).detach().clone()
				superres_kernels[:,:,:,i::resolution_factor,j::resolution_factor] = kernels
		
		# buffer kernels
		superres_kernels = superres_kernels.permute(0,2,1,3,4)
		kernel_buffer_velocity_superres[res_key] = superres_kernels
		save_buffers()
	
	output = F.conv_transpose2d(weights,superres_kernels[0],padding=0,stride=resolution_factor)
	
	return output[:,0:1],output[:,1:3],output[:,3:7],output[:,7:9]

def interpolate_2d_pressure(weights,offsets,orders=[0,0]):
	"""
	Idea: return derivatives of splines directly, implement with convolutions
	:weights: size: bs x (orders[0]+1) * (orders[1]+1) x w x h
	:offsets: offsets to interpolate in between weights, size: 2
	:orders: orders of spline for each dimension (note: counting starts at 0 => 0 ~ 1st order, 1 ~ 2nd order, 2 ~ 3rd order)
	:return: p,grad_p - note that, width / height is decreased by 1, because we only interpolate in between support points (weights)
		:p: pressure field, size: bs x 1 x (w-1) x (h-1)
		:grad(p): gradient of pressure field, size: bs x 2 x (w-1) x (h-1)
	"""
	offset_key = f"{offsets[0]} {offsets[1]}, orders: {orders}"
	if offset_key in kernel_buffer_pressure.keys():
		kernels = kernel_buffer_pressure[offset_key]
	else:
		offsets = (offsets.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-offset_summary)
		offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1).detach().requires_grad_(True)
		
		kernels = toCuda(torch.zeros(1,1+2,(orders[0]+1),(orders[1]+1),2,2))
		for l in range(orders[0]+1):
			for m in range(orders[1]+1):
				kernels[0:1,0:1,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
		
		# grad p
		kernels[0:1,1:3,:,:,:,:] = grad(kernels[:,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)
		
		kernels = kernels.reshape(1,1+2,(orders[0]+1)*(orders[1]+1),2,2).detach()
		
		# buffer kernels
		kernel_buffer_pressure[offset_key] = kernels
		save_buffers()
	
	output = F.conv2d(weights,kernels[0],padding=0)
	
	return output[:,0:1],output[:,1:3]

#  superresolution with strided convolution
def superres_2d_pressure(weights,orders=[0,0],resolution_factor=1):
	res_key = f"{resolution_factor}, orders: {orders}"
	if res_key in kernel_buffer_pressure_superres.keys():
		superres_kernels = kernel_buffer_pressure_superres[res_key]
	else:
		superres_kernels = toCuda(torch.zeros(1,1+2,(orders[0]+1)*(orders[1]+1),2*resolution_factor,2*resolution_factor))
		
		for i in range(resolution_factor):
			for j in range(resolution_factor):
				offsets = (toCuda(torch.tensor([i/resolution_factor,j/resolution_factor])).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-1+offset_summary)
				offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1).detach().requires_grad_(True)
				
				kernels = toCuda(torch.zeros(1,1+2,(orders[0]+1),(orders[1]+1),2,2))
				for l in range(orders[0]+1):
					for m in range(orders[1]+1):
						kernels[0:1,0:1,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
				
				# grad p
				kernels[0:1,1:3,:,:,:,:] = grad(kernels[:,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)
				
				kernels = kernels.reshape(1,1+2,(orders[0]+1)*(orders[1]+1),2,2).detach().clone()
				superres_kernels[:,:,:,i::resolution_factor,j::resolution_factor] = kernels
		
		# buffer kernels
		superres_kernels = superres_kernels.permute(0,2,1,3,4)
		kernel_buffer_pressure_superres[res_key] = superres_kernels
		save_buffers()
	
	output = F.conv_transpose2d(weights,superres_kernels[0],padding=0,stride=resolution_factor)
	
	return output[:,0:1],output[:,1:3]


def interpolate_2d_wave(weights,offsets,orders=[1,1]):
	"""
	Idea: return derivatives of splines directly, implement with convolutions
	:weights: size: bs x (orders[0]+1) * (orders[1]+1) x w x h
	:offsets: offsets to interpolate in between weights, size: 2
	:orders: orders of spline for each dimension (note: counting starts at 0 => 0 ~ 1st order, 1 ~ 2nd order, 2 ~ 3rd order)
	:return: z,grad_z,laplace_z - note that, width / height is decreased by 1, because we only interpolate in between support points (weights)
		:z: z value of membrane, size: bs x 1 x (w-1) x (h-1)
		:grad(z): gradient of z field (dz/dx dz/dy), size: bs x 2 x (w-1) x (h-1)
		:laplace(z): laplacian of z values, size:  bs x 1 x (w-1) x (h-1)
	"""
	offset_key = f"{offsets[0]} {offsets[1]}, orders: {orders}"
	if offset_key in kernel_buffer_wave.keys():
		kernels = toCuda(kernel_buffer_wave[offset_key])
	else:
		offsets = (offsets.clone().unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-offset_summary)
		offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1).detach().requires_grad_(True)
		
		# z_value
		kernels = toCuda(torch.zeros(1,1+2+1,(orders[0]+1),(orders[1]+1),2,2))
		for l in range(orders[0]+1):
			for m in range(orders[1]+1):
				kernels[0:1,0:1,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
		
		# gradients of z_value
		kernels[0:1,1:3] = grad(kernels[0:1,0:1,:,:,:,:],offsets,create_graph=True,retain_graph=True)
		
		# laplace of z_value
		kernels[0:1,3:4] = div(kernels[0:1,1:3],offsets,retain_graph=False)
		
		kernels = kernels.reshape(1,1+2+1,(orders[0]+1)*(orders[1]+1),2,2).detach()
		
		# buffer kernels
		kernel_buffer_wave[offset_key] = kernels
		save_buffers()
	
	output = F.conv2d(weights,kernels[0],padding=0)
	
	# CODO: to be even more efficient, we could separate interpolation in x/y direction
	return output[:,0:1],output[:,1:3],output[:,3:4]

#  superresolution with strided convolution
def superres_2d_wave(weights,orders=[1,1],resolution_factor=1):
	"""
	:return: z,grad_z,laplace_z,v,a
	"""
	res_key = f"{resolution_factor}, orders: {orders}"
	if res_key in kernel_buffer_wave_superres.keys():
		superres_kernels = kernel_buffer_wave_superres[res_key]
	else:
		superres_kernels = toCuda(torch.zeros(1,1+2+1+1+1,2*(orders[0]+1)*(orders[1]+1),2*resolution_factor,2*resolution_factor))
		for i in range(resolution_factor):
			for j in range(resolution_factor):
				offsets = (toCuda(torch.tensor([i/resolution_factor,j/resolution_factor])).unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(1,1,2,2)-1+offset_summary)
				offsets = offsets.unsqueeze(2).unsqueeze(3).repeat(1,1,(orders[0]+1),(orders[1]+1),1,1).detach().requires_grad_(True)
				
				kernels = toCuda(torch.zeros(1,1+2+1+1+1,2,(orders[0]+1),(orders[1]+1),2,2))
				for l in range(orders[0]+1):
					for m in range(orders[1]+1):
						kernels[0:1,0:1,0,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
				
				# gradients of z_value
				kernels[0:1,1:3,0,:,:,:,:] = grad(kernels[0:1,0:1,0,:,:,:,:],offsets,create_graph=True,retain_graph=True)
				
				# laplace of z_value
				kernels[0:1,3:4,0,:,:,:,:] = div(kernels[0:1,1:3],offsets,retain_graph=False)
				
				#v and a
				for l in range(orders[0]+1):
					for m in range(orders[1]+1):
						kernels[0:1,4:5,1,l,m,:,:] = p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
						kernels[0:1,5:6,0,l,m,:,:] = -6*p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
						kernels[0:1,5:6,1,l,m,:,:] = -4*p_multidim(offsets[:,:,l,m],[orders[0],orders[1]],[l,m])
				
				kernels = kernels.reshape(1,1+2+1+1+1,2*(orders[0]+1)*(orders[1]+1),2,2).detach()
			
				kernels = kernels.reshape(1,1+2+1+1+1,2*(orders[0]+1)*(orders[1]+1),2,2).detach().clone()
				superres_kernels[:,:,:,i::resolution_factor,j::resolution_factor] = kernels
		
		# buffer kernels
		superres_kernels = superres_kernels.permute(0,2,1,3,4)
		kernel_buffer_wave_superres[res_key] = superres_kernels
		save_buffers()
	
	output = F.conv_transpose2d(weights,superres_kernels[0],padding=0,stride=resolution_factor)
	
	return output[:,0:1],output[:,1:3],output[:,3:4],output[:,4:5],output[:,5:6]


