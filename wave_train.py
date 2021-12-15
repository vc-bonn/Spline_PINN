from wave_setups import Dataset
from spline_models import interpolate_wave_states_2,superres_2d_wave,get_Net
from operators import vector2HSV
from get_param import params,toCuda,toCpu,get_hyperparam_wave
from Logger import Logger
import cv2
from torch.optim import Adam
import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

stiffness = params.stiffness
damping = params.damping
dt = params.dt
params.width = 200 if params.width is None else params.width
params.height = 200 if params.height is None else params.height
resolution_factor = params.resolution_factor
orders_z = [params.orders_z,params.orders_z]
z_size = np.prod([i+1 for i in orders_z])

# initialize dataset
dataset = Dataset(params.width,params.height,hidden_size=2*z_size,batch_size=params.batch_size,n_samples=params.n_samples,dataset_size=params.dataset_size,average_sequence_length=params.average_sequence_length,dt=params.dt,types=["super_simple","oscillator","box"])#,"oscillator","box"

# initialize fluid model and optimizer
model = toCuda(get_Net(params))
optimizer = Adam(model.parameters(),lr=params.lr)

# initialize Logger and load model / optimizer if according parameters were given
logger = Logger(get_hyperparam_wave(params),use_csv=False,use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_hyperparam_wave(params),use_csv=False,use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = logger.load_state(model,optimizer,params.load_date_time,params.load_index)
	else:
		params.load_date_time, params.load_index = logger.load_state(model,None,params.load_date_time,params.load_index)
	params.load_index=int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
	if params.n_warmup_steps is not None:
		model.eval()
		for i in range(params.n_warmup_steps):
			z_cond,z_mask,old_hidden_state,_,_,_ = toCuda(dataset.ask())
			new_hidden_state = model(old_hidden_state,z_cond,z_mask)
			dataset.tell(toCpu(new_hidden_state))
			if i%(params.n_warmup_steps//100)==0:
				print(f"warmup {i/(params.n_warmup_steps//100)} %")
		model.train()
		
params.load_index = 0 if params.load_index is None else params.load_index

# diffusion operation (needed, if we want to put more loss-weight to regions close to the domain boundaries)
kernel_width = 3
kernel = torch.exp(-torch.arange(-2,2.001,4/(2*kernel_width)).float()**2)
kernel /= torch.sum(kernel)
kernel_x = toCuda(kernel.unsqueeze(0).unsqueeze(1).unsqueeze(3))
kernel_y = toCuda(kernel.unsqueeze(0).unsqueeze(1).unsqueeze(2))
def diffuse(T): # needed to put extra weight on domain borders
	T = F.conv2d(T,kernel_x,padding=[kernel_width,0])
	T = F.conv2d(T,kernel_y,padding=[0,kernel_width])
	return T

for epoch in range(params.load_index,params.n_epochs):
	
	for i in range(params.n_batches_per_epoch):
		print(f"{epoch} / {params.n_epochs}: {i}")
		
		# retrieve batch from training pool
		z_cond,z_mask,old_hidden_state,offsets,sample_z_conds,sample_z_masks = toCuda(dataset.ask())
		
		# predict new hidden state that contains the spline coeffients of the next timestep
		new_hidden_state = model(old_hidden_state,z_cond,z_mask)
		
		# compute physics informed loss
		loss_total = 0
		for j in range(params.n_samples):
			offset = torch.floor(offsets[j]*resolution_factor)/resolution_factor
			sample_z_cond = sample_z_conds[j]
			sample_z_mask = sample_z_masks[j]
			sample_domain_mask = 1-sample_z_mask
			# optionally: put additional border_weight on domain boundaries:
			sample_z_mask = (sample_z_mask + sample_z_mask*diffuse(sample_domain_mask)*params.border_weight).detach()
			
			# interpolate spline coeffients to obtain: z, grad_z, laplace_z, dz_dt, v, a
			z,grad_z,laplace_z,dz_dt,v,a = interpolate_wave_states_2(old_hidden_state,new_hidden_state,offset,dt=dt,orders_z=orders_z)
			
			# boundary loss
			loss_boundary = torch.mean(sample_z_mask[:,:,1:-1,1:-1]*((z-sample_z_cond[:,:,1:-1,1:-1])**2),dim=(1,2,3))
			loss_boundary_reg = torch.mean(sample_z_mask[:,:,1:-1,1:-1]*(a**2),dim=(1,2,3))
			
			# loss to connect dz_dt and v
			loss_v = torch.mean((v-dz_dt)**2,dim=(1,2,3))
			
			# wave equation loss
			loss_wave = torch.mean((a-stiffness*laplace_z+damping*v)**2,dim=(1,2,3))
			
			loss_total = loss_total + params.loss_bound*loss_boundary + params.loss_bound_reg*loss_boundary_reg + params.loss_wave*loss_wave + params.loss_v*loss_v
		
		if params.log_loss:
			loss_total = torch.mean(torch.log(loss_total))/params.n_samples
		else:
			loss_total = torch.mean(loss_total)/params.n_samples
		
		# reset old gradients to 0 and compute new gradients with backpropagation
		model.zero_grad()
		loss_total.backward()
		
		# clip gradients
		if params.clip_grad_value is not None:
			torch.nn.utils.clip_grad_value_(model.parameters(),params.clip_grad_value)
		
		if params.clip_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(model.parameters(),params.clip_grad_norm)
		
		# optimize wave model
		optimizer.step()
		
		# tell dataset new hidden_state
		dataset.tell(toCpu(new_hidden_state))
		
		# log training metrics
		if i%10 == 0:
			logger.log(f"loss_total",loss_total.detach().cpu().numpy(),epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_wave",torch.mean(loss_wave).detach().cpu().numpy(),epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_v",torch.mean(loss_v).detach().cpu().numpy(),epoch*params.n_batches_per_epoch+i)
			logger.log(f"loss_boundary",torch.mean(loss_boundary).detach().cpu().numpy(),epoch*params.n_batches_per_epoch+i)
			
	if params.log:
		logger.save_state(model,optimizer,epoch+1)
		
