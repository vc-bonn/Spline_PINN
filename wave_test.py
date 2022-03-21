from fluid_setups import Dataset
from wave_setups import Dataset
from spline_models import superres_2d_wave,get_Net,interpolate_wave_states_2
from operators import vector2HSV
from get_param import params,toCuda,toCpu,get_hyperparam_wave
from Logger import Logger
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time,os
from numpy2vtk import imageToVTK
from mpl_toolkits.axes_grid1 import make_axes_locatable


torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

n_iterations_per_visualization = 1 # this value can be set to a higher integer if the cv2 visualizations impose a bottleneck on your computer and you want to speed up the simulation
save_movie=False#True#
movie_FPS = 20 # ... choose FPS as provided in visualization
params.width = 200 if params.width is None else params.width
params.height = 200 if params.height is None else params.height
resolution_factor = params.resolution_factor
orders_z = [params.orders_z,params.orders_z]
z_size = np.prod([i+1 for i in orders_z])
types = ["reflection","oscillator","doppler"]# further types: "box","simple","super_simple"

# initialize dataset
dataset = Dataset(params.width,params.height,hidden_size=2*z_size,interactive=True,batch_size=1,n_samples=params.n_samples,dataset_size=1,average_sequence_length=params.average_sequence_length,types=types,dt=params.dt,resolution_factor=resolution_factor)

# initialize windows / movies / mouse handler
cv2.namedWindow('z',cv2.WINDOW_NORMAL)
cv2.namedWindow('v',cv2.WINDOW_NORMAL)
cv2.namedWindow('a',cv2.WINDOW_NORMAL)

if save_movie:
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	movie_z = cv2.VideoWriter(f'plots/z_{get_hyperparam_wave(params)}.avi', fourcc, movie_FPS, (params.height*resolution_factor,params.width*resolution_factor))
	movie_v = cv2.VideoWriter(f'plots/v_{get_hyperparam_wave(params)}.avi', fourcc, movie_FPS, (params.height*resolution_factor,params.width*resolution_factor))
	movie_a = cv2.VideoWriter(f'plots/a_{get_hyperparam_wave(params)}.avi', fourcc, movie_FPS, (params.height*resolution_factor,params.width*resolution_factor))

def mousePosition(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
		dataset.mousex = y/resolution_factor
		dataset.mousey = x/resolution_factor

cv2.setMouseCallback("z",mousePosition)
cv2.setMouseCallback("v",mousePosition)
cv2.setMouseCallback("a",mousePosition)

# load fluid model
model = toCuda(get_Net(params))
logger = Logger(get_hyperparam_wave(params),use_csv=False,use_tensorboard=False)
date_time,index = logger.load_state(model,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, index: {index}")
model.eval()

FPS = 0
last_FPS = 0
last_time = time.time()

# simulation loop
exit_loop = False
while not exit_loop:
	
	# reset environment (choose new random environment from types-list and reset z / v_z field to 0)
	dataset.reset_env(0)
	
	for i in range(params.average_sequence_length):
		
		# obtain boundary conditions / mask as well as spline coefficients of previous timestep from dataset
		z_cond,z_mask,old_hidden_state,_,_,_ = toCuda(dataset.ask())
		
		# apply wave model to obtain spline coefficients of next timestep
		new_hidden_state = model(old_hidden_state,z_cond,z_mask)
		
		# feed new spline coefficients back to the dataset
		dataset.tell(toCpu(new_hidden_state))
		
		# visualize fields
		if i%n_iterations_per_visualization==0:
			
			print(f"env_info: {dataset.env_info[0]}")
			
			# obtain interpolated field values for z,grad_z,laplace_z,v,a from spline coefficients
			z,grad_z,laplace_z,v,a = superres_2d_wave(new_hidden_state[0:1],orders_z,resolution_factor)
			
			# visualize field values
			image = z[0,0].cpu().detach().clone()
			image = torch.clamp(0.5*image+0.5,min=0,max=1)
			image = toCpu(image).unsqueeze(2).repeat(1,1,3).numpy()
			if save_movie:
				movie_z.write((255*image).astype(np.uint8))
			cv2.imshow('z',image)
			
			image = v[0,0].cpu().detach().clone()
			image = torch.clamp(0.2*image+0.5,min=0,max=1)
			image = toCpu(image).unsqueeze(2).repeat(1,1,3).numpy()
			if save_movie:
				movie_v.write((255*image).astype(np.uint8))
			cv2.imshow('v',image)
			
			image = a[0,0].cpu().detach().clone()
			image = torch.clamp(0.03*image+0.5,min=0,max=1)
			image = toCpu(image).unsqueeze(2).repeat(1,1,3).numpy()
			if save_movie:
				movie_a.write((255*image).astype(np.uint8))
			cv2.imshow('a',image)
			
			key = cv2.waitKey(1)
			
			if key==ord('x'): # increase frequency (works only for 'box' environment)
				dataset.mousev*=1.1
			elif key==ord('y'): # decrease frequency (works only for 'box' environment)
				dataset.mousev/=1.1
			
			if key==ord('n'): # start with new environment
				break
			
			if key==ord('q'): # quit simulation
				exit_loop = True
				break
			
			if key==ord('p'): # print fields using matplotlib
				fig = plt.figure(1,(12,6))
				ax = fig.add_subplot(1,2,1)
				cond_mask = dataset.z_mask_full_res[0,0]
				pm = np.ma.masked_where(toCpu(cond_mask).numpy()==1, z[0,0].cpu().detach().clone())
				palette = plt.cm.viridis#plasma#gnuplot2#magma#inferno#
				palette.set_bad('k',1.0)
				plt.imshow(pm,cmap=palette)
				plt.axis('off')
				plt.title("z")
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right",size="5%",pad=0.05)
				plt.colorbar(cax=cax)
				
				ax = fig.add_subplot(1,2,2)
				cond_mask = dataset.z_mask_full_res[0,0]
				pm = np.ma.masked_where(toCpu(cond_mask).numpy()==1, v[0,0].cpu().detach().clone())
				plt.imshow(pm,cmap=palette)
				plt.axis('off')
				plt.title("v")
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right",size="5%",pad=0.05)
				plt.colorbar(cax=cax)
				
				name = dataset.env_info[0]["type"]
				if name=="image":
					name = name+"_"+dataset.env_info[0]["image"]
				plt.savefig(f"plots/wave_z_v_{name}_{get_hyperparam_wave(params)}.png", bbox_inches='tight')
				plt.show()
			
			print(f"FPS: {last_FPS}")
			FPS += 1
			if time.time()-last_time>=1:
				last_time = time.time()
				last_FPS=FPS
				FPS = 0

if save_movie:
	movie_z.release()
	movie_v.release()
	movie_a.release()


