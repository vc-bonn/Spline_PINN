from fluid_setups import Dataset
from spline_models import superres_2d_velocity,superres_2d_pressure,get_Net,interpolate_states
from operators import vector2HSV
from get_param import params,toCuda,toCpu,get_hyperparam_fluid
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

save_movie=False#True#
movie_FPS = 30.0#8.0#15.0 # ... choose FPS as provided in visualization
params.width = 230 if params.width is None else params.width
params.height = 49 if params.height is None else params.height
resolution_factor = params.resolution_factor
orders_v = [params.orders_v,params.orders_v]
orders_p = [params.orders_p,params.orders_p]
v_size,p_size = np.prod([i+1 for i in orders_v]),np.prod([i+1 for i in orders_p])
mouse_paint,mouse_erase = False,False

# dataset types to randomly choose from
types = ["paint","DFG_benchmark","magnus"] # interactive paint, DFG benchmark and Magnus effect environment. Further types to choose from: "box","image","ecmo","poiseuille"

# initialize dataset
dataset = Dataset(params.width,params.height,hidden_size=v_size+p_size,interactive=True,batch_size=1,n_samples=1,dataset_size=1,average_sequence_length=params.average_sequence_length,types=types,dt=params.dt,resolution_factor=resolution_factor,images=["cyber","fish","smiley","wing"])

# initialize windows / movies / mouse handler
cv2.namedWindow('a_z',cv2.WINDOW_NORMAL)
cv2.namedWindow('v',cv2.WINDOW_NORMAL)
cv2.namedWindow('p',cv2.WINDOW_NORMAL)

if save_movie:
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	movie_p = cv2.VideoWriter(f'plots/p_{get_hyperparam_fluid(params)}.avi', fourcc, movie_FPS, ((params.height-2)*resolution_factor,(params.width-2)*resolution_factor))
	movie_v = cv2.VideoWriter(f'plots/v_{get_hyperparam_fluid(params)}.avi', fourcc, movie_FPS, ((params.height-2)*resolution_factor,(params.width-2)*resolution_factor))
	movie_a = cv2.VideoWriter(f'plots/a_{get_hyperparam_fluid(params)}.avi', fourcc, movie_FPS, ((params.height-2)*resolution_factor,(params.width-2)*resolution_factor))

def mousePosition(event,x,y,flags,param):
	global dataset
	if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1 or mouse_erase or mouse_paint:
		if mouse_erase:
			dataset.mouse_erase = True
		if mouse_paint:
			dataset.mouse_paint = True
		dataset.mousex = y/resolution_factor
		dataset.mousey = x/resolution_factor

cv2.setMouseCallback("a_z",mousePosition)
cv2.setMouseCallback("v",mousePosition)
cv2.setMouseCallback("p",mousePosition)

cv2.namedWindow('v_legend',cv2.WINDOW_NORMAL)
vector = torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)])
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('v_legend',image)

# load fluid model
model = toCuda(get_Net(params))
logger = Logger(get_hyperparam_fluid(params),use_csv=False,use_tensorboard=False)
date_time,index = logger.load_state(model,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, index: {index}")
model.eval()

FPS = 0
last_FPS = 0
last_time = time.time()

# compute pressure and viscous forces
def forces(grad_v,p,x,n):
	"""
	:grad_v: gradient of velocity field
	:p: pressure field
	:x: location of measurement (will be rounded to integers) [n_samples x 2]
	:n: surface normal [n_samples x 2]
	:return:
		:pressure_force: pressure force [n_samples x 2]
		:viscous_force: viscous force [n_samples x 2]
	"""
	pos = x.long()
	grads_v = grad_v[0,:,pos[:,0],pos[:,1]].permute(1,0)
	grads_v = torch.cat([grads_v[:,0:2].unsqueeze(1),grads_v[:,2:4].unsqueeze(1)],dim=1)
	dv_dn = torch.matmul(grads_v,n.unsqueeze(2))
	ps = p[0,0,pos[:,0],pos[:,1]]
	ps = ps-torch.mean(ps)
	pressure_force = ps.unsqueeze(1)*n[:,:]
	viscous_force = params.mu*dv_dn[:,:,0]
	return -pressure_force.detach(),viscous_force.detach()

# simulation loop
exit_loop = False
while not exit_loop:
	
	# reset environment (choose new random environment from types-list and reset velocity / pressure field to 0)
	dataset.reset_env(0)
	
	# buffers for drag / lift coefficients
	cd_cl_t = torch.zeros(200,2)
	
	dataset.mousev = 1#1.5
	
	for i in range(params.average_sequence_length):
		#print(f"{epoch} / {i}")
		
		# obtain boundary conditions / mask as well as spline coefficients of previous timestep from dataset
		v_cond,v_mask,old_hidden_state,_,_,_ = toCuda(dataset.ask())
		
		# apply fluid model to obtain spline coefficients of next timestep
		new_hidden_state = model(old_hidden_state,v_cond,v_mask)
		
		# feed new spline coefficients back to the dataset
		dataset.tell(toCpu(new_hidden_state))
		
		
		# compute drag and lift coefficients
		if dataset.env_info[0]["type"] == "magnus" or dataset.env_info[0]["type"] == "DFG_benchmark":
			radius = dataset.env_info[0]["r"]
			x = torch.FloatTensor([[(dataset.env_info[0]["x"]+radius*np.cos(phi))*resolution_factor,(dataset.env_info[0]["y"]+radius*np.sin(phi))*resolution_factor] for phi in np.arange(0,2*np.pi,0.05)])
			n = torch.FloatTensor([[np.cos(phi),np.sin(phi)] for phi in np.arange(0,2*np.pi,0.05)])
			a_z,v,grad_v,laplace_v = superres_2d_velocity(new_hidden_state[0:1,:v_size],orders_v,resolution_factor)
			p,grad_p = superres_2d_pressure(new_hidden_state[0:1,v_size:],orders_p,resolution_factor)
			pressure_force,viscous_force = forces(grad_v.cpu(),p.cpu(),x,n)
			pressure_force = torch.mean(pressure_force,dim=0)*2*3.14*radius
			viscous_force = torch.mean(viscous_force,dim=0)*2*3.14*radius
			total_force = pressure_force+viscous_force
			cd_cl = total_force*2/(2*radius)/((dataset.mousev*2/3)**2)/params.rho
			cd_cl_t[0:-1,:] = cd_cl_t[1:,:].clone()
			cd_cl_t[-1] = cd_cl
			print(f"C_D / C_L = {cd_cl}")
		
		# visualize fields
		if i%1==0:
			
			print(f"env_info: {dataset.env_info[0]}")
			
			# obtain interpolated field values for a_z,v,grad_v,laplace_v from spline coefficients of velocity field
			a_z,v,grad_v,laplace_v = superres_2d_velocity(new_hidden_state[0:1,:v_size],orders_v,resolution_factor)
			
			image = a_z[0,0,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu().detach().clone()
			image = image - torch.min(image)
			image /= torch.max(image)
			image = toCpu(image).unsqueeze(2).repeat(1,1,3).numpy()
			if save_movie:
				movie_a.write((255*image).astype(np.uint8))
			cv2.imshow('a_z',image)
			
			vector = v[0,:,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu().detach().clone()
			image = vector2HSV(vector)
			image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
			if save_movie:
				movie_v.write((255*image).astype(np.uint8))
			cv2.imshow('v',image)
			
			# obtain interpolated field values for p,grad_p from spline coefficients of pressure field
			p,grad_p = superres_2d_pressure(new_hidden_state[0:1,v_size:],orders_p,resolution_factor)
			image = (1-dataset.v_mask_full_res)*p.cpu()
			image = image[0,0,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].detach().clone()
			image = image - torch.min(image)
			image /= torch.max(image)
			image = toCpu(image).unsqueeze(2).repeat(1,1,3).numpy()
			if save_movie:
				movie_p.write((255*image).astype(np.uint8))
			cv2.imshow('p',image)
			
			key = cv2.waitKey(1)
			
			if key==ord('x'): # increase flow velocity
				dataset.mousev+=0.1
			elif key==ord('y'): # decrease flow velocity
				dataset.mousev-=0.1
			
			elif key==ord('1'): # different flow velocities...
				dataset.mousev=0.3
			elif key==ord('2'):
				dataset.mousev=0.5
			elif key==ord('3'):
				dataset.mousev=1
			elif key==ord('4'):
				dataset.mousev=1.5

			if key==ord('s'): # increase spin of cylinder
				dataset.mousew+=0.1
			elif key==ord('a'): # decrease spin of cylinder
				dataset.mousew-=0.1
			
			elif key==ord('r'): # reset position of cylinder
				dataset.mousex=25
				dataset.mousey=24
			
			if key==ord('w'): # 'write' in paint environment
				mouse_paint=True
			else:
				dataset.mouse_paint=False
				mouse_paint=False
			if key==ord('e'): # 'erase' in paint environment
				mouse_erase=True
			else:
				dataset.mouse_erase=False
				mouse_erase=False
			
			if key==ord('p'): # print fields using matplotlib
				
				os.makedirs("plots",exist_ok=True)
				name = dataset.env_info[0]["type"]
				if name=="image":
					name = name+"_"+dataset.env_info[0]["image"]
				
				# create pressure plot with streamlines
				flow = v[0,:,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu().detach().clone()
				image = vector2HSV(flow)
				flow = toCpu(flow).numpy()
				fig = plt.figure(1,figsize=(15,5))
				ax = fig.add_subplot()
				Y,X = np.mgrid[0:flow.shape[1],0:flow.shape[2]]
				linewidth = image[:,:,2]/np.max(image[:,:,2])
				ax.streamplot(Y.transpose(),X.transpose(),  flow[0].transpose()[::-1], -flow[1].transpose()[::-1], color='k', density=3,linewidth=2*linewidth.transpose()[::-1])
				palette = plt.cm.gnuplot2
				palette.set_bad('k',1.0)
				
				p = p[0,0,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu().detach()
				cond_mask = dataset.v_mask_full_res[0,0,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor]
				pm = np.ma.masked_where(toCpu(cond_mask).numpy()==1, toCpu(p).numpy())
				plt.imshow(pm.transpose()[::-1],cmap=palette)
				plt.axis('off')
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right",size="5%",pad=0.05)
				plt.colorbar(cax=cax)
				plt.savefig(f"plots/flow_and_pressure_field_{name}_{get_hyperparam_fluid(params)}_vel_{dataset.mousev}.png", bbox_inches='tight',dpi=300)
				
				# create velocity magnitude plot and pressure plot with streamlines
				fig = plt.figure(2,figsize=(30,5))
				fig.tight_layout()
				ax = fig.add_subplot(1,2,1)
				plt.imshow(np.linalg.norm(flow,axis=0).transpose()[::-1])
				plt.axis('off')
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right",size="5%",pad=0.05)
				plt.colorbar(cax=cax)
				
				ax = fig.add_subplot(1,2,2)
				ax.streamplot(Y.transpose(),X.transpose(),  flow[0].transpose()[::-1], -flow[1].transpose()[::-1], color='k', density=3,linewidth=2*linewidth.transpose()[::-1])
				plt.imshow(pm.transpose()[::-1],cmap=palette)
				plt.axis('off')
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right",size="5%",pad=0.05)
				plt.colorbar(cax=cax)
				plt.subplots_adjust(0.05,0.05,0.95,0.95,0.1,0.1)
				plt.savefig(f"plots/flow_and_pressure_field_sep_{name}_{get_hyperparam_fluid(params)}_vel_{dataset.mousev}.png", bbox_inches='tight',dpi=300)
				
				
				"""
				# save results in vtk files
				os.makedirs(f"vtk/{name}/{get_hyperparam_fluid(params)}",exist_ok=True)
				p -= torch.mean(p)
				imageToVTK(f"vtk/{name}/{get_hyperparam_fluid(params)}/pressure",pointData={"pressure":(p*(1-cond_mask)).unsqueeze(2).numpy()})
				v_new = (v[0,:,resolution_factor:-resolution_factor,resolution_factor:-resolution_factor].cpu()*(1-cond_mask)).unsqueeze(3).detach()
				imageToVTK(f"vtk/{name}/{get_hyperparam_fluid(params)}/velocity",cellData={"velocity":(v_new[0].numpy(),v_new[1].numpy(),0*v_new[1].numpy())})
				"""
				
				
				if dataset.env_info[0]["type"] == "magnus" or dataset.env_info[0]["type"] == "DFG_benchmark":
					
					
					# print min / average / max for c_d / c_l
					print(f"C_D: (min: {torch.min(cd_cl_t[:,0])} / avrg: {torch.mean(cd_cl_t[:,0])} / max: {torch.max(cd_cl_t[:,0])}) ; C_L: (min: {torch.min(cd_cl_t[:,1])} / avrg: {torch.mean(cd_cl_t[:,1])} / max: {torch.max(cd_cl_t[:,1])})")
					
					# compute c_d / c_l again
					p,_ = superres_2d_pressure(new_hidden_state[0:1,v_size:],orders_p,resolution_factor)
					pressure_force,viscous_force = forces(grad_v.cpu(),p.cpu(),x,n)
					total_force = torch.mean(pressure_force+viscous_force,dim=0)*2*3.14*radius
					cd_cl = total_force*2/(2*radius)/((dataset.mousev*2/3)**2)
					
					# visualize pressure forces:
					palette = plt.cm.gnuplot2
					palette.set_bad('k',1.0)
					p = p[0,0].cpu().detach()
					cond_mask = dataset.v_mask_full_res[0,0]
					pm = np.ma.masked_where(toCpu(cond_mask).numpy()==1, toCpu(p).numpy())
					flow = v[0,:].cpu().detach().clone()
					flow[:,:resolution_factor] = 0
					flow[:,:,:resolution_factor] = 0
					flow[:,-resolution_factor:] = 0
					flow[:,:,-resolution_factor:] = 0
					flow = toCpu(flow).numpy()
					
					fig = plt.figure(3,figsize=(12,5))
					plt.clf()
					ax = fig.add_subplot(1,2,2)
					plt.imshow(pm.transpose(),cmap=palette)
					pressure_force/= torch.max(pressure_force)
					pressure_force*=resolution_factor
					for j in range(0,x.shape[0],2):
						plt.arrow(x[j,0],x[j,1],pressure_force[j,0],pressure_force[j,1],head_width=0.4*resolution_factor,head_length=0.4*resolution_factor,zorder=10,fc='w',ec='w')
					plt.plot(torch.cat([x[:,0],x[0:1,0]]),torch.cat([x[:,1],x[0:1,1]]),color='w')
					plt.xlim((torch.min(x[:,0])-30),(torch.max(x[:,0])+30))
					plt.ylim((torch.min(x[:,1])-30),(torch.max(x[:,1])+30))
					plt.title("Pressure forces")
					plt.xlabel("x axis")
					plt.ylabel("y axis")
					plt.axis('off')
					divider = make_axes_locatable(ax)
					cax = divider.append_axes("right",size="5%",pad=0.05)
					plt.colorbar(cax=cax)
					
					# visualize viscous forces:
					ax = fig.add_subplot(1,2,1)
					plt.imshow(np.linalg.norm(flow,axis=0).transpose())
					viscous_force/= torch.max(viscous_force)
					viscous_force*=resolution_factor
					for j in range(0,x.shape[0],2):
						plt.arrow(x[j,0],x[j,1],viscous_force[j,0],viscous_force[j,1],head_width=0.4*resolution_factor,head_length=0.4*resolution_factor,zorder=10,fc='w',ec='w')
					plt.plot(torch.cat([x[:,0],x[0:1,0]]),torch.cat([x[:,1],x[0:1,1]]),color='w')
					plt.xlim((torch.min(x[:,0])-30),(torch.max(x[:,0])+30))
					plt.ylim((torch.min(x[:,1])-30),(torch.max(x[:,1])+30))
					plt.title("Viscous forces")
					plt.xlabel("x axis")
					plt.ylabel("y axis")
					plt.axis('off')
					divider = make_axes_locatable(ax)
					cax = divider.append_axes("right",size="5%",pad=0.05)
					plt.colorbar(cax=cax)
					plt.savefig(f"plots/pressure_viscous_forces_{name}_{get_hyperparam_fluid(params)}_vel_{dataset.mousev}.png", bbox_inches='tight',dpi=300)
					
					# plot drag and lift coefficients over time
					plt.figure(4)
					plt.plot(cd_cl_t[:,0])
					plt.plot(cd_cl_t[:,1])
					plt.title("$C_D$ / $C_L$ over time")
					plt.legend(["$C_D$","$C_L$"])
					plt.xlabel("time")
					plt.ylabel("$C_D$ / $C_L$")
				
				plt.show()
			
			if key==ord('n'):
				break
			
			if key==ord('q'):
				exit_loop = True
				break

			print(f"FPS: {last_FPS}")
			FPS += 1
			if time.time()-last_time>=1:
				last_time = time.time()
				last_FPS=FPS
				FPS = 0
				
if save_movie:
	movie_p.release()
	movie_v.release()
	movie_a.release()


