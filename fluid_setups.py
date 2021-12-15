import torch
import torch.nn.functional as f
import numpy as np
from PIL import Image

"""
design of dataset-grid to make training easier:
o---o---o---o
|   |   |   |
| + | + | + |
|   |   |   |
o---O---O---o
|   |   |   |
| + | + | + |
|   |   |   |
o---O---O---o
|   |   |   |
| + | + | + |
|   |   |   |
o---o---o---o

-> in this example, the grid size is 3 x 3
-> and the resolution factor is 4
-> hidden_grid (big "O") size is 2 x 2
-> offsets are between 0 and 1 relative to "o/O"
-> inputs are averaged onto "+"
=> first, model needs to process inputs with a 2x2 convolution (without padding) before concatenating with hidden_grid.
=> the fully interpolated fields can only be computed if they are surrounded by the hidden_grid -> only in the center cell possible (1 x 1)
"""

cyber_truck = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/cyber.png'))).float(),dim=2)<100).float()
fish = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/fish.png'))).float(),dim=2)<100).float()
smiley = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/smiley.png'))).float(),dim=2)<100).float()
wing = (torch.mean(torch.tensor(np.asarray(Image.open('imgs/wing_profile.png'))).float(),dim=2)<100).float()

images = {"cyber":cyber_truck, "fish":fish, "smiley":smiley, "wing":wing}

class Dataset():
	
	def __init__(self,w,h,hidden_size,resolution_factor=4,batch_size=100,n_samples=1,dataset_size=1000,average_sequence_length=5000,interactive=False,max_speed=1,brown_damping=0.9995,brown_velocity=0.005,init_velocity=0,dt=1,types=["simple"],images=["cyber","fish","smiley","wing"]):
		"""
		:w,h: width / height of grid
		:hidden_size: size of hidden state
		:n_samples: number of samples (different "offsets") per batch
		:interactive: allows to interact with the dataset by changing the following variables:
			- mousex: x-position of obstacle
			- mousey: y-position of obstacle
			- mousev: velocity of fluid
			- mousew: angular velocity of ball
		:max_speed: maximum speed at dirichlet boundary conditions
		:brown_damping / brown_velocity: parameters for random motions in dataset
		:init_velocity: initial velocity of objects in simulation (can be ignored / set to 0)
		:resolution_factor: resolution factor for internal grid representation (e.g. 4 times higher resolution)
		"""
		self.w,self.h = w,h
		self.w_full_res,self.h_full_res = w*resolution_factor,h*resolution_factor
		#print(f"resolution: {self.w_full_res,self.h_full_res}")
		self.resolution_factor = resolution_factor
		self.batch_size = batch_size
		self.n_samples = n_samples
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		self.interactive = interactive
		self.interactive_spring = 150#300#200#~ 1/spring constant to move object
		self.max_speed = max_speed
		self.brown_damping = brown_damping
		self.brown_velocity = brown_velocity
		self.init_velocity = init_velocity
		self.dt = dt
		self.types = types
		self.images = images
		self.env_info = [{} for _ in range(dataset_size)]
		
		self.padding_x = 5
		self.padding_y = 4
		self.ecmo_padding_y = 6
		
		self.v_cond = torch.zeros(dataset_size,2,w,h)
		self.v_mask = torch.zeros(dataset_size,1,w,h)
		self.v_cond_full_res = torch.zeros(dataset_size,2,self.w_full_res,self.h_full_res)
		self.v_mask_full_res = torch.zeros(dataset_size,1,self.w_full_res,self.h_full_res)
		
		self.hidden_states = torch.zeros(dataset_size,hidden_size,w-1,h-1)#hidden state is 1 smaller than dataset-size!
		self.t = 0
		self.i = 0
		
		self.mousex = 0
		self.mousey = 0
		self.mousev = 0
		self.mousew = 0
		self.mouse_paint = False
		self.mouse_radius = 3
		self.mouse_erase = False
		
		for i in range(dataset_size):
			self.reset_env(i)
	
	def reset_env(self,index):
		#print(f"reset env {index}")
		self.hidden_states[index] = 0
		self.v_cond_full_res[index] = 0
		self.v_mask_full_res[index] = 0
		
		self.v_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
		self.v_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
		self.v_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
		self.v_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
		
		type = np.random.choice(self.types)
		self.env_info[index]["type"] = type
		
		if type=="simple":
			# simple obstacle
			self.v_mask_full_res[index,:,(self.w_full_res//2-5*self.resolution_factor):(self.w_full_res//2+5*self.resolution_factor),(self.h_full_res//2-5*self.resolution_factor):(self.h_full_res//2+5*self.resolution_factor)] = 1
			
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(10*self.resolution_factor):-(10*self.resolution_factor)] = 1
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(10*self.resolution_factor):-(10*self.resolution_factor)] = 1
		
		if type == "box":# block at random position
			#object_h = 5+15*np.random.rand() # object height / 2
			#object_w = 5+15*np.random.rand() # object width / 2
			object_w = 5+((self.w/2-self.padding_x)/2)*np.random.rand() # object width / 2
			object_h = 5+((self.h/2-self.padding_y)/2)*np.random.rand() # object height / 2
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			self.v_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.v_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = object_vx
			self.v_cond_full_res[index,1,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = object_vy
			
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
		
		if type == "magnus": # magnus effekt
			flow_v = self.max_speed*(np.random.rand()-0.5)*2 #flow velocity
			#object_r = 5+15*np.random.rand() # object radius
			object_r = 5+(min(self.w/2-self.padding_x,self.h/2-self.padding_y)/2)*np.random.rand() # object radius
			
			object_y = np.random.randint(self.padding_y+object_r+2,self.h-self.padding_y-object_r-2)
			object_x = np.random.randint(self.padding_x+object_r+2,self.w-self.padding_x-object_r-2)
			
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			object_w = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = torch.meshgrid([torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1),torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1)])
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<(self.resolution_factor*object_r)**2).float().unsqueeze(0)
			
			# 3. generate v_cond and multiply with mask
			v_ball = object_w/self.resolution_factor*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / v_cond
			x_pos1, y_pos1 = int(self.resolution_factor*(object_x-object_r)),int(self.resolution_factor*(object_y-object_r))
			x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
			self.v_mask_full_res[index,:,x_pos1:x_pos2,y_pos1:y_pos2] += mask_ball
			self.v_cond_full_res[index,0,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[0]+object_vx
			self.v_cond_full_res[index,1,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[1]+object_vy
			self.v_cond_full_res[index] = self.v_cond_full_res[index]*self.v_mask_full_res[index]
			
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w*object_r # soll das wirklich so sein?
		
		if type == "DFG_benchmark" or type == "poiseuille" or type == "paint": # DFG benchmark problem as specified in http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html
			flow_v = self.max_speed*(np.random.rand()-0.5)*2 #flow velocity
			object_r = 0.05/0.41*(self.h-2*self.padding_y) # object radius
			
			object_y = 0.2/0.41*(self.h-2*self.padding_y)+self.padding_y
			object_x = 0.2/0.41*(self.h-2*self.padding_y)+self.padding_x
			
			object_vx,object_vy,object_w = 0,0,0 # object angular velocity
			
			if type == "DFG_benchmark":
				# 1. generate mesh 2 x [2r x 2r]
				y_mesh,x_mesh = torch.meshgrid([torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1),torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1)])
				
				# 2. generate mask
				mask_ball = ((x_mesh**2+y_mesh**2)<(self.resolution_factor*object_r)**2).float().unsqueeze(0)
				
				# 3. generate v_cond and multiply with mask
				v_ball = object_w/self.resolution_factor*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
				
				# 4. add masks / v_cond
				x_pos1, y_pos1 = int(self.resolution_factor*(object_x-object_r)),int(self.resolution_factor*(object_y-object_r))
				x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
				self.v_mask_full_res[index,:,x_pos1:x_pos2,y_pos1:y_pos2] += mask_ball
				self.v_cond_full_res[index,0,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[0]+object_vx
				self.v_cond_full_res[index,1,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[1]+object_vy
				self.v_cond_full_res[index] = self.v_cond_full_res[index]*self.v_mask_full_res[index]
				
			# inlet / outlet flow
			profile_size = self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)].shape[1]
			flow_profile = torch.arange(0,profile_size,1.0)
			flow_profile *= 0.41/flow_profile[-1]
			flow_profile = 4*flow_profile*(0.41-flow_profile)/0.1681
			flow_profile = flow_profile.unsqueeze(0)
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = flow_v*flow_profile
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = flow_v*flow_profile
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w*object_r # soll das wirklich so sein?
			
		if type == "ecmo": # extrakorporale membranoxygenierung
			flow_v = self.max_speed*(np.random.rand()-0.5)*2 #flow velocity
			object_w = -self.max_speed*(np.random.rand()) #flow velocity (CODO: allow negative velocities -> could be interesting as well...)
			
			object_r = 5+(min(self.w/2-self.padding_x,self.h/2-self.padding_y)/2)*np.random.rand()*0.3 # width of tube
			
			#object_y = np.random.randint(self.padding_y+object_r+2,self.h-self.padding_y-object_r-2)
			object_y = np.random.randint(self.ecmo_padding_y+object_r+2,self.h-self.ecmo_padding_y-object_r-2)
			object_x = np.random.randint(self.padding_x+object_r+2,self.w-self.padding_x-object_r-2)
			
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			in_flow = (self.h_full_res-2*self.ecmo_padding_y*self.resolution_factor)*flow_v
			ecmo_flow = (int(self.resolution_factor*(object_y+object_r-1))-int(self.resolution_factor*(object_y-object_r+1)))*object_w
			out_flow = (in_flow-ecmo_flow)/(self.h_full_res-2*self.ecmo_padding_y*self.resolution_factor-int(self.resolution_factor*(object_y+object_r))+int(self.resolution_factor*(object_y-object_r)))
			
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(self.ecmo_padding_y*self.resolution_factor):-(self.ecmo_padding_y*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(self.ecmo_padding_y*self.resolution_factor):-(self.ecmo_padding_y*self.resolution_factor)] = out_flow
			
			self.v_mask_full_res[index,:,int(object_x*self.resolution_factor):,int(self.resolution_factor*(object_y-object_r)):int(self.resolution_factor*(object_y+object_r))] = 1
			self.v_cond_full_res[index,:,int(object_x*self.resolution_factor):,int(self.resolution_factor*(object_y-object_r)):int(self.resolution_factor*(object_y+object_r))] = 0
			self.v_cond_full_res[index,0,int(object_x*self.resolution_factor):,int(self.resolution_factor*(object_y-object_r+1)):int(self.resolution_factor*(object_y+object_r-1))] = object_w
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w # flow velocity of ecmo device
		
		if type == "image":
			image = np.random.choice(self.images)
			image_mask = images[image]
			object_h, object_w = image_mask.shape[0], image_mask.shape[1]
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			
			
			object_y = np.random.randint(object_w/2/self.resolution_factor + self.padding_y + 1,self.h - object_w/2/self.resolution_factor - self.padding_y - 1)
			object_x = np.random.randint(object_h/2/self.resolution_factor + self.padding_x + 1,self.w - object_h/2/self.resolution_factor - self.padding_x - 1)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			
			
			self.v_mask_full_res[index,:,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)] = 1-(1-self.v_mask_full_res[index,:,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)])*(1-image_mask)
			self.v_cond_full_res[index,:]=0
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0:1,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)] = object_vx*image_mask
			self.v_cond_full_res[index,1:2,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)] = object_vy*image_mask
			
			self.env_info[index]["type"] = type
			self.env_info[index]["image"] = image
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
		
		self.v_cond[index:(index+1)] = f.avg_pool2d(self.v_cond_full_res[index:(index+1)],self.resolution_factor)
		self.v_mask[index:(index+1)] = f.avg_pool2d(self.v_mask_full_res[index:(index+1)],self.resolution_factor)
	
	def update_env(self,index):
		#CODO: introduce "layers" in between time-steps (e.g. for dt/2)
		
		if self.env_info[index]["type"] == "box":
			object_h = self.env_info[index]["h"]
			object_w = self.env_info[index]["w"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_w + self.padding_x + 1:
					object_x = object_w + self.padding_x + 1
					object_vx = -object_vx
				if object_x > self.w - object_w - self.padding_x - 1:
					object_x = self.w - object_w - self.padding_x - 1
					object_vx = -object_vx
					
				if object_y < object_h + self.padding_y + 1:
					object_y = object_h + self.padding_y + 1
					object_vy = -object_vy
				if object_y > self.h - object_h - self.padding_y - 1:
					object_y = self.h - object_h - self.padding_y - 1
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_w + self.padding_x + 1:
					object_x = object_w + self.padding_x + 1
					object_vx = 0
				if object_x > self.w - object_w - self.padding_x - 1:
					object_x = self.w - object_w - self.padding_x - 1
					object_vx = 0
					
				if object_y < object_h + self.padding_y + 1:
					object_y = object_h + self.padding_y + 1
					object_vy = 0
				if object_y > self.h - object_h - self.padding_y - 1:
					object_y = self.h - object_h - self.padding_y - 1
					object_vy = 0
			
			self.v_mask_full_res[index] = 0
			self.v_cond_full_res[index] = 0
			
			self.v_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.v_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.v_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.v_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			self.v_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.v_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = object_vx
			self.v_cond_full_res[index,1,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = object_vy
			
			self.v_cond_full_res[index] = self.v_cond_full_res[index]*self.v_mask_full_res[index]
			
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["flow_v"] = flow_v
		
		if self.env_info[index]["type"] == "magnus":
			object_r = self.env_info[index]["r"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_w = self.env_info[index]["w"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = -object_vx
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = -object_vx
					
				if object_y < object_r + self.padding_y + 1:
					object_y = object_r + self.padding_y + 1
					object_vy = -object_vy
				if object_y > self.h - object_r - self.padding_y - 1:
					object_y = self.h - object_r - self.padding_y - 1
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_w = self.mousew/object_r
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = 0
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = 0
					
				if object_y < object_r + self.padding_y + 1:
					object_y = object_r + self.padding_y + 1
					object_vy = 0
				if object_y > self.h - object_r - self.padding_y - 1:
					object_y = self.h - object_r - self.padding_y - 1
					object_vy = 0
			
			self.v_mask_full_res[index] = 0
			self.v_cond_full_res[index] = 0
			
			self.v_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.v_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.v_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.v_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = torch.meshgrid([torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1),torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1)])
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<(self.resolution_factor*object_r)**2).float().unsqueeze(0)
			
			# 3. generate v_cond and multiply with mask
			v_ball = object_w/self.resolution_factor*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / v_cond
			x_pos1, y_pos1 = int(self.resolution_factor*(object_x-object_r)),int(self.resolution_factor*(object_y-object_r))
			x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
			self.v_mask_full_res[index,:,x_pos1:x_pos2,y_pos1:y_pos2] += mask_ball
			self.v_cond_full_res[index,0,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[0]+object_vx
			self.v_cond_full_res[index,1,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[1]+object_vy
			self.v_cond_full_res[index] = self.v_cond_full_res[index]*self.v_mask_full_res[index]
			
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
		
		if self.env_info[index]["type"] == "DFG_benchmark" or self.env_info[index]["type"] == "poiseuille" or self.env_info[index]["type"] == "paint":
			object_r = self.env_info[index]["r"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_w = self.env_info[index]["w"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = -object_vx
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = -object_vx
					
				if object_y < object_r + self.padding_y + 1:
					object_y = object_r + self.padding_y + 1
					object_vy = -object_vy
				if object_y > self.h - object_r - self.padding_y - 1:
					object_y = self.h - object_r - self.padding_y - 1
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_w = self.mousew/object_r
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = 0
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = 0
					
				if object_y < object_r + self.padding_y + 1:
					object_y = object_r + self.padding_y + 1
					object_vy = 0
				if object_y > self.h - object_r - self.padding_y - 1:
					object_y = self.h - object_r - self.padding_y - 1
					object_vy = 0
			
			if not self.env_info[index]["type"] == "paint":
				self.v_mask_full_res[index] = 0
				self.v_cond_full_res[index] = 0
				
			
			if self.env_info[index]["type"] == "paint":
				if self.mouse_paint:
					x_pos1, y_pos1 = int(self.resolution_factor*(self.mousex-self.mouse_radius)),int(self.resolution_factor*(self.mousey-self.mouse_radius))
					x_pos2, y_pos2 = x_pos1+2*self.resolution_factor*self.mouse_radius,y_pos1+2*self.resolution_factor*self.mouse_radius
					self.v_mask_full_res[index,:,x_pos1:x_pos2,y_pos1:y_pos2] = 1
				if self.mouse_erase:
					x_pos1, y_pos1 = int(self.resolution_factor*(self.mousex-self.mouse_radius)),int(self.resolution_factor*(self.mousey-self.mouse_radius))
					x_pos2, y_pos2 = x_pos1+2*self.resolution_factor*self.mouse_radius,y_pos1+2*self.resolution_factor*self.mouse_radius
					self.v_mask_full_res[index,:,x_pos1:x_pos2,y_pos1:y_pos2] = 0
			
			self.v_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.v_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.v_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.v_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
		
			if self.env_info[index]["type"] == "DFG_benchmark":
				# 1. generate mesh 2 x [2r x 2r]
				y_mesh,x_mesh = torch.meshgrid([torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1),torch.arange(-int(self.resolution_factor*object_r),int(self.resolution_factor*object_r)+1)])
				
				# 2. generate mask
				mask_ball = ((x_mesh**2+y_mesh**2)<(self.resolution_factor*object_r)**2).float().unsqueeze(0)
				
				# 3. generate v_cond and multiply with mask
				v_ball = object_w/self.resolution_factor*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
				
				# 4. add masks / v_cond
				x_pos1, y_pos1 = int(self.resolution_factor*(object_x-object_r)),int(self.resolution_factor*(object_y-object_r))
				x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
				self.v_mask_full_res[index,:,x_pos1:x_pos2,y_pos1:y_pos2] += mask_ball
				self.v_cond_full_res[index,0,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[0]+object_vx
				self.v_cond_full_res[index,1,x_pos1:x_pos2,y_pos1:y_pos2] += v_ball[1]+object_vy
				self.v_cond_full_res[index] = self.v_cond_full_res[index]*self.v_mask_full_res[index]
				
			# inlet / outlet flow
			profile_size = self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)].shape[1]
			flow_profile = torch.arange(0,profile_size,1.0)
			flow_profile *= 0.41/flow_profile[-1]
			flow_profile = 4*flow_profile*(0.41-flow_profile)/0.1681
			flow_profile = flow_profile.unsqueeze(0)
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = flow_v*flow_profile
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = flow_v*flow_profile
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			
		if self.env_info[index]["type"] == "ecmo":
			object_r = self.env_info[index]["r"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_w = self.env_info[index]["w"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = -object_vx
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = -object_vx
					
				if object_y < object_r + self.ecmo_padding_y + 1:
					object_y = object_r + self.ecmo_padding_y + 1
					object_vy = -object_vy
				if object_y > self.h - object_r - self.ecmo_padding_y - 1:
					object_y = self.h - object_r - self.ecmo_padding_y - 1
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_w = self.mousew
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = 0
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = 0
					
				if object_y < object_r + self.ecmo_padding_y + 1:
					object_y = object_r + self.ecmo_padding_y + 1
					object_vy = 0
				if object_y > self.h - object_r - self.ecmo_padding_y - 1:
					object_y = self.h - object_r - self.ecmo_padding_y - 1
					object_vy = 0
			
			self.v_mask_full_res[index] = 0
			self.v_cond_full_res[index] = 0
			
			self.v_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.v_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.v_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.v_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			
			in_flow = (self.h_full_res-2*self.ecmo_padding_y*self.resolution_factor)*flow_v
			ecmo_flow = (int(self.resolution_factor*(object_y+object_r-1))-int(self.resolution_factor*(object_y-object_r+1)))*object_w+object_vx*(int(self.resolution_factor*(object_y+object_r))-int(self.resolution_factor*(object_y-object_r)))
			out_flow = (in_flow-ecmo_flow)/(self.h_full_res-2*self.ecmo_padding_y*self.resolution_factor-int(self.resolution_factor*(object_y+object_r))+int(self.resolution_factor*(object_y-object_r)))
			
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(self.ecmo_padding_y*self.resolution_factor):-(self.ecmo_padding_y*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(self.ecmo_padding_y*self.resolution_factor):-(self.ecmo_padding_y*self.resolution_factor)] = out_flow
			
			self.v_mask_full_res[index,:,int(object_x*self.resolution_factor):,int(self.resolution_factor*(object_y-object_r)):int(self.resolution_factor*(object_y+object_r))] = 1
			self.v_cond_full_res[index,0,int(object_x*self.resolution_factor):,int(self.resolution_factor*(object_y-object_r)):int(self.resolution_factor*(object_y+object_r))] = object_vx
			self.v_cond_full_res[index,1,int(object_x*self.resolution_factor):,int(self.resolution_factor*(object_y-object_r)):int(self.resolution_factor*(object_y+object_r))] = object_vy
			self.v_cond_full_res[index,0,int(object_x*self.resolution_factor):,int(self.resolution_factor*(object_y-object_r+1)):int(self.resolution_factor*(object_y+object_r-1))] = object_w+object_vx
			
			self.v_cond_full_res[index] = self.v_cond_full_res[index]*self.v_mask_full_res[index]
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["flow_v"] = flow_v
		
		if self.env_info[index]["type"] == "image":
			object_h = self.env_info[index]["h"]
			object_w = self.env_info[index]["w"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_h/2/self.resolution_factor + self.padding_x + 1:
					object_x = object_h/2/self.resolution_factor + self.padding_x + 1
					object_vx = -object_vx
				if object_x > self.w - object_h/2/self.resolution_factor - self.padding_x - 1:
					object_x = self.w - object_h/2/self.resolution_factor - self.padding_x - 1
					object_vx = -object_vx
					
				if object_y < object_w/2/self.resolution_factor + self.padding_y + 1:
					object_y = object_w/2/self.resolution_factor + self.padding_y + 1
					object_vy = -object_vy
				if object_y > self.h - object_w/2/self.resolution_factor - self.padding_y - 1:
					object_y = self.h - object_w/2/self.resolution_factor - self.padding_y - 1
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_h/2/self.resolution_factor + self.padding_x + 1:
					object_x = object_h/2/self.resolution_factor + self.padding_x + 1
					object_vx = 0
				if object_x > self.w - object_h/2/self.resolution_factor - self.padding_x - 1:
					object_x = self.w - object_h/2/self.resolution_factor - self.padding_x - 1
					object_vx = 0
					
				if object_y < object_w/2/self.resolution_factor + self.padding_y + 1:
					object_y = object_w/2/self.resolution_factor + self.padding_y + 1
					object_vy = 0
				if object_y > self.h - object_w/2/self.resolution_factor - self.padding_y - 1:
					object_y = self.h - object_w/2/self.resolution_factor - self.padding_y - 1
					object_vy = 0
			
			self.v_mask_full_res[index] = 0
			self.v_cond_full_res[index] = 0
			
			self.v_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.v_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.v_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.v_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			image = self.env_info[index]["image"]
			image_mask = images[image]
			
			self.v_mask_full_res[index,:,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)] = 1-(1-self.v_mask_full_res[index,:,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)])*(1-image_mask)
			self.v_cond_full_res[index,:]=0
			self.v_cond_full_res[index,0,:(self.padding_x*self.resolution_factor),(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0,-(self.padding_x*self.resolution_factor):,(10*self.resolution_factor):-(10*self.resolution_factor)] = flow_v
			self.v_cond_full_res[index,0:1,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)] = object_vx*image_mask
			self.v_cond_full_res[index,1:2,int(self.resolution_factor*object_x-object_h//2):int(self.resolution_factor*object_x-object_h//2+object_h),int(self.resolution_factor*object_y-object_w//2):int(self.resolution_factor*object_y-object_w//2+object_w)] = object_vy*image_mask
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["flow_v"] = flow_v
		
		self.v_cond[index:(index+1)] = f.avg_pool2d(self.v_cond_full_res[index:(index+1)],self.resolution_factor)
		self.v_mask[index:(index+1)] = f.avg_pool2d(self.v_mask_full_res[index:(index+1)],self.resolution_factor)
	
	def update_envs(self,indices):
		for index in indices:
			self.update_env(index)
	
	def ask(self):
		"""
		:return:
			grids:
				boundary-features:
					v_cond						-> shape: bs x 2 x w x h
					v_mask (continuous) 		-> shape: bs x 1 x w x h differentiable renderer would allow for differentiable geometries
				hidden_state					-> shape: bs x hidden_size x (w-1) x (h-1)
			sample-grids:
				- grid-offsets (x,y,t) 			-> shape: bs x 3 x 1 x 1 (values between 0,1; all offsets are the same within an "image" - otherwise: bsx3xwxh)
				- sample_v_cond					-> shape: bs x 2 x w x h
				- sample_v_mask (boolean)		-> shape: bs x 1 x w x h
		"""
		
		if self.interactive:
			self.mousev = min(max(self.mousev,-self.max_speed),self.max_speed)
			self.mousew = min(max(self.mousew,-self.max_speed),self.max_speed)
		
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		self.update_envs(self.indices)
		grid_offsets = []
		sample_v_cond = []
		sample_v_mask = []
		for i in range(self.n_samples):
			offset = torch.rand(3)
			grid_offsets.append(offset)
			x_offset = min(int(self.resolution_factor*offset[0]),self.resolution_factor-1)
			y_offset = min(int(self.resolution_factor*offset[1]),self.resolution_factor-1)
			sample_v_cond.append(self.v_cond_full_res[self.indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
			sample_v_mask.append(self.v_mask_full_res[self.indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
		
		
		return self.v_cond[self.indices],self.v_mask[self.indices],self.hidden_states[self.indices],grid_offsets,sample_v_cond,sample_v_mask
	
	def tell(self,hidden_state):
		
		self.hidden_states[self.indices,:,:,:] = hidden_state.detach()
	
		self.t += 1
		if self.t % int(self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
			self.reset_env(int(self.i))
			self.i = (self.i+1)%self.dataset_size
