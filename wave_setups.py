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

class Dataset():
	
	def generate_random_z_cond(self,time): # modulate x/y/t with random sin / cos waves => used in "simple" environment
		
		return torch.sin(self.x_mesh*0.02+torch.cos(self.y_mesh*0.01*(np.cos(time*0.0021)+2))*np.cos(time*0.01)*3+torch.cos(self.x_mesh*0.011*(np.sin(time*0.00221)+2))*np.cos(time*0.00321)*3+0.01*self.y_mesh*np.cos(time*0.0215))
	
	def __init__(self,w,h,hidden_size,resolution_factor=4,batch_size=100,n_samples=1,dataset_size=1000,average_sequence_length=5000,interactive=False,max_speed=10,brown_damping=0.9995,brown_velocity=0.05,init_velocity=0,dt=1,types=["simple"]):
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
		self.interactive_spring = 5#300#200#~ 1/spring constant to move object
		self.max_speed = max_speed
		self.brown_damping = brown_damping
		self.brown_velocity = brown_velocity
		self.init_velocity = init_velocity
		self.dt = dt
		self.types = types
		self.env_info = [{} for _ in range(dataset_size)]
		
		self.x_mesh,self.y_mesh = torch.meshgrid([torch.arange(0,self.w_full_res),torch.arange(0,self.h_full_res)])
		self.x_mesh,self.y_mesh = 1.0*self.x_mesh,1.0*self.y_mesh
		
		self.padding_x = 4
		self.padding_y = 4
		
		self.z_cond = torch.zeros(dataset_size,1,w,h)
		self.z_mask = torch.zeros(dataset_size,1,w,h)
		self.z_cond_full_res = torch.zeros(dataset_size,1,self.w_full_res,self.h_full_res)
		self.z_mask_full_res = torch.zeros(dataset_size,1,self.w_full_res,self.h_full_res)
		
		self.hidden_states = torch.zeros(dataset_size,hidden_size,w-1,h-1)#hidden state is 1 smaller than dataset-size!
		self.t = 0
		self.i = 0
		
		self.mousex = 0
		self.mousey = 0
		self.mousev = 0
		self.mousew = 0
		
		for i in range(dataset_size):
			self.reset_env(i)
	
	def reset_env(self,index):
		#print(f"reset env {index}")
		self.hidden_states[index] = 0
		self.z_cond_full_res[index] = 0
		self.z_mask_full_res[index] = 1
		self.z_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):(self.padding_x*self.resolution_factor),-(self.padding_y*self.resolution_factor):(self.padding_y*self.resolution_factor)] = 0
		
		type = np.random.choice(self.types)
		self.env_info[index]["type"] = type
		
		if type=="super_simple":
			# frame
			self.z_mask_full_res[index] = 1
			self.z_mask_full_res[index,:,(self.padding_x*self.resolution_factor):-(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = 0
			
			# obstabcles (oscillators)
			for x in [0]:
				self.z_mask_full_res[index,:,(self.w_full_res//2+(-5+x)*self.resolution_factor):(self.w_full_res//2+(5+x)*self.resolution_factor),(self.h_full_res//2-5*self.resolution_factor):(self.h_full_res//2+5*self.resolution_factor)] = 1
			
			self.z_cond_full_res[index,0,(self.padding_x*self.resolution_factor):-(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = 1
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
		
		
		if type=="simple":
			# simple obstacle
			
			self.env_info[index]["seed"] = 1000*torch.rand(1)
			self.z_mask_full_res[index] = 0
			self.z_mask_full_res[index] = (self.generate_random_z_cond(self.env_info[index]["seed"])>0.6).float()
			self.z_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.z_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.z_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.z_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			self.z_cond_full_res[index,0,:,:] = self.generate_random_z_cond(self.env_info[index]["seed"])
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
			self.env_info[index]["time"] = 0
		
		if type=="oscillator":
			self.env_info[index]["seed"] = 1000*torch.rand(1)
			# frame
			self.z_mask_full_res[index] = 1
			self.z_mask_full_res[index,:,(self.padding_x*self.resolution_factor):-(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = 0
			
			# obstabcles (oscillators)
			for x in [0]:#[-45,-15,15,45]:#[-40,-20,0,20,40]:# [-30,0,30]:
				for y in [-45,-15,15,45]:#[0]:
					self.z_mask_full_res[index,:,(self.w_full_res//2+(-5+x)*self.resolution_factor):(self.w_full_res//2+(5+x)*self.resolution_factor),(self.h_full_res//2+(-5+y)*self.resolution_factor):(self.h_full_res//2+(5+y)*self.resolution_factor)] = 1
			
			self.z_cond_full_res[index,0,(self.padding_x*self.resolution_factor):-(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = np.sin(self.env_info[index]["seed"])
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
			self.env_info[index]["time"] = 0
		
		if type == "box": # block at random position
			self.env_info[index]["phase"] = torch.rand(1)*2*np.pi
			object_w = 5+((self.w/2-self.padding_x)/2)*np.random.rand() # object width / 2
			object_h = 5+((self.h/2-self.padding_y)/2)*np.random.rand() # object height / 2
			rand = np.random.rand(1)
			f_max,f_min = 2,0.1 # could be properly parameterized
			freq = 1/np.exp(rand*np.log(1/f_max)+(1-rand)*np.log(1/f_min)) # <- we want high frequencies to appear more often than low frequencies
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			object_x = np.random.randint(self.w//2-10,self.w//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			self.z_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.z_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = np.sin(self.env_info[index]["phase"])
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["time"] = 0
			self.env_info[index]["freq"] = freq
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = freq
		
		if type == "doppler":# constant moving block (for results)
			self.env_info[index]["phase"] = 0
			object_w = 10
			object_h = 10
			rand = np.random.rand(1)
			freq = 1
			object_y = self.h//2
			object_x = self.w//2
			object_vx = 0
			object_vy = 0.6
			
			self.z_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.z_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = np.sin(self.env_info[index]["phase"])
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["time"] = 0
			self.env_info[index]["freq"] = freq
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = freq
		
		if type == "reflection":# reflection (for results)
			self.env_info[index]["phase"] = 0
			object_w = 10
			object_h = 10
			rand = np.random.rand(1)
			freq = 1
			object_y = self.h//4*3
			object_x = self.w//2
			object_vx = 0
			object_vy = 0
			
			self.z_mask_full_res[index,:,:(self.w//2*self.resolution_factor),((self.h//2-2)*self.resolution_factor):((self.h//2+2)*self.resolution_factor)] = 1
			self.z_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.z_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = np.sin(self.env_info[index]["phase"])
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["time"] = 0
			self.env_info[index]["freq"] = freq
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = freq
		
		self.z_cond[index:(index+1)] = f.avg_pool2d(self.z_cond_full_res[index:(index+1)],self.resolution_factor)
		self.z_mask[index:(index+1)] = f.avg_pool2d(self.z_mask_full_res[index:(index+1)],self.resolution_factor)
	
	def update_env(self,index):
		#CODO: introduce "layers" in between time-steps (e.g. for dt/2)
		
		if self.env_info[index]["type"] == "simple":
			time = self.env_info[index]["time"]
			
			self.z_mask_full_res[index] = 0
			self.z_mask_full_res[index] = (self.generate_random_z_cond(-time+self.env_info[index]["seed"])>0.6).float()
			self.z_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.z_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.z_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.z_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			self.z_cond_full_res[index,0,:,:] = self.generate_random_z_cond(time+self.env_info[index]["seed"])
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
			self.env_info[index]["time"] = time + 1
			
		if self.env_info[index]["type"] == "oscillator":
			time = self.env_info[index]["time"]
			
			self.z_cond_full_res[index,0,(self.padding_x*self.resolution_factor):-(self.padding_x*self.resolution_factor),(self.padding_y*self.resolution_factor):-(self.padding_y*self.resolution_factor)] = np.sin(time*1+self.env_info[index]["seed"])
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
			self.env_info[index]["time"] = time + 1
			
		if self.env_info[index]["type"] == "box":
			time = self.env_info[index]["time"]
			object_h = self.env_info[index]["h"]
			object_w = self.env_info[index]["w"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			self.env_info[index]["phase"] += self.env_info[index]["freq"]
			
			if not self.interactive:
				freq = self.env_info[index]["freq"]
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
				freq = self.mousev
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
			
			self.z_mask_full_res[index] = 0
			self.z_cond_full_res[index] = 0
			
			self.z_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.z_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.z_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.z_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			self.z_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.z_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = np.sin(self.env_info[index]["phase"])
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["freq"] = freq
			self.env_info[index]["time"] = time + 1
		
		if self.env_info[index]["type"] == "doppler":
			time = self.env_info[index]["time"]
			object_h = self.env_info[index]["h"]
			object_w = self.env_info[index]["w"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			self.env_info[index]["phase"] += self.env_info[index]["freq"]
			
			freq = self.env_info[index]["freq"]
			object_vx = vx_old
			object_vy = vy_old
			
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
			
			self.z_mask_full_res[index] = 0
			self.z_cond_full_res[index] = 0
			
			self.z_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.z_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.z_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.z_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			
			self.z_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.z_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = np.sin(self.env_info[index]["phase"])
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["freq"] = freq
			self.env_info[index]["time"] = time + 1
		
		if self.env_info[index]["type"] == "reflection":
			time = self.env_info[index]["time"]
			object_h = self.env_info[index]["h"]
			object_w = self.env_info[index]["w"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			self.env_info[index]["phase"] += self.env_info[index]["freq"]
			
			freq = self.env_info[index]["freq"]
			object_vx = vx_old
			object_vy = vy_old
			
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
			
			self.z_mask_full_res[index] = 0
			self.z_cond_full_res[index] = 0
			
			self.z_mask_full_res[index,:,:(self.padding_x*self.resolution_factor),:] = 1
			self.z_mask_full_res[index,:,-(self.padding_x*self.resolution_factor):,:] = 1
			self.z_mask_full_res[index,:,:,:(self.padding_y*self.resolution_factor)] = 1
			self.z_mask_full_res[index,:,:,-(self.padding_y*self.resolution_factor):] = 1
			self.z_mask_full_res[index,:,:(self.w//2*self.resolution_factor),((self.h//2-2)*self.resolution_factor):((self.h//2+2)*self.resolution_factor)] = 1
			
			self.z_mask_full_res[index,:,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = 1
			self.z_cond_full_res[index,0,int(self.resolution_factor*(object_x-object_w)):int(self.resolution_factor*(object_x+object_w)),int(self.resolution_factor*(object_y-object_h)):int(self.resolution_factor*(object_y+object_h))] = np.sin(self.env_info[index]["phase"])
			self.z_cond_full_res[index] = self.z_cond_full_res[index]*self.z_mask_full_res[index]
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["freq"] = freq
			self.env_info[index]["time"] = time + 1
		
		self.z_cond[index:(index+1)] = f.avg_pool2d(self.z_cond_full_res[index:(index+1)],self.resolution_factor)
		self.z_mask[index:(index+1)] = f.avg_pool2d(self.z_mask_full_res[index:(index+1)],self.resolution_factor)
	
	def update_envs(self,indices):
		for index in indices:
			self.update_env(index)
	
	def ask(self):
		"""
		:return:
			grids:
				boundary-features:
					z_cond						-> shape: bs x 2 x w x h
					z_mask (continuous) 		-> shape: bs x 1 x w x h differentiable renderer would allow for differentiable geometries
				hidden_state					-> shape: bs x hidden_size x (w-1) x (h-1)
			sample-grids:
				- grid-offsets (x,y,t) 			-> shape: bs x 3 x 1 x 1 (values between 0,1; all offsets are the same within an "image" - otherwise: bsx3xwxh)
				- sample_z_cond					-> shape: bs x 2 x w x h
				- sample_z_mask (boolean)		-> shape: bs x 1 x w x h
		"""
		
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		self.update_envs(self.indices)
		"""
		grid_offsets = torch.rand(self.batch_size,3,1,1) # atm: ignore temporal offset, as we are only looking at steady environments
		sample_z_cond = torch.zeros(self.batch_size,2,self.w,self.h)
		sample_z_mask = torch.zeros(self.batch_size,1,self.w,self.h)
		for i,index in enumerate(self.indices):
			x_offset = min(int(self.resolution_factor*grid_offsets[i,0]),self.resolution_factor-1)
			y_offset = min(int(self.resolution_factor*grid_offsets[i,1]),self.resolution_factor-1)
			sample_z_cond[i] = self.z_cond_full_res[index,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor]
			sample_z_mask[i] = self.z_mask_full_res[index,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor]
		"""
		grid_offsets = []
		sample_z_cond = []
		sample_z_mask = []
		for i in range(self.n_samples):
			offset = torch.rand(3)
			grid_offsets.append(offset)
			x_offset = min(int(self.resolution_factor*offset[0]),self.resolution_factor-1)
			y_offset = min(int(self.resolution_factor*offset[1]),self.resolution_factor-1)
			sample_z_cond.append(self.z_cond_full_res[self.indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
			sample_z_mask.append(self.z_mask_full_res[self.indices,:,x_offset::self.resolution_factor,y_offset::self.resolution_factor])
		
		
		return self.z_cond[self.indices],self.z_mask[self.indices],self.hidden_states[self.indices],grid_offsets,sample_z_cond,sample_z_mask
	
	def tell(self,hidden_state):
		
		#CODO: do not update the state at every step...
		self.hidden_states[self.indices,:,:,:] = hidden_state.detach()
	
		self.t += 1
		#print(f"t: {self.t} - {(self.average_sequence_length/self.batch_size)}")
		if self.t % int(self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
			self.reset_env(int(self.i))
			self.i = (self.i+1)%self.dataset_size
