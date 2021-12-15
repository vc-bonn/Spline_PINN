import argparse

def str2bool(v):
	"""
	'type variable' for add_argument
	"""
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		raise argparse.ArgumentTypeError('boolean value expected.')

def params():
	"""
	return parameters for training / testing / plotting of models
	:return: parameter-Namespace
	"""
	parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')
	
	# Training parameters
	parser.add_argument('--n_epochs', default=100000, type=int, help='number of epochs (after each epoch, the model gets saved)')
	parser.add_argument('--n_batches_per_epoch', default=10000, type=int, help='number of batches per epoch (default: 10000)')
	parser.add_argument('--batch_size', default=50, type=int, help='batch size (default: 30)')
	parser.add_argument('--n_samples', default=10, type=int, help='number of samples (different offsets) per batch (default: 10)')
	parser.add_argument('--dataset_size', default=1000, type=int, help='size of dataset (default: 1000)')
	parser.add_argument('--average_sequence_length', default=5000, type=int, help='average sequence length in dataset (default: 5000)')
	parser.add_argument('--resolution_factor', default=8, type=int, help='resolution factor for superres / kernels (default: 8)')
	parser.add_argument('--loss_bound', default=20, type=float, help='loss factor for boundary conditions')
	parser.add_argument('--loss_bound_reg', default=0, type=float, help='loss factor to regularize acceleration on boundaries for wave equation (default:0)')
	parser.add_argument('--loss_domain_res', default=10, type=float, help='loss factor for residual loss in fluid domain')
	parser.add_argument('--loss_domain_up', default=0, type=float, help='loss factor for "upwind loss" in fluid domain')
	parser.add_argument('--loss_domain_p', default=0, type=float, help='loss factor for pressure in fluid domain')
	parser.add_argument('--loss_wave', default=1, type=float, help='loss factor for wave equation')
	parser.add_argument('--loss_v', default=100, type=float, help='loss factor to connect dz_dt and v')
	parser.add_argument('--loss_mean_a', default=0, type=float, help='loss factor to make mean of a small inside fluid domain')
	parser.add_argument('--loss_mean_p', default=0, type=float, help='loss factor to make mean of p small inside fluid domain')
	parser.add_argument('--border_weight', default=0, type=float, help='extra weight on fluid domain borders')
	parser.add_argument('--lr', default=0.0001, type=float, help='learning rate of ADAM-optimizer (default: 0.0001)')
	parser.add_argument('--clip_grad_norm', default=None, type=float, help='gradient norm clipping (default: None)')
	parser.add_argument('--clip_grad_value', default=None, type=float, help='gradient value clipping (default: None)')
	parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')
	parser.add_argument('--detach', default=False, type=str2bool, help='detach gradients in between steps (for train_wave_3)')
	parser.add_argument('--log_loss', default=True, type=str2bool, help='logarithmic loss to "normalize" gradients')
	
	# Network parameters
	parser.add_argument('--net', default="Shortcut", type=str, help='network to train', choices=["Shortcut","Shortcut2","Shortcut2_residual","Shortcut4","Shortcut4_residual","Shortcut3","Fluid_model","Wave_model"])
	parser.add_argument('--hidden_size', default=20, type=int, help='hidden size of network (default: 20)')
	parser.add_argument('--orders_v', default=2, type=int, help='spline order for velocity potential field (default: 2)')
	parser.add_argument('--orders_p', default=2, type=int, help='spline order for pressure field (default: 2)')
	parser.add_argument('--orders_z', default=2, type=int, help='For wave equation: spline order for z positions and velocities of membrane (default: 2)')
	
	# Fluid parameters
	parser.add_argument('--rho', default=1, type=float, help='fluid density rho')
	parser.add_argument('--mu', default=1, type=float, help='fluid viscosity mu')
	parser.add_argument('--dt', default=1, type=float, help='dt per time intetgration step')
	
	# Wave parameters
	parser.add_argument('--stiffness', default=10, type=float, help='stiffness coefficient for wave equation')
	parser.add_argument('--damping', default=0.1, type=float, help='damping coefficient for wave equation')
	
	# Domain parameters
	parser.add_argument('--width', default=None, type=int, help='domain width')
	parser.add_argument('--height', default=None, type=int, help='domain height')
	parser.add_argument('--max_speed', default=1, type=float, help='max speed for boundary conditions in dataset (default: 1)')
	
	# Logger / Load parameters
	parser.add_argument('--plot', default=False, type=str2bool, help='plot during training')
	parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')
	parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
	parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
	parser.add_argument('--n_warmup_steps', default=None, type=int, help='number of warm up steps to perform when loading model in order to initialize dataset (default: None)')
	parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: True)')
	parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
	
	# parse parameters
	params = parser.parse_args()
	print(f"Parameters: {vars(params)}")
	
	return params

def get_hyperparam_fluid(params):
	return f"fluid net {params.net}; hs {params.hidden_size}; ov {params.orders_v}; op {params.orders_p}; mu {params.mu}; rho {params.rho}; dt {params.dt};"

def get_hyperparam_wave(params):
	return f"wave net {params.net}; hs {params.hidden_size}; oz {params.orders_z}; stiffness {params.stiffness}; damping {params.damping}; dt {params.dt};"

params = params()

def toCuda(x):
	if type(x) is tuple or type(x) is list:
		return [toCuda(xi) if params.cuda else xi for xi in x]
	return x.cuda() if params.cuda else x

def toCpu(x):
	if type(x) is tuple:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()
 
