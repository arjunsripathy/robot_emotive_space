import casadi as ca
import numpy as np
import math
import pandas as pd

# Specifies some numerical values and methods useful for SLIP_opti

class SLIP():
	def __init__(self, bound_path):
		self.g_ = 9.81
		self.mass_ = 31.884
		self.l0_ = 1.0
		# Make larger
		self.state_bound_lb = np.array([-50, -50, 0.65, -1, -1., -0.2, 
										-np.radians(5), -np.radians(15),
										-np.radians(180), -np.radians(5),
										-np.radians(15), -np.radians(30)])
		self.state_bound_ub = np.array([ 50,  50, 1.0,   1,  1.,  0.2,
										np.radians(5), np.radians(15),
										np.radians(180), np.radians(5),
										np.radians(15), np.radians(30)])
		# Double check input bounds	
		self.input_bound_lb = np.array(6*[-100])
		self.input_bound_ub = np.array(6*[ 100])
		
		self.__bound_set = pd.read_csv(bound_path, header=None).to_numpy()
		self.height_const_offset = 0.05
		self.vx_const_offset = 0.1
		self.vy_ratio_init = 1.0/1.5
		self.vy_ratio_max = 1.0/5.0
		self.vx_ratio_init = -0.2
		self.vx_ratio_max = 0.1
		# self.z_range = [0.6, 0.95]
		self.z_range = [0.7, 1.0]

	def __stiff(self, L):
		belta0 = -6251
		belta1 = 3.779*pow(10,4)
		belta2 = -4.495*pow(10,4)
		belta4 = 2.177*pow(10,4)
		K_L = belta0 + belta1*L + belta2*L**2 + belta4*L**4
		return K_L
	
	def singleSupportDynamics(self):
		x = ca.SX.sym("x", 12)
		u = ca.SX.sym("u", 6)
		xp = ca.SX.sym("xp", 2)

		LL = math.sqrt(pow(x[0]-xp[0], 2) + pow(x[1]-xp[1], 2) + pow(x[2], 2))
		sa = x[2]/LL 

		a = self.__stiff(LL)*(self.l0_-LL)/self.mass_
		xddot = a*(x[0]-xp[0])/LL + u[0]
		yddot = a*(x[1]-xp[1])/LL + u[1]
		zddot = a*sa - self.g_ + u[2]
		rollddot, pitchddot, yawddot = u[3], u[4], u[5]

		xdot = x[3]
		ydot = x[4]
		zdot = x[5]
		rolldot, pitchdot, yawdot = x[9], x[10], x[11]

		dx = ca.vertcat(xdot, ydot, zdot, xddot, yddot, zddot, 
						rolldot, pitchdot, yawdot,
						rollddot, pitchddot, yawddot)
		
		return ca.Function('single_dynamics', [x, u, xp], [dx])
	
	def touchDownLength(self, vx, z):
		p00 =    -0.04091
		p10 =      0.9212
		p01 =   -0.008783
		touch_down_length = p00 + p10*z + p01*vx
		return touch_down_length

	def touchDownAngle(self, vx, z):
		p00 =      0.1452
		p10 =     -0.1017
		p01 =     -0.2149
		touch_down_angle = 3.14159265/2-(p00 + p10*z + p01*vx)
		return touch_down_angle

	def touchDownAbduction(self, vy, z):
		p00 =    0.003686
		p10 =    0
		p01 =    -0.1663
		touch_down_abduction = p00 + p10*z + p01*vy
		return touch_down_abduction
		

	def getBoundSet(self):
		return self.__bound_set

