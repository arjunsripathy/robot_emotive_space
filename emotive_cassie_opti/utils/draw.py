#!/usr/bin/env python
# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import pickle

SHARED_FOLDER = "../../shared/"

class CassieDrawer():
	def __init__(self, robot_states, init_state, target_state, obstacles, obstacle_margin=0.1, export_fig=False):
		self.robot_states = robot_states
		self.init_state = init_state
		self.target_state = target_state
		self.robot_radius = 0.5/2.0

		self.obstacles = obstacles
		self.obstacle_margin = obstacle_margin

		self.fig = plt.figure()
		self.ax = plt.axes(xlim=(-3, 3), ylim=(-2.5, 2.5))
		# self.fig.set_dpi(400)
		self.fig.set_size_inches(7, 6.5)
		# init for plot
		self.animation_init()

		self.ani = animation.FuncAnimation(self.fig, self.animation_loop, range(len(self.robot_states)),
										   init_func=self.animation_init, interval=100, repeat=False)

		plt.grid(linestyle='--')
		if export_fig:
			self.ani.save('./v1.gif', writer='imagemagick', fps=100)
		plt.show()

	def animation_init(self):
		# plot obstacle
		if len(self.obstacles)!=0:
			for i in range(len(self.obstacles)):
				self.obstacle_circle = plt.Circle([self.obstacles[i][0],self.obstacles[i][1]], self.obstacle_margin + self.obstacles[i][2], color='y',fill=True)
				self.ax.add_artist(self.obstacle_circle)
		# plot target state
		theta = self.target_state[6]
		self.target_rec = plt.Rectangle([self.target_state[0] - self.robot_radius*np.cos(theta) + self.robot_radius*np.sin(theta), \
											self.target_state[1] - self.robot_radius*np.sin(theta) - self.robot_radius*np.cos(theta)], \
											self.robot_radius*2, self.robot_radius*2, np.degrees(theta), fill=False)
		self.ax.add_artist(self.target_rec)

		self.robot_body = plt.Rectangle([self.init_state[0] - self.robot_radius*np.cos(self.init_state[6]) + self.robot_radius*np.sin(self.init_state[6]), \
									   self.init_state[1] - self.robot_radius*np.sin(self.init_state[6]) - self.robot_radius*np.cos(self.init_state[6])], \
									   self.robot_radius*2, self.robot_radius*2, np.degrees(self.init_state[6]), fill=False)
		self.ax.add_artist(self.robot_body)

		return self.target_rec, self.robot_body

	def animation_loop(self, indx):
		position = self.robot_states[indx][:3]
		theta = self.robot_states[indx][6]
		
		self.robot_body.remove()
		radius = self.robot_radius * 0.7 / (1.3 - position[2])
		self.robot_body = plt.Rectangle([position[0] - radius*np.cos(theta) + radius*np.sin(theta), \
										 position[1] - radius*np.sin(theta) - radius*np.cos(theta)], \
										 radius*2, radius*2, np.degrees(theta), fill=True)
		self.ax.add_artist(self.robot_body)
				
		return self.robot_body

if (__name__ == '__main__'):
	walk_style = input("Walk Style: ")
	file_path = SHARED_FOLDER + f"{walk_style}_walk.pkl"
	drawing_info = pickle.load(open(file_path, 'rb'))

	drawing_result = CassieDrawer(**drawing_info)

