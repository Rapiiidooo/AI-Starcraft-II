from pysc2.agents import base_agent
from pysc2.lib import actions

class Simple(base_agent.BaseAgent):

	def __init__(self):
		self.x = 0
		self.y = 0
		self.episodes = -1
		self.steps = -1
		self.reward = -1
		self.recule = False


	def avance_x(self):
		self.x = self.x + 1

	def avance_y(self):
		self.y = self.y + 1

	def recule_x(self):
		self.x = self.x - 1

	def step(self, obs):
		super(Simple, self).step(obs)

		for action in obs.observation.available_actions:
			print(actions.FUNCTIONS[action])


		switch = {
			1: self.avance_x,
			2: self.avance_y,
			3: self.recule_x,
		}

		## Version classique	
		# if self.x < 63:
		# 	self.x = self.x + 1
		# else:
		# 	self.x = 0
		# 	self.y = self.y + 16
		# if self.y == 16:
		# 	self.y = 32
		# if self.y == 64:
		# 	self.y = 63
		# elif self.y > 64:
		# 	self.x = 0
		# 	self.y = 0

		## Version cool
		if (self.x < 63 and self.recule == False):
			etape = 1
		else:
			if self.recule == True and self.x < 1 and self.y < 47:
				etape = 2
			elif self.y > 31 and self.y < 47:
				self.recule = True
				etape = 3
			elif self.y >= 63:
				etape = 3
				self.recule = True
			else:
				self.recule = False
				etape = 2

		if self.x >= 63 and self.y >= 64:
			etape = 3
			self.recule = True
		func = switch.get(etape, lambda: "Invalid state")
		func()

		if self.x >= 64 and self.y >= 64 or self.x < 1 and self.y >= 64 or self.x < 0 or self.y > 64:
			self.x = 0
			self.y = 0
			self.recule = False

		print("x = " + str(self.x))
		print("y = " + str(self.y))
		dest = [self.x, self.y]
		# return actions.FUNCTIONS.no_op()
		return actions.FUNCTIONS.move_camera(dest)
