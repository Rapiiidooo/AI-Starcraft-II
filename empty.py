from pysc2.agents import base_agent
from pysc2.lib import actions


class Simple(base_agent.BaseAgent):

	def step(self, obs):
		super(Simple, self).step(obs)

		for action in obs.observation.available_actions:
			print(actions.FUNCTIONS[action])

		return actions.FUNCTIONS.no_op()
