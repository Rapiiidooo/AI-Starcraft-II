from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units
from math import sqrt
import sys


def printf(str_format, *args):
	sys.stdout.write(str_format % args)


# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_X_COORD = features.FeatureUnit.x
_Y_COORD = features.FeatureUnit.y


# python3 -m pysc2.bin.agent --map CollectMineralShards
# --agent locate_units.Simple --max_episodes 1 
# --use_feature_units

class Simple(base_agent.BaseAgent):
	def setup(self, obs_spec, action_spec):
		super(Simple, self).setup(obs_spec, action_spec)
		if "feature_units" not in obs_spec:
			raise Exception("feature_units observation NOT ACTIVATED")

	def step(self, obs):
		super(Simple, self).step(obs)
		unites = [unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.SELF]
		shards = [shard for shard in obs.observation.feature_units if shard.alliance == features.PlayerRelative.NEUTRAL]

		for unit in unites:
			self.get_nearest(unit, shards)

		# printf("--- %d units : ", len(unites))
		# for u in unites:
		# 	printf("(%d %d) ", u[_X_COORD], u[_Y_COORD])
		# print("")

		# printf("--- %d shards : ", len(shards))
		# for u in shards:
		# 	printf("(%d %d) ", u[_X_COORD], u[_Y_COORD])
		# print("")
		return actions.FUNCTIONS.no_op()

	def get_nearest(self, unit, targets):
		closest = targets[0]
		unita, unitb = unit

		closesta, closestb = closest
		dclosest = self.distance_euclid(unita, unitb, closesta, closestb)
		for x, y, i in enumerate(targets):
			dtmp = self.distance_euclid(unita, unitb, x, y)
			if dclosest > dtmp:
				dclosest = dtmp
				closest = targets[i]
		return closest

	@staticmethod
	def distance_euclid(xa, xb, ya, yb):
		return sqrt((xb - xa) ** 2 + (yb - ya) ** 2)
