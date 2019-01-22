from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from math import sqrt

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_X_COORD = features.FeatureUnit.x
_Y_COORD = features.FeatureUnit.y
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

# python3 -m pysc2.bin.agent --map CollectMineralShards
# --agent locate_units.Simple --max_episodes 1 
# --use_feature_units


class Simple(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self._previous_mineral_xy = [-1, -1]
        self._marine_selected = False

    def setup(self, obs_spec, action_spec):
        super(Simple, self).setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("feature_units observation NOT ACTIVATED")

    def reset(self):
        super(Simple, self).reset()
        self._marine_selected = False
        self._previous_mineral_xy = [-1, -1]

    def step(self, obs):
        super(Simple, self).step(obs)
        unites = [unit for unit in obs.observation.feature_units if unit.alliance == features.PlayerRelative.SELF]
        shards = [shard for shard in obs.observation.feature_units if shard.alliance == features.PlayerRelative.NEUTRAL]
        if not unites:
            return FUNCTIONS.no_op()
        unite = next((m for m in unites if m.is_selected == self._marine_selected), unites[0])
        unite_xy = [unite.x, unite.y]

        if not unite.is_selected:
            # Nothing selected or the wrong marine is selected.
            self._marine_selected = True
            return FUNCTIONS.select_point("select", unite_xy)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            for index, shard in enumerate(shards):
                # Don't go for the same mineral shard as other marine.
                if self._previous_mineral_xy == [shard.x, shard.y]:
                    shards.pop(index)

            if shards:
                closest_mineral = self.get_nearest(unite, shards)

                # Swap to the other marine.
                self._marine_selected = False
                self._previous_mineral_xy = [closest_mineral.x, closest_mineral.y]
                return FUNCTIONS.Move_screen("now", self._previous_mineral_xy)
        return FUNCTIONS.no_op()

    def move_unit(self):
        pass

    def get_nearest(self, unit, targets):
        closest = targets[0]

        dclosest = self.distance_euclid(unit.x, unit.y, closest.x, closest.y)
        for index, target in enumerate(targets):
            dtmp = self.distance_euclid(unit.x, unit.y, target.x, target.y)
            if dclosest > dtmp:
                dclosest = dtmp
                closest = targets[index]
        return closest

    @staticmethod
    def distance_euclid(xa, xb, ya, yb):
        return sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
