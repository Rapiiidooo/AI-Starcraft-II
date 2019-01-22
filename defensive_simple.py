import time

import random as rand
from pysc2.agents import base_agent
from pysc2.lib import actions, features

# Functions
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_BARRACKS = 21

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SCV_WORKING = 6
_SUPPLY_USED = 3
_SUPPLY_MAX = 4


class SimpleAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.building = []
        self.selected = ''
        self.base_top_left = None
        self.command_center_selected = False
        self.max_SCV = False
        self.supply_depot_built = 0
        self.barracks_built = False
        self.scv_selected = False
        self.barracks_selected = False
        self.barracks_rallied = False
        self.army_selected = False
        self.army_rallied = False
        self.zone_supply_x = 0
        self.zone_supply_y = 20

    def reset(self):
        self.__init__()

    def step(self, obs):
        super(SimpleAgent, self).step(obs)

        # time.sleep(0.5)
        # init position de la base
        if self.base_top_left is None:
            player_y, player_x = (obs.observation["feature_minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        # print(obs.observation["player"])

        # construction max d'ouvrier pour la ou les mines
        if obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and \
                obs.observation["player"][_SCV_WORKING] < 14:
            if self.selected != 'command_center':
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.selected = 'command_center'
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _TRAIN_SCV in obs.observation["available_actions"]:
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
        # construction d'un dépot de ravitaillement
        if self.supply_depot_built == 0:
            if self.selected != 'scv':
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                target = [int(unit_x[0]), int(unit_y[0])]
                self.selected = 'scv'
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                target = self.transform_location(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                self.supply_depot_built += 1
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
        # construction d'une caserne
        elif not self.barracks_built:
            if self.selected != 'scv':
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                target = [int(unit_x[0]), int(unit_y[0])]
                self.selected = 'scv'
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                target = self.transform_location(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                self.barracks_built = True
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        # ralliement de la caserne
        elif not self.barracks_rallied:
            if self.selected != 'barracks':
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.selected = 'barracks'
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True
                # TODO Trouver avec sarsa ou qlearning quel serait le meilleur point stratégique!
                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])
                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])
        # entrainement de marine
        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX]:
            if self.selected != 'barracks':
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.selected = 'barracks'
                    self.army_rallied = False
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _TRAIN_MARINE in obs.observation["available_actions"]:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        # attaquer avec l'armée full
        elif not self.army_rallied:
            if self.selected != 'army':
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.selected = 'army'
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.selected = ''

                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])

        return actions.FunctionCall(_NOOP, [])

    def transform_location(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    @staticmethod
    def rand_location():
        return [rand.randint(0, 83), rand.randint(0, 83)]
