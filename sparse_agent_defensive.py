import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from absl import flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features

# region Variables

# MyGlobalVar
# ------------------------------------
_SCORE_FILE = 'scores.txt'
_STEPS_BEFORE_WIN = 5000

# PySc2 Var
# ------------------------------------
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_CAMERA = features.MINIMAP_FEATURES.camera.index
_MOVE_CAMERA = 1

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_MISSILE_TURRET = actions.FUNCTIONS.Build_MissileTurret_screen.id
_BUILD_ENGINEERING_BAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id
_BUILD_BUNKER = actions.FUNCTIONS.Build_Bunker_screen.id
_LOAD_BUNKER_SCREEN = actions.FUNCTIONS.Load_Bunker_screen.id

_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_ENGINEERING_BAY = 22
_TERRAN_MISSILE_TURRET = 23
_TERRAN_BUNKER = 24
_TERRAN_SCV = 45
_TERRAN_MARINE = 48

_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'defensive_agent_data'

ACTION_ID_DO_NOTHING = 0
ACTION_ID_BUILD_SCV = 1
ACTION_ID_BUILD_SUPPLY_DEPOT = 2
ACTION_ID_BUILD_BARRACKS = 3
ACTION_ID_BUILD_MARINE = 4
ACTION_ID_BUILD_MISSILE_TURRET = 5
ACTION_ID_BUILD_ENGINEERING_BAY = 6
ACTION_ID_SCV_INACTIV_TO_MINE = 7  # must be the last one before ACTION_ID_DEFEND_POSITION
ACTION_ID_DEFEND_POSITION = []

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SCV = 'buildscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_BUILD_MISSILE_TURRET = 'buildmissileturret'
ACTION_BUILD_ENGINEERING_BAY = 'buildengineeringbay'
ACTION_SCV_INACTIV_TO_MINE = 'reactiveworker'
ACTION_DEFEND_POSITION = 'defend'

ACTIONS_SCV = [
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MISSILE_TURRET,
    ACTION_BUILD_ENGINEERING_BAY
]

SMART_ACTIONS = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_BUILD_MISSILE_TURRET,
    ACTION_BUILD_ENGINEERING_BAY,
    ACTION_SCV_INACTIV_TO_MINE
]

MAP_ROUTE = [
    [12, 12],
    [32, 12],
    [52, 12],
    [12, 32],
    [32, 32],
    [52, 32],
    [12, 52],
    [32, 52],
    [52, 52]
]

for index, route in enumerate(MAP_ROUTE):
    routex, routey = route
    SMART_ACTIONS.append(ACTION_DEFEND_POSITION + '_' + str(routex) + '_' + str(routey))
    ACTION_ID_DEFEND_POSITION.append(ACTION_ID_SCV_INACTIV_TO_MINE + index + 1)
# endregion


# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=None):
        if excluded_actions is None:
            excluded_actions = []
        self.check_state_exist(observation)

        self.disallowed_actions[observation] = excluded_actions

        state_action = self.q_table.loc[observation, :]

        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            action = np.random.choice(state_action.index)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.loc[s, a]

        s_rewards = self.q_table.loc[s_, :]

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SparseAgentDefensive(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgentDefensive, self).__init__()
        self.start_time = time.time()
        self.qlearn = QLearningTable(actions=list(range(len(SMART_ACTIONS))))

        self.previous_action = None
        self.previous_state = None

        self.base_top_left = 0
        self.cc_screen_y = None
        self.cc_screen_x = None
        self.cc_minimap_y = None
        self.cc_minimap_x = None

        self.move_number = 0

        self.supply_depot_y = -30
        self.supply_depot_x = -36

        self.army_selected = False
        self.bunker_selected = False

        self.testx = 0
        self.testy = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def reset(self):
        super().reset()
        self.__init__()

    def transform_distance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def transform_location(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        return [x, y]

    @staticmethod
    def split_action(action_id):
        smart_action = SMART_ACTIONS[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return smart_action, int(x), int(y)

    @staticmethod
    def save_score(score, steps):
        try:
            num_lines = sum(1 for _ in open(_SCORE_FILE)) + 1
        except:
            num_lines = 1
        with open(_SCORE_FILE, "a+") as scores_file:
            print("{};{};{}"
                  # .format(num_lines, datetime.timedelta(seconds=(time.time() - start_time)), score),
                  .format(num_lines, steps, score),
                  file=scores_file)

    def ending_game(self, obs):
        # reward = obs.reward
        score = obs.observation["score_cumulative"]["score"]
        reward = 1 if self.steps > _STEPS_BEFORE_WIN or score > 2000 else -1

        if obs.reward > 0:
            reward = 10

        self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        self.previous_action = None
        self.previous_state = None
        self.move_number = 0

        self.save_score(obs.observation["score_cumulative"]["score"], self.steps)
        return actions.FunctionCall(_NO_OP, [])

    def init_base(self, obs, unit_type):
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        self.cc_screen_y, self.cc_screen_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        if self.base_top_left == 1:
            self.cc_minimap_x = 18
            self.cc_minimap_y = 24
        else:
            self.cc_minimap_x = 40
            self.cc_minimap_y = 47

    @staticmethod
    def get_excluded_actions(
            obs,
            cc_count,
            scv_count,
            supply_depot_count,
            barracks_count,
            missile_turret_count,
            engineering_bay
    ):
        excluded_actions = []
        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        supply_free = supply_limit - supply_used
        inactiv_worker = obs.observation['player'][7]

        if cc_count == 0 or scv_count >= 15 or worker_supply > 20:
            excluded_actions.append(ACTION_ID_BUILD_SCV)

        if supply_depot_count >= 6 or worker_supply == 0 or supply_free > 4:
            excluded_actions.append(ACTION_ID_BUILD_SUPPLY_DEPOT)

        if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
            excluded_actions.append(ACTION_ID_BUILD_BARRACKS)

        if supply_free == 0 or barracks_count == 0:
            excluded_actions.append(ACTION_ID_BUILD_MARINE)

        if inactiv_worker <= 0:
            excluded_actions.append(ACTION_ID_SCV_INACTIV_TO_MINE)

        if supply_depot_count == 0 or barracks_count == 0 or worker_supply == 0 or engineering_bay is True:
            excluded_actions.append(ACTION_ID_BUILD_ENGINEERING_BAY)

        if supply_depot_count == 0 or barracks_count == 0 or worker_supply == 0 or missile_turret_count > 0 is True or \
                engineering_bay is False:
            excluded_actions.append(ACTION_ID_BUILD_MISSILE_TURRET)

        if supply_depot_count == 0 or \
                barracks_count == 0 or \
                worker_supply == 0 or \
                army_supply <= 0 or \
                engineering_bay is False:
            for action in ACTION_ID_DEFEND_POSITION:
                excluded_actions.append(action)

        return excluded_actions

    def move_camera_to_base(self, obs):
        if _MOVE_CAMERA in obs.observation["available_actions"]:
            return actions.FUNCTIONS.move_camera([self.cc_minimap_x, self.cc_minimap_y])
        else:
            return actions.FunctionCall(_NO_OP, [])

    def step(self, obs):
        super(SparseAgentDefensive, self).step(obs)

        if obs.last():
            return self.ending_game(obs)

        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        if obs.first():
            self.init_base(obs, unit_type)

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        marine_y, marine_x = (unit_type == _TERRAN_MARINE).nonzero()
        marine_count = int(round(len(marine_y) / 9))

        scv_y, scv_x = (unit_type == _TERRAN_SCV).nonzero()
        scv_count = int(round(len(scv_y) / 9))

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        missile_turret_y, missile_turret_x = (unit_type == _TERRAN_MISSILE_TURRET).nonzero()
        missile_turret_count = int(round(len(missile_turret_y) / 52))

        bunker_y, bunker_x = (unit_type == _TERRAN_BUNKER).nonzero()
        bunker_count = int(round(len(bunker_y) / 12))

        engineering_bay_y, engineering_bay_x = (unit_type == _TERRAN_ENGINEERING_BAY).nonzero()
        engineering_bay_built = True if engineering_bay_y.any() else False

        # étape 1 choisir une smart_action, en fonction des éléments du jeu
        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(12)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

            hot_squares = np.zeros(4)
            enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]

            green_squares = np.zeros(4)
            friendly_y, friendly_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))

                green_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                green_squares = green_squares[::-1]

            for i in range(0, 4):
                current_state[i + 8] = green_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            excluded_actions = self.get_excluded_actions(obs,
                                                         cc_count,
                                                         scv_count,
                                                         supply_depot_count,
                                                         barracks_count,
                                                         missile_turret_count,
                                                         engineering_bay_built)

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.split_action(self.previous_action)

            # switch case smart action actuel étape 1
            if smart_action in ACTIONS_SCV or smart_action == ACTION_DEFEND_POSITION:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_SCV:
                if cc_count > 0:
                    target = [int(cc_x.mean()), int(cc_y.mean())]
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_SCV_INACTIV_TO_MINE:
                if _SELECT_IDLE_WORKER in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])

        # switch case smart action actuel étape 2
        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.split_action(self.previous_action)

            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 6 and \
                        _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if self.cc_screen_y.any():
                        target = self.transform_distance(
                            round(self.cc_screen_x.mean()),
                            self.supply_depot_x,
                            round(self.cc_screen_y.mean()),
                            self.supply_depot_y
                        )
                        self.supply_depot_y += 2
                        if 0 <= target[0] < 83 and 0 <= target[1] < 83:
                            return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_BARRACKS:
                if 0 <= barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                    if self.cc_screen_y.any():
                        if barracks_count == 0:
                            target = self.transform_distance(round(self.cc_screen_x.mean()), 15,
                                                             round(self.cc_screen_y.mean()), -9)
                        else:
                            target = self.transform_distance(round(self.cc_screen_x.mean()), 15,
                                                             round(self.cc_screen_y.mean()), 12)

                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_ENGINEERING_BAY:
                if _BUILD_ENGINEERING_BAY in obs.observation['available_actions']:
                    if self.cc_screen_y.any():
                        target = self.transform_distance(round(self.cc_screen_x.mean()), -8,
                                                         round(self.cc_screen_y.mean()), 15)
                        return actions.FunctionCall(_BUILD_ENGINEERING_BAY, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MISSILE_TURRET:
                if _BUILD_MISSILE_TURRET in obs.observation['available_actions']:
                    if self.cc_screen_y.any():
                        target = self.transform_distance(
                            round(self.cc_screen_x.mean()),
                            0,
                            round(self.cc_screen_y.mean()),
                            15
                        )
                        return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_SCV and _TRAIN_SCV in obs.observation["available_actions"]:
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

            elif smart_action == ACTION_BUILD_MARINE and _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_DEFEND_POSITION and _MOVE_CAMERA in obs.observation["available_actions"]:
                return actions.FUNCTIONS.move_camera([x, y])

        # switch case smart action actuel étape 3
        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.split_action(self.previous_action)

            if smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action in ACTIONS_SCV or smart_action == ACTION_SCV_INACTIV_TO_MINE:
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
                    else:
                        target = [self.cc_minimap_x, self.cc_minimap_y]
                        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

            elif smart_action == ACTION_DEFEND_POSITION:
                self.move_number = 3
                x = random.randint(0, 83)
                y = random.randint(0, 83)
                if 0 <= bunker_count < 2 and _BUILD_BUNKER in obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_BUNKER, [_NOT_QUEUED, [x, y]])
                elif 0 <= missile_turret_count < 2 and _BUILD_MISSILE_TURRET in obs.observation['available_actions']:
                    if _BUILD_MISSILE_TURRET in obs.observation['available_actions']:
                            return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, [x, y]])
                elif _SELECT_ARMY in obs.observation['available_actions']:
                    self.army_selected = True
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        # switch case pour l'action defend position uniquement (pour l'instant)
        elif self.move_number == 3:
            self.move_number = 0

            smart_action, x, y = self.split_action(self.previous_action)

            if self.army_selected is True:
                self.move_number = 4
                self.army_selected = False
                if bunker_y.any() and _MOVE_SCREEN in obs.observation['available_actions']:
                    i = random.randint(0, len(bunker_y) - 1)
                    target = [bunker_x[i], bunker_y[i]]
                    return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
                elif _ATTACK_MINIMAP in obs.observation['available_actions']:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [x, y]])

        elif self.move_number == 4:
            self.move_number += 1
            if bunker_y.any():
                i = random.randint(0, len(bunker_y) - 1)
                target = [bunker_x[i], bunker_y[i]]
                self.bunker_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

        elif self.move_number == 5:
            self.move_number = 0
            if self.bunker_selected is True:
                self.bunker_selected = False
                if _LOAD_BUNKER_SCREEN in obs.observation['available_actions'] and marine_count > 0:
                    target = [int(round(marine_x.mean())), int(round(marine_y.mean()))]
                    return actions.FunctionCall(_LOAD_BUNKER_SCREEN, [_NOT_QUEUED, target])

        return self.move_camera_to_base(obs)

    # Check by self hand some position
    # def handself_checking():
    #     user_input = input("Enter  coord")
    #     try:
    #         if user_input.__contains__('x'):
    #             self.testx = int(user_input.replace('x', ''))
    #         if user_input.__contains__('y'):
    #             self.testy = int(user_input.replace('y', ''))
    #     except:
    #         pass
    #     print(self.testx, ', ', self.testy)
    #     if _MOVE_CAMERA in obs.observation["available_actions"]:
    #         return actions.FUNCTIONS.move_camera([self.testx, self.testy])
    #     else:
    #         return actions.FunctionCall(_NO_OP, [])


def run(agent):
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[
                        sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(
                            sc2_env.Race.random,
                            sc2_env.Difficulty.easy
                        )
                    ],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True,
                        hide_specific_actions=False
                    ),
                    step_mul=8,
                    game_steps_per_episode=0,
                    visualize=False) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass


def main():
    agent = SparseAgentDefensive()
    run(agent)


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    main()
