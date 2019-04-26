import math
import os
import random
import sys
import time

from QLearningTable import QLearningTable
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

ACTIONS_BUILD_BUILDING = [
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MISSILE_TURRET,
    ACTION_BUILD_ENGINEERING_BAY
]

ACTIONS_BUILD_UNIT = [
    ACTION_BUILD_SCV,
    ACTION_BUILD_MARINE
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


class SparseAgentDefensive(base_agent.BaseAgent):
    def __init__(self):
        super(SparseAgentDefensive, self).__init__()

        # Timer pour avoir la durée de la partie (not efficient)
        self.start_time = time.time()
        # Table qLearning
        self.qlearn = QLearningTable(actions=list(range(len(SMART_ACTIONS))))
        # Info player
        self.base_top_left = 0
        self.cc_screen_y = None
        self.cc_screen_x = None
        self.cc_minimap_y = None
        self.cc_minimap_x = None
        self.supply_depot_y = -30
        self.supply_depot_x = -36
        self.unit_selected = ""
        self.cc_y = None
        self.cc_x = None
        self.cc_count = None
        self.marine_y = None
        self.marine_x = None
        self.marine_count = None
        self.scv_y = None
        self.scv_x = None
        self.scv_count = None
        self.depot_y = None
        self.depot_x = None
        self.supply_depot_count = None
        self.barracks_y = None
        self.barracks_x = None
        self.barracks_count = None
        self.missile_turret_y = None
        self.missile_turret_x = None
        self.missile_turret_count = None
        self.bunker_y = None
        self.bunker_x = None
        self.bunker_count = None
        self.engineering_bay_y = None
        self.engineering_bay_x = None
        self.engineering_bay_built = None
        # Info actions
        self.smart_action = None
        self.smart_action_x = None
        self.smart_action_y = None
        self.previous_action = None
        self.previous_state = None
        self.move_number = 0
        # Info general
        self.obs = None
        self.unit_type = None
        self.current_state = np.zeros(12)
        # self.testx = 0
        # self.testy = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def reset(self):
        super().reset()
        self.__init__()

    def init_base(self, obs):
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        self.cc_screen_y, self.cc_screen_x = (self.unit_type == _TERRAN_COMMANDCENTER).nonzero()
        if self.base_top_left == 1:
            self.cc_minimap_x = 18
            self.cc_minimap_y = 24
        else:
            self.cc_minimap_x = 40
            self.cc_minimap_y = 47

    def move_camera_to_base(self):
        if _MOVE_CAMERA in self.obs.observation["available_actions"]:
            return actions.FUNCTIONS.move_camera([self.cc_minimap_x, self.cc_minimap_y])
        else:
            return actions.FunctionCall(_NO_OP, [])

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

        save_score(obs.observation["score_cumulative"]["score"], self.steps)
        return actions.FunctionCall(_NO_OP, [])

    def init_current_state(self, obs):
        self.obs = obs
        # NB de COMMAND CENTER
        self.cc_y, self.cc_x = (self.unit_type == _TERRAN_COMMANDCENTER).nonzero()
        self.cc_count = 1 if self.cc_y.any() else 0

        # NB de MARINE
        self.marine_y, self.marine_x = (self.unit_type == _TERRAN_MARINE).nonzero()
        self.marine_count = int(round(len(self.marine_y) / 9))

        # NB de OUVRIER
        self.scv_y, self.scv_x = (self.unit_type == _TERRAN_SCV).nonzero()
        self.scv_count = int(round(len(self.scv_y) / 9))

        # NB de SUPPLY DEPOT
        self.depot_y, self.depot_x = (self.unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        self.supply_depot_count = int(round(len(self.depot_y) / 69))

        # NB de CASERNE
        self.barracks_y, self.barracks_x = (self.unit_type == _TERRAN_BARRACKS).nonzero()
        self.barracks_count = int(round(len(self.barracks_y) / 137))

        # NB de TOURRELLE LANCE MISSILE
        self.missile_turret_y, self.missile_turret_x = (self.unit_type == _TERRAN_MISSILE_TURRET).nonzero()
        self.missile_turret_count = int(round(len(self.missile_turret_y) / 52))

        # NB de BUNKER
        self.bunker_y, self.bunker_x = (self.unit_type == _TERRAN_BUNKER).nonzero()
        self.bunker_count = int(round(len(self.bunker_y) / 12))

        # NB de centre d'usine (pour débloquer la construction des bunkers
        self.engineering_bay_y, self.engineering_bay_x = (self.unit_type == _TERRAN_ENGINEERING_BAY).nonzero()
        self.engineering_bay_built = True if self.engineering_bay_y.any() else False

        self.current_state = np.zeros(12)
        self.current_state[0] = self.cc_count
        self.current_state[1] = self.supply_depot_count
        self.current_state[2] = self.barracks_count
        self.current_state[3] = obs.observation['player'][_ARMY_SUPPLY]

        hot_squares = np.zeros(4)
        enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 32))
            x = int(math.ceil((enemy_x[i] + 1) / 32))
            hot_squares[((y - 1) * 2) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 4):
            self.current_state[i + 4] = hot_squares[i]

        green_squares = np.zeros(4)
        friendly_y, friendly_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        for i in range(0, len(friendly_y)):
            y = int(math.ceil((friendly_y[i] + 1) / 32))
            x = int(math.ceil((friendly_x[i] + 1) / 32))
            green_squares[((y - 1) * 2) + (x - 1)] = 1

        if not self.base_top_left:
            green_squares = green_squares[::-1]

        for i in range(0, 4):
            self.current_state[i + 8] = green_squares[i]

    def get_excluded_actions(self, obs):
        excluded_actions = []
        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        supply_free = supply_limit - supply_used
        inactiv_worker = obs.observation['player'][7]

        if self.cc_count == 0 or self.scv_count >= 15 or worker_supply > 20:
            excluded_actions.append(ACTION_ID_BUILD_SCV)

        if self.supply_depot_count >= 6 or worker_supply == 0 or supply_free > 4:
            excluded_actions.append(ACTION_ID_BUILD_SUPPLY_DEPOT)

        if self.supply_depot_count == 0 or self.barracks_count >= 2 or worker_supply == 0:
            excluded_actions.append(ACTION_ID_BUILD_BARRACKS)

        if supply_free == 0 or self.barracks_count == 0:
            excluded_actions.append(ACTION_ID_BUILD_MARINE)

        if inactiv_worker <= 0:
            excluded_actions.append(ACTION_ID_SCV_INACTIV_TO_MINE)

        if self.supply_depot_count == 0 or self.barracks_count == 0 or worker_supply == 0 or self.engineering_bay_built is True:
            excluded_actions.append(ACTION_ID_BUILD_ENGINEERING_BAY)

        if self.supply_depot_count == 0 or self.barracks_count == 0 or worker_supply == 0 or self.missile_turret_count > 0 is True or \
                self.engineering_bay_built is False:
            excluded_actions.append(ACTION_ID_BUILD_MISSILE_TURRET)

        if self.supply_depot_count == 0 or \
                self.barracks_count == 0 or \
                worker_supply == 0 or \
                army_supply <= 0 or \
                self.engineering_bay_built is False:
            for action in ACTION_ID_DEFEND_POSITION:
                excluded_actions.append(action)

        return excluded_actions

    def inc_move_number(self):
        self.move_number += 1

    def init_move_number(self):
        self.move_number = 0

    # Fonction principal appelé n fois à chaque étape par le jeu n correspondant au step_mul
    def step(self, obs):
        super(SparseAgentDefensive, self).step(obs)

        # Unité / Batiment et ressources présentes sur l'écran actuelle
        self.unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        # Fin de la partie
        if obs.last():
            return self.ending_game(obs)

        # Début de la partie (Appelé une fois)
        if obs.first():
            self.init_base(obs)

        # Initialisation de l'état actuelle des éléments du joueurs.
        self.init_current_state(obs)

        # Choix de la smart_action en fonction des éléments du jeu, si plus aucune action en cours
        if self.smart_action is None:
            self.init_move_number()
            excluded_actions = self.get_excluded_actions(obs)
            self.previous_state = self.current_state
            self.previous_action = self.qlearn.choose_action(str(self.current_state), excluded_actions)
            self.smart_action, self.smart_action_x, self.smart_action_y = self.split_action(self.previous_action)

        # Ajout de la précèdente action et des scores y résultant dans la table Qlearning
        if self.previous_action is not None:
            self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(self.current_state))

        # smart action choisi par le qlearning ?
        if self.smart_action in ACTIONS_BUILD_BUILDING:
            return self.actions_build_building()
        elif self.smart_action in ACTIONS_BUILD_UNIT:
            return self.actions_build_unit()
        elif self.smart_action == ACTION_DEFEND_POSITION:
            return self.action_defend_position()
        elif self.smart_action == ACTION_SCV_INACTIV_TO_MINE:
            return self.action_scv_inactiv_to_mine()
        return self.move_camera_to_base()

    def select_unit(self, unit):
        # Séléctionner un SCV
        if unit == "SCV":
            unit_y, unit_x = (self.unit_type == _TERRAN_SCV).nonzero()
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                self.unit_selected = "SCV"
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        # Séléctionner l'armée totale
        elif unit == "ARMY":
            if _SELECT_ARMY in self.obs.observation['available_actions']:
                self.unit_selected = "ARMY"
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        # Séléctionner un bunker
        elif unit == "BUNKER":
            if self.bunker_y.any():
                i = random.randint(0, len(self.bunker_y) - 1)
                target = [self.bunker_x[i], self.bunker_y[i]]
                self.unit_selected = "BUNKER"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        # Sélécionner une caserne
        elif unit == "BARRACK":
            if self.barracks_y.any():
                i = random.randint(0, len(self.barracks_y) - 1)
                target = [self.barracks_x[i], self.barracks_y[i]]
                self.unit_selected = "BARRACK"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        return self.move_camera_to_base()

    # ACTIONS GROUPES
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def actions_build_building(self):
        if self.move_number == 0:
            self.inc_move_number()
            return self.select_unit("SCV")

        if self.smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            return self.action_build_supply_depot()
        elif self.smart_action == ACTION_BUILD_BARRACKS:
            return self.action_build_barracks()
        elif self.smart_action == ACTION_BUILD_ENGINEERING_BAY:
            return self.action_build_engineering_bay()
        elif self.smart_action == ACTION_BUILD_MISSILE_TURRET:
            return self.action_build_missile_turret()
        else:
            return self.move_camera_to_base()

    def actions_build_unit(self):
        if self.smart_action == ACTION_BUILD_SCV:
            return self.action_build_scv()
        elif self.smart_action == ACTION_BUILD_MARINE:
            return self.action_build_marine()
        else:
            return self.move_camera_to_base()

    # ACTIONS SPECIFIQUES
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def action_re_init_smart_action(self):
        self.smart_action = None
        self.init_move_number()
        return self.move_camera_to_base()

    def action_build_scv(self):
        self.inc_move_number()

        # Séléction command center
        if self.move_number == 1:
            if self.cc_count > 0:
                target = [int(self.cc_x.mean()), int(self.cc_y.mean())]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        # Lancement construction du worker
        elif self.move_number == 2:
            if _TRAIN_SCV in self.obs.observation["available_actions"]:
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
        return self.action_re_init_smart_action()


    def action_build_supply_depot(self):
        # Construction du supply depot
        if _BUILD_SUPPLY_DEPOT in self.obs.observation['available_actions']:
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
        return self.action_re_init_smart_action()

    def action_build_barracks(self):
        if _BUILD_BARRACKS in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                if self.barracks_count == 0:
                    target = self.transform_distance(round(self.cc_screen_x.mean()), 15,
                                                     round(self.cc_screen_y.mean()), -9)
                else:
                    target = self.transform_distance(round(self.cc_screen_x.mean()), 15,
                                                     round(self.cc_screen_y.mean()), 12)
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_marine(self):
        if self.move_number == 0:
            self.select_unit("BARRACK")
        elif self.move_number == 1:
            if _TRAIN_MARINE in self.obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

    def action_build_missile_turret(self):
        if _BUILD_MISSILE_TURRET in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                target = self.transform_distance(
                    round(self.cc_screen_x.mean()),
                    0,
                    round(self.cc_screen_y.mean()),
                    15
                )
                return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, target])

    def action_build_engineering_bay(self):
        if _BUILD_ENGINEERING_BAY in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                target = self.transform_distance(round(self.cc_screen_x.mean()), -8,
                                                 round(self.cc_screen_y.mean()), 15)
                return actions.FunctionCall(_BUILD_ENGINEERING_BAY, [_NOT_QUEUED, target])

    def action_scv_inactiv_to_mine(self):
        # step 0
        if _SELECT_IDLE_WORKER in self.obs.observation['available_actions']:
            return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
        # step 1
        if _HARVEST_GATHER in self.obs.observation['available_actions']:
            unit_y, unit_x = (self.unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)

                m_x = unit_x[i]
                m_y = unit_y[i]

                target = [int(m_x), int(m_y)]

                return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
            else:
                target = [self.cc_minimap_x, self.cc_minimap_y]
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

    def action_defend_position(self):
        # step 0
        if _MOVE_CAMERA in self.obs.observation["available_actions"]:
            return actions.FUNCTIONS.move_camera([self.smart_action_x, self.smart_action_y])
        # step 1
        self.move_number = 3
        x = random.randint(0, 83)
        y = random.randint(0, 83)
        if 0 <= self.bunker_count < 2 and _BUILD_BUNKER in self.obs.observation['available_actions']:
            return actions.FunctionCall(_BUILD_BUNKER, [_NOT_QUEUED, [x, y]])
        elif 0 <= self.missile_turret_count < 2 and _BUILD_MISSILE_TURRET in self.obs.observation['available_actions']:
            if _BUILD_MISSILE_TURRET in self.obs.observation['available_actions']:
                return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, [x, y]])
        else:
            return self.select_unit("ARMY")
        # step 2
        if self.unit_selected == "ARMY":
            self.move_number = 4
            self.unit_selected = ""
            if self.bunker_y.any() and _MOVE_SCREEN in self.obs.observation['available_actions']:
                i = random.randint(0, len(self.bunker_y) - 1)
                target = [self.bunker_x[i], self.bunker_y[i]]
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
            elif _ATTACK_MINIMAP in self.obs.observation['available_actions']:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [x, y]])
        # step 3
        self.select_unit("BUNKER")
        # step 4
        if self.unit_selected == "BUNKER":
            self.unit_selected = ""
            if _LOAD_BUNKER_SCREEN in self.obs.observation['available_actions'] and self.marine_count > 0:
                target = [int(round(self.marine_x.mean())), int(round(self.marine_y.mean()))]
                return actions.FunctionCall(_LOAD_BUNKER_SCREEN, [_NOT_QUEUED, target])

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # FIN DES ACTIONS

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
