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
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
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
_TERRAN_REFINERY = 20
_TERRAN_BARRACKS = 21
_TERRAN_ENGINEERING_BAY = 22
_TERRAN_MISSILE_TURRET = 23
_TERRAN_BUNKER = 24
_TERRAN_SCV = 45
_TERRAN_MARINE = 48

_NEUTRAL_MINERAL_FIELD = 341
_NEUTRAL_VESPENE_GEYSER = 342

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'defensive_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_TRAIN_SCV = 'trainscv'
ACTION_TRAIN_MARINE = 'trainmarine'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MISSILE_TURRET = 'buildmissileturret'
ACTION_BUILD_ENGINEERING_BAY = 'buildengineeringbay'
ACTION_BUILD_REFINERY = 'buildrefinery'
ACTION_ATTACK = 'attack'
ACTION_SCV_INACTIV_TO_MINE = 'reactiveworker'
ACTION_DEFEND_POSITION = 'defend'
ACTION_ECONOMISE = 'economise'

# The order of ACTION ID AND IN SMART ACTION ARE VERY IMPORTANT
ACTION_ID_DO_NOTHING = 0
ACTION_ID_TRAIN_SCV = 1
ACTION_ID_TRAIN_MARINE = 2
ACTION_ID_BUILD_SUPPLY_DEPOT = 3
ACTION_ID_BUILD_BARRACKS = 4
ACTION_ID_BUILD_MISSILE_TURRET = 5
ACTION_ID_BUILD_ENGINEERING_BAY = 6
ACTION_ID_BUILD_REFINERY = 7
ACTION_ID_ATTACK = 8
ACTION_ID_ECONOMISE = 9
ACTION_ID_SCV_INACTIV_TO_MINE = 10
ACTION_ID_DEFEND_POSITION = []

SMART_ACTIONS = [
    ACTION_DO_NOTHING,
    ACTION_TRAIN_SCV,
    ACTION_TRAIN_MARINE,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MISSILE_TURRET,
    ACTION_BUILD_ENGINEERING_BAY,
    ACTION_BUILD_REFINERY,
    ACTION_ATTACK,
    ACTION_ECONOMISE,
    ACTION_SCV_INACTIV_TO_MINE,
]

ACTIONS_BUILD_BUILDING = [
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MISSILE_TURRET,
    ACTION_BUILD_ENGINEERING_BAY,
    ACTION_BUILD_REFINERY
]

ACTIONS_TRAIN_UNIT = [
    ACTION_TRAIN_SCV,
    ACTION_TRAIN_MARINE
]

# attack haut gauche x_18 y_24
MAP_ROUTE = [
    [36, 20],
    [38, 32],
    [22, 40],
    [20, 50]
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
        self.vespene_y = None
        self.vespene_x = None
        self.vespene_geyser_count = None
        self.refinery_y = None
        self.refinery_x = None
        self.refinery_count = None
        self.engineering_bay_y = None
        self.engineering_bay_x = None
        self.engineering_bay_built = None
        self.mineral_restant = -1
        self.mineral_x = None
        self.mineral_y = None
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
        self.mining_owned = None
        self.vespene_owned = None

        # Variable pour tester la position à la main
        self.testx = 0
        self.testy = 0

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

        # NB de mineral restant seulement autour de la base principale
        if self.cc_count >= 1:
            self.mineral_y, self.mineral_x = (self.unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
            self.mineral_restant = int(round(len(self.mineral_y)))

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

        # VESPENE présent
        self.vespene_y, self.vespene_x = (self.unit_type == _NEUTRAL_VESPENE_GEYSER).nonzero()
        self.vespene_geyser_count = int(math.ceil(len(self.vespene_y) / 97))

        # NB de RAFINERIE
        self.refinery_y, self.refinery_x = (self.unit_type == _TERRAN_REFINERY).nonzero()
        self.refinery_count = int(round(len(self.refinery_y) / 12))

        # NB de centre d'usine (pour débloquer la construction des bunkers
        self.engineering_bay_y, self.engineering_bay_x = (self.unit_type == _TERRAN_ENGINEERING_BAY).nonzero()
        self.engineering_bay_built = True if self.engineering_bay_y.any() else False

        # 'ARGENT' ACTUEL
        self.mining_owned = obs.observation['player'][1]
        self.vespene_owned = obs.observation['player'][2]

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

        # Si pas de centre de command disponible ou le nombre de SCV >= 15 (à l'écran) ou nb total de worker > 20
        if self.cc_count == 0 or self.scv_count >= 15 or worker_supply > 20 or self.mining_owned < 50:
            excluded_actions.append(ACTION_ID_TRAIN_SCV)

        # Si nb de supply depot >= 6 ou pas de worker ou place disponible > 4
        if self.supply_depot_count >= 6 or worker_supply == 0 or supply_free > 4 or self.mining_owned < 100:
            excluded_actions.append(ACTION_ID_BUILD_SUPPLY_DEPOT)

        # Si pas de supply depot ou nb de barrack >= 2 ou pas de worker
        if self.supply_depot_count == 0 or self.barracks_count >= 2 or worker_supply == 0 or self.mining_owned < 150:
            excluded_actions.append(ACTION_ID_BUILD_BARRACKS)

        # Si pas de place ou pas de barrack
        if supply_free == 0 or self.barracks_count == 0 or self.mining_owned < 50:
            excluded_actions.append(ACTION_ID_TRAIN_MARINE)

        # Si pas de worker dispo ou déjà 2 rafinery ou TODO il faudrait vérifier que les deux minerais sont bien présent
        if worker_supply == 0 or self.refinery_count >= 2 or self.mining_owned < 75:
            excluded_actions.append(ACTION_ID_BUILD_REFINERY)

        # Si pas de worker inactif
        if inactiv_worker <= 0:
            excluded_actions.append(ACTION_ID_SCV_INACTIV_TO_MINE)

        # Si pas de supply ou pas de barrack ou pas de worker ou usine déja construite
        if \
                self.supply_depot_count == 0 or \
                self.barracks_count == 0 or \
                worker_supply == 0 or \
                self.engineering_bay_built is True or \
                self.mining_owned < 125:
            excluded_actions.append(ACTION_ID_BUILD_ENGINEERING_BAY)

        # Si pas de supply ou pas de barrack ou pas de worker ou tourelle missile construite ou usine pas construite
        if \
                self.supply_depot_count == 0 or \
                self.barracks_count == 0 or \
                worker_supply == 0 or \
                self.missile_turret_count > 0 is True or \
                self.engineering_bay_built is False or \
                self.mining_owned < 100:
            excluded_actions.append(ACTION_ID_BUILD_MISSILE_TURRET)

        # Si pas de supply ou pas de barrack ou pas de worker ou pas d'armée ou usine pas construite
        if self.supply_depot_count == 0 or \
                self.barracks_count == 0 or \
                worker_supply == 0 or \
                army_supply <= 2 or \
                self.engineering_bay_built is False or \
                self.mining_owned < 200:
            for action in ACTION_ID_DEFEND_POSITION:
                excluded_actions.append(action)

        # Si l'armée est inférieur à 8
        if army_supply <= 8:
            excluded_actions.append(ACTION_ID_ATTACK)

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

        # Ajout de la précèdente action et des scores y résultant dans la table Qlearning
        if self.previous_action is not None:
            self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(self.current_state))

        # Choix de la smart_action en fonction des éléments du jeu, si plus aucune action en cours
        if self.smart_action is None:
            self.init_move_number()
            excluded_actions = self.get_excluded_actions(obs)
            self.previous_state = self.current_state
            self.previous_action = self.qlearn.choose_action(str(self.current_state), excluded_actions)
            self.smart_action, self.smart_action_x, self.smart_action_y = self.split_action(self.previous_action)
            print(self.smart_action)
            print(self.previous_action)

        # smart action choisi par le qlearning ?
        if self.smart_action in ACTIONS_BUILD_BUILDING:
            return self.actions_build_building()
        elif self.smart_action in ACTIONS_TRAIN_UNIT:
            return self.actions_build_unit()
        elif self.smart_action == ACTION_DEFEND_POSITION:
            return self.action_defend_position()
        elif self.smart_action == ACTION_SCV_INACTIV_TO_MINE:
            return self.action_scv_inactiv_to_mine()
        elif self.smart_action == ACTION_ATTACK:
            return self.action_attack()
        elif self.smart_action == ACTION_DO_NOTHING:
            return self.action_do_nothing()
        elif self.smart_action == ACTION_ECONOMISE:
            return self.action_economise()
        return self.move_camera_to_base()

    def select_unit(self, unit):
        # Séléctionner un SCV
        if unit == "SCV":
            unit_y, unit_x = (self.unit_type == _TERRAN_SCV).nonzero()
            if unit_y.any() and _SELECT_POINT in self.obs.observation['available_actions']:
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                self.unit_selected = "SCV"
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        # Séléctionner un worker inactif
        elif unit == "IDLEWORKER":
            if _SELECT_IDLE_WORKER in self.obs.observation['available_actions']:
                self.unit_selected = "IDLEWORKER"
                return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
        # Séléctionner l'armée totale
        elif unit == "ARMY":
            if _SELECT_ARMY in self.obs.observation['available_actions']:
                self.unit_selected = "ARMY"
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        # Séléctionner un bunker
        elif unit == "BUNKER":
            if self.bunker_y.any() and _SELECT_POINT in self.obs.observation['available_actions']:
                i = random.randint(0, len(self.bunker_y) - 1)
                target = [self.bunker_x[i], self.bunker_y[i]]
                self.unit_selected = "BUNKER"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        # Séléctionner une caserne
        elif unit == "BARRACK":
            if self.barracks_y.any() and _SELECT_POINT in self.obs.observation['available_actions']:
                i = random.randint(0, len(self.barracks_y) - 1)
                target = [self.barracks_x[i], self.barracks_y[i]]
                self.unit_selected = "BARRACK"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        # Séléctionner le command center
        elif unit == "COMMANDCENTER":
            if self.cc_count > 0 and _SELECT_POINT in self.obs.observation['available_actions']:
                target = [int(self.cc_x.mean()), int(self.cc_y.mean())]
                self.unit_selected = "COMMANDCENTER"
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        return self.move_camera_to_base()

    # ACTIONS GLOBAL
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def move_camera_to_base(self):
        if _MOVE_CAMERA in self.obs.observation["available_actions"]:
            return actions.FUNCTIONS.move_camera([self.cc_minimap_x, self.cc_minimap_y])
        else:
            return actions.FunctionCall(_NO_OP, [])

    def end_action(self):
        self.smart_action = None
        self.init_move_number()

    def action_re_init_smart_action(self):
        self.end_action()
        return self.move_camera_to_base()

    def action_do_nothing(self):
        self.end_action()
        return actions.FunctionCall(_NO_OP, [])

    def action_economise(self):
        if self.mining_owned < 200 and self.mineral_restant > 0:
            return actions.FunctionCall(_NO_OP, [])
        else:
            return self.action_do_nothing()

    def action_scv_inactiv_to_mine(self):
        if self.unit_selected != "IDLEWORKER":
            return self.select_unit("IDLEWORKER")

        if self.move_number == 0:
            self.inc_move_number()
            # Déplacer le worker vers le minerai s'il est disponible à l'écran
            if _HARVEST_GATHER in self.obs.observation['available_actions'] and self.cc_count > 0:
                unit_y, unit_x = (self.unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)

                    m_x = unit_x[i]
                    m_y = unit_y[i]

                    target = [int(m_x), int(m_y)]
                    return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
            # Sinon le déplacer vers la base
            elif _MOVE_SCREEN in self.obs.observation['available_actions']:
                target = [self.cc_minimap_x, self.cc_minimap_y]
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_defend_position(self):
        if self.unit_selected != "SCV" and self.move_number == 0:
            return self.select_unit("SCV")

        # Déplacer la caméra vers le zone choisi par la smart action
        if self.move_number == 0:
            if _MOVE_CAMERA in self.obs.observation["available_actions"]:
                self.inc_move_number()
                return actions.FUNCTIONS.move_camera([self.smart_action_x, self.smart_action_y])
        # Construire dans la zone à un point random au moins 2 bunker et jusqu'à 2 tourelles
        elif self.move_number == 1:
            self.inc_move_number()
            x = random.randint(0, 83)
            y = random.randint(0, 83)
            if 0 <= self.bunker_count < 2 and _BUILD_BUNKER in self.obs.observation['available_actions']:
                return actions.FunctionCall(_BUILD_BUNKER, [_NOT_QUEUED, [x, y]])
            elif \
                    0 <= self.missile_turret_count < 2 and \
                    _BUILD_MISSILE_TURRET in self.obs.observation['available_actions']:
                return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, [x, y]])

        elif self.move_number == 2:
            army_supply = self.obs.observation['player'][5]
            # Si l'armée n'est pas séléctionnée
            if self.unit_selected != "ARMY" and self.move_number == 2 and army_supply > 0:
                return self.select_unit("ARMY")
            # Si l'armée est séléctionnée
            self.inc_move_number()
            if self.bunker_y.any() and \
                    _MOVE_SCREEN in self.obs.observation['available_actions'] and \
                    self.unit_selected == "ARMY":
                i = random.randint(0, len(self.bunker_y) - 1)
                target = [self.bunker_x[i], self.bunker_y[i]]
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
            elif _ATTACK_MINIMAP in self.obs.observation['available_actions'] and self.unit_selected == "ARMY":
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [self.smart_action_x, self.smart_action_y]])

        elif self.move_number == 3:
            # Séléctionner le bunker pour y faire rentrer un marine
            if self.unit_selected != "BUNKER" and self.move_number == 3 and self.bunker_y.any():
                return self.select_unit("BUNKER")
            self.inc_move_number()
            if _LOAD_BUNKER_SCREEN in self.obs.observation['available_actions'] and self.marine_count > 0:
                target = [int(round(self.marine_x.mean())), int(round(self.marine_y.mean()))]
                return actions.FunctionCall(_LOAD_BUNKER_SCREEN, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_attack(self):
        if self.unit_selected != "ARMY":
            return self.select_unit("ARMY")

        if _ATTACK_MINIMAP in self.obs.observation['available_actions']:
            self.end_action()
            if self.base_top_left:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
            return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])
        return self.action_re_init_smart_action()

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
        elif self.smart_action == ACTION_BUILD_REFINERY:
            return self.action_build_refinery()
        elif self.smart_action == ACTION_BUILD_MISSILE_TURRET:
            return self.action_build_missile_turret()
        else:
            return self.action_re_init_smart_action()

    def actions_build_unit(self):
        if self.smart_action == ACTION_TRAIN_SCV:
            return self.action_train_scv()
        elif self.smart_action == ACTION_TRAIN_MARINE:
            return self.action_train_marine()
        else:
            return self.action_re_init_smart_action()

    # ACTIONS BUILD
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
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
                    self.end_action()
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
                self.end_action()
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_missile_turret(self):
        if _BUILD_MISSILE_TURRET in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                target = self.transform_distance(
                    round(self.cc_screen_x.mean()),
                    0,
                    round(self.cc_screen_y.mean()),
                    15
                )
                self.end_action()
                return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_engineering_bay(self):
        if _BUILD_ENGINEERING_BAY in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                target = self.transform_distance(round(self.cc_screen_x.mean()), -8,
                                                 round(self.cc_screen_y.mean()), 15)
                self.end_action()
                return actions.FunctionCall(_BUILD_ENGINEERING_BAY, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_refinery(self):
        if _BUILD_REFINERY in self.obs.observation['available_actions']:
            if self.refinery_count <= 1:
                target = [round(self.vespene_x.mean()), round(self.vespene_y.mean())]
                self.end_action()
                return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    # ACTIONS TRAIN
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def action_train_scv(self):
        if self.unit_selected != "COMMANDCENTER":
            return self.select_unit("COMMANDCENTER")
        else:
            # Lancement construction du worker
            if _TRAIN_SCV in self.obs.observation["available_actions"]:
                self.end_action()
                return actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
        return self.action_re_init_smart_action()

    def action_train_marine(self):
        if self.unit_selected != "BARRACK":
            return self.select_unit("BARRACK")
        else:
            # Lancement construction du marine
            if _TRAIN_MARINE in self.obs.observation['available_actions']:
                self.end_action()
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        return self.action_re_init_smart_action()
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # FIN DES ACTIONS

    # Check by self hand some position
    def handself_checking(self):
        user_input = input("Enter  coord")
        try:
            if user_input.__contains__('x'):
                self.testx = int(user_input.replace('x', ''))
            if user_input.__contains__('y'):
                self.testy = int(user_input.replace('y', ''))
        except:
            pass
        print(self.testx, ', ', self.testy)
        if _MOVE_CAMERA in self.obs.observation["available_actions"]:
            return actions.FUNCTIONS.move_camera([self.testx, self.testy])
        else:
            return actions.FunctionCall(_NO_OP, [])


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
                    print(step_actions)
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass


def main():
    agent = SparseAgentDefensive()
    run(agent)


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    main()
