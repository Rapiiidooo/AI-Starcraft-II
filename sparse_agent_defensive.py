import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import pkg_resources
from absl import flags
from packaging import version
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions
from pysc2.lib import features
from sklearn.cluster import KMeans

from QLearningTable import QLearningTable

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
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_EFFECT_REPAIR_SCV_AUTOCAST = actions.FUNCTIONS.Effect_Repair_SCV_autocast.id
_MORPH_SUPPLYDEPOT_LOWER_QUICK = actions.FUNCTIONS.Morph_SupplyDepot_Lower_quick.id
_MORPH_SUPPLYDEPOT_RAISE_QUICK = actions.FUNCTIONS.Morph_SupplyDepot_Raise_quick.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_MISSILE_TURRET = actions.FUNCTIONS.Build_MissileTurret_screen.id
_BUILD_ENGINEERING_BAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Factory_screen.id
_BUILD_STARPORT = actions.FUNCTIONS.Build_Starport_screen.id
_BUILD_TECHLAB_STARPORT = actions.FUNCTIONS.Build_TechLab_Starport_quick.id
_BUILD_FUSION_CORE = actions.FUNCTIONS.Build_FusionCore_screen.id
_BUILD_BUNKER = actions.FUNCTIONS.Build_Bunker_screen.id
_LOAD_BUNKER_SCREEN = actions.FUNCTIONS.Load_Bunker_screen.id

_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_BATTLE_CRUISER = actions.FUNCTIONS.Train_Battlecruiser_quick.id

_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_Attack_screen.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_REFINERY = 20
_TERRAN_BARRACKS = 21
_TERRAN_ENGINEERING_BAY = 22
_TERRAN_MISSILE_TURRET = 23
_TERRAN_BUNKER = 24
_TERRAN_FACTORY = 27
_TERRAN_STARPORT = 28
_TERRAN_FUSION_CORE = 30
_TERRAN_STARPORT_TECHLAB = 41
_TERRAN_SCV = 45
_TERRAN_MARINE = 48
_TERRAN_BATTLE_CRUISER = 57

_NEUTRAL_MINERAL_FIELD = 341
_NEUTRAL_VESPENE_GEYSER = 342

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'defensive_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_TRAIN_SCV = 'trainscv'
ACTION_TRAIN_MARINE = 'trainmarine'
ACTION_TRAIN_BATTLE_CRUISER = 'trainbattlecruiser'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MISSILE_TURRET = 'buildmissileturret'
ACTION_BUILD_ENGINEERING_BAY = 'buildengineeringbay'
ACTION_BUILD_REFINERY = 'buildrefinery'
ACTION_BUILD_FACTORY = 'buildfactory'
ACTION_BUILD_STARPORT = 'buildstarpor'
ACTION_BUILD_FUSION_CORE = 'buildfusioncore'
ACTION_UPGRADE_STARPORT_TECHLAB = 'upgradestarportechlab'
ACTION_ECONOMISE = 'economise'
ACTION_SCV_TO_VESPENE = 'workertovespene'
ACTION_SCV_INACTIV_TO_MINE = 'reactiveworker'
ACTION_DEFEND_POSITION = 'defend'
ACTION_DEFEND_VS_ENEMY = 'defendvsenemy'
ACTION_ATTACK = 'attack'
ACTION_SUPPLY_DEPOT_RAISE_QUICK = 'supplyraiseup'

# The order of ACTION ID AND IN SMART ACTION ARE VERY IMPORTANT
ACTION_ID_DO_NOTHING = 0
ACTION_ID_TRAIN_SCV = 1
ACTION_ID_TRAIN_MARINE = 2
ACTION_ID_TRAIN_BATTLE_CRUISER = 3
ACTION_ID_BUILD_SUPPLY_DEPOT = 4
ACTION_ID_BUILD_BARRACKS = 5
ACTION_ID_BUILD_MISSILE_TURRET = 6
ACTION_ID_BUILD_ENGINEERING_BAY = 7
ACTION_ID_BUILD_REFINERY = 8
ACTION_ID_BUILD_FACTORY = 9
ACTION_ID_BUILD_STARPORT = 10
ACTION_ID_BUILD_FUSION_CORE = 11
ACTION_ID_UPGRADE_STARPORT_TECHLAB = 12
ACTION_ID_ECONOMISE = 13
ACTION_ID_SCV_TO_VESPENE = 14
ACTION_ID_SCV_INACTIV_TO_MINE = 15
ACTION_ID_DEFEND_POSITION = 16
ACTION_ID_DEFEND_VS_ENEMY = 17
ACTION_ID_ATTACK = 18
ACTION_ID_SUPPLY_DEPOT_RAISE_QUICK = 19

SMART_ACTIONS = [
    ACTION_DO_NOTHING,
    ACTION_TRAIN_SCV,
    ACTION_TRAIN_MARINE,
    ACTION_TRAIN_BATTLE_CRUISER,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MISSILE_TURRET,
    ACTION_BUILD_ENGINEERING_BAY,
    ACTION_BUILD_REFINERY,
    ACTION_BUILD_FACTORY,
    ACTION_BUILD_STARPORT,
    ACTION_BUILD_FUSION_CORE,
    ACTION_UPGRADE_STARPORT_TECHLAB,
    ACTION_ECONOMISE,
    ACTION_SCV_TO_VESPENE,
    ACTION_SCV_INACTIV_TO_MINE,
    ACTION_DEFEND_POSITION,
    ACTION_DEFEND_VS_ENEMY,
    ACTION_ATTACK,
    ACTION_SUPPLY_DEPOT_RAISE_QUICK
]

ACTIONS_BUILD_BUILDING = [
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MISSILE_TURRET,
    ACTION_BUILD_ENGINEERING_BAY,
    ACTION_BUILD_REFINERY,
    ACTION_BUILD_FACTORY,
    ACTION_BUILD_STARPORT,
    ACTION_BUILD_FUSION_CORE,
    ACTION_UPGRADE_STARPORT_TECHLAB
]

ACTIONS_TRAIN_UNIT = [
    ACTION_TRAIN_SCV,
    ACTION_TRAIN_MARINE,
    ACTION_TRAIN_BATTLE_CRUISER
]

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
        self.vespene_center = None
        self.refinery_y = None
        self.refinery_x = None
        self.refinery_count = None
        self.engineering_bay_y = None
        self.engineering_bay_x = None
        self.engineering_bay_built = None
        self.mineral_restant = -1
        self.mineral_x = None
        self.mineral_y = None
        self.scv_in_vespene1 = 0
        self.scv_in_vespene2 = 0
        self.vespene_step_acc = 0
        self.vespene_owned_prev = 0
        self.factory_count = None
        self.factory_x = None
        self.factory_y = None
        self.starport_count = None
        self.starport_x = None
        self.starport_y = None
        self.starport_techlab_count = None
        self.starport_techlab_x = None
        self.starport_techlab_y = None
        self.fusion_core_count = None
        self.fusion_core_x = None
        self.fusion_core_y = None
        self.rally_unit_starport = False
        self.rally_unit_barracks = False
        self.target_rally_unit_minimap = None
        self.battle_cruiser_built = 0
        self.exclude_build_supply_depot = False
        self.anti_zerg_rush_wall = False
        self.supply_downed = False
        # Info actions
        self.smart_action = None
        self.smart_action_x = None
        self.smart_action_y = None
        self.previous_action = None
        self.previous_state = None
        self.move_number = 0
        # Info general
        self.obs = None
        self.player_relative = None
        self.unit_type = None
        self.current_state = np.zeros(20)
        self.mining_owned = None
        self.vespene_owned = None
        self.target_enemis = None
        self.enemy_x = None
        self.enemy_y = None
        # Variable pour tester la position à la main
        self.testx = 80
        self.testy = 0
        self.essai = 0

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
            self.target_rally_unit_minimap = [27, 22]
        else:
            self.cc_minimap_x = 40
            self.cc_minimap_y = 47
            self.target_rally_unit_minimap = [30, 45]

        # VESPENE présent
        self.vespene_y, self.vespene_x = (self.unit_type == _NEUTRAL_VESPENE_GEYSER).nonzero()
        self.vespene_geyser_count = int(math.ceil(len(self.vespene_y) / 97))
        units = []
        for i in range(0, len(self.vespene_y)):
            units.append((self.vespene_x[i], self.vespene_y[i]))
        kmeans = KMeans(n_clusters=self.vespene_geyser_count)
        kmeans.fit(units)
        self.vespene_center = kmeans.cluster_centers_

    def transform_distance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    @staticmethod
    def split_action(action_id):
        smart_action = SMART_ACTIONS[action_id]
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
        return smart_action, int(x), int(y)

    @staticmethod
    def _xy_locs(mask):
        """Mask should be a set of bools from comparison with a feature layer.
        Function from https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py
        """
        y, x = mask.nonzero()
        return list(zip(x, y))

    def ending_game(self, obs):
        # reward = obs.reward
        reward = obs.reward

        self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', compression='gzip')

        self.previous_action = None
        self.previous_state = None
        self.move_number = 0

        save_score(obs.observation["score_cumulative"]["score"], self.steps, reward)
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

        # NB de vespene geyser restant
        self.vespene_y, self.vespene_x = (self.unit_type == _NEUTRAL_VESPENE_GEYSER).nonzero()
        self.vespene_geyser_count = int(math.ceil(len(self.vespene_y) / 97))

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

        # NB de RAFINERIE
        self.refinery_y, self.refinery_x = (self.unit_type == _TERRAN_REFINERY).nonzero()
        self.refinery_count = int(round(len(self.refinery_y) / 97))

        # NB d'USINE
        self.factory_y, self.factory_x = (self.unit_type == _TERRAN_FACTORY).nonzero()
        self.factory_count = int(round(len(self.factory_y) / 97))

        # NB de STARPORT
        self.starport_y, self.starport_x = (self.unit_type == _TERRAN_STARPORT).nonzero()
        self.starport_count = int(round(len(self.starport_y) / 97))

        # NB de NOYAU DE FUSION
        self.fusion_core_y, self.fusion_core_x = (self.unit_type == _TERRAN_FUSION_CORE).nonzero()
        self.fusion_core_count = int(round(len(self.fusion_core_y) / 97))

        # NB de STARPORT AMELIORE
        self.starport_techlab_y, self.starport_techlab_x = (self.unit_type == _TERRAN_STARPORT_TECHLAB).nonzero()
        self.starport_techlab_count = int(round(len(self.starport_techlab_y) / 97))

        # NB de centre d'usine (pour débloquer la construction des bunkers
        self.engineering_bay_y, self.engineering_bay_x = (self.unit_type == _TERRAN_ENGINEERING_BAY).nonzero()
        self.engineering_bay_built = True if self.engineering_bay_y.any() else False

        # 'ARGENT' ACTUEL
        self.mining_owned = obs.observation['player'][1]
        self.vespene_owned = obs.observation['player'][2]
        self.vespene_step_acc += 1
        # Si ya plus de 80 step et que des ouvriers sont rescencé
        if self.vespene_step_acc > 80 and self.scv_in_vespene1 + self.scv_in_vespene2 > 2:
            # si le vespene précèdent = 0 c'est l'init ...
            if self.vespene_owned_prev == 0:
                self.vespene_owned_prev = self.vespene_owned
            # si vespene prev == actuel depuis 80 step = surement plus de worker dessus
            elif self.vespene_owned_prev == self.vespene_owned:
                self.scv_in_vespene1 = 0
                self.scv_in_vespene2 = 0
            # surement batiment construit
            elif self.vespene_owned_prev > self.vespene_owned:
                self.vespene_owned_prev = self.vespene_owned
            # le vespene n'a pas augmenté beaucoup
            elif (self.vespene_owned - self.vespene_owned_prev) <= 40:
                self.scv_in_vespene1 = 1
                self.scv_in_vespene2 = 1
            # sinon attribuer la valeur actuel en mémoire
            else:
                self.vespene_owned_prev = self.vespene_owned
            self.vespene_step_acc = 0

        self.current_state = np.zeros(20)
        self.current_state[0] = self.cc_count
        self.current_state[1] = self.supply_depot_count
        self.current_state[2] = self.barracks_count
        self.current_state[3] = self.starport_count
        self.current_state[4] = self.starport_techlab_count
        self.current_state[5] = self.fusion_core_count
        self.current_state[6] = self.factory_count
        self.current_state[7] = self.refinery_count
        self.current_state[8] = self.vespene_geyser_count
        self.current_state[9] = self.scv_count
        self.current_state[10] = self.battle_cruiser_built
        self.current_state[11] = obs.observation['player'][_ARMY_SUPPLY]
        self.current_state[12] = obs.observation["score_cumulative"]["score"]

        hot_squares = np.zeros(4)
        self.enemy_y, self.enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(self.enemy_y)):
            y = int(math.ceil((self.enemy_y[i] + 1) / 32))
            x = int(math.ceil((self.enemy_x[i] + 1) / 32))
            hot_squares[((y - 1) * 2) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 4):
            self.current_state[i + 12] = hot_squares[i]

        green_squares = np.zeros(4)
        friendly_y, friendly_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        for i in range(0, len(friendly_y)):
            y = int(math.ceil((friendly_y[i] + 1) / 32))
            x = int(math.ceil((friendly_x[i] + 1) / 32))
            green_squares[((y - 1) * 2) + (x - 1)] = 1

        if not self.base_top_left:
            green_squares = green_squares[::-1]

        for i in range(0, 4):
            self.current_state[i + 16] = green_squares[i]

        # Détéction d'ennemis à l'écran
        self.player_relative = self.obs.observation.feature_screen.player_relative
        enemies = self._xy_locs(self.player_relative == _PLAYER_ENEMY)
        if enemies:
            # noinspection PyTypeChecker
            self.target_enemis = enemies[np.argmax(np.array(enemies)[:, 1])]
        else:
            self.target_enemis = None

    def get_excluded_actions(self, obs):
        excluded_actions = []
        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        # army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        army_count = obs.observation['player'][8]
        supply_free = supply_limit - supply_used
        inactiv_worker = obs.observation['player'][7]

        # Si pas de centre de command disponible ou le nombre de SCV >= 15 (à l'écran) ou nb total de worker > 20
        if self.cc_count == 0 or self.scv_count >= 20 or worker_supply > 22 or self.mining_owned < 50 or \
                self.mineral_restant <= 0:
            excluded_actions.append(ACTION_ID_TRAIN_SCV)

        # Si pas de worker ou place disponible > 4
        if self.supply_depot_count >= 6 or worker_supply == 0 or supply_free > 17 or self.mining_owned < 100 or \
                self.exclude_build_supply_depot is True:
            excluded_actions.append(ACTION_ID_BUILD_SUPPLY_DEPOT)

        # Si pas de mur de supply depot ou nb de barrack >= 2 ou pas de worker
        if supply_limit <= 30 or self.barracks_count >= 2 or worker_supply == 0 or self.mining_owned < 150:
            excluded_actions.append(ACTION_ID_BUILD_BARRACKS)

        # Si pas de place ou pas de barrack ou déjà beacuoup de marine
        if supply_free <= 0 or self.barracks_count == 0 or self.marine_count >= 15 or self.mining_owned < 50:
            excluded_actions.append(ACTION_ID_TRAIN_MARINE)

        # Si pas de place ou pas de starport amélioré
        if supply_free <= 3 or self.starport_techlab_count == 0 or self.mining_owned < 400 or self.vespene_owned < 300:
            excluded_actions.append(ACTION_ID_TRAIN_BATTLE_CRUISER)

        # Si pas de worker dispo ou déjà rafinery présentes sur les vespene restant
        if worker_supply == 0 or self.refinery_count >= self.vespene_geyser_count or self.mining_owned < 75:
            excluded_actions.append(ACTION_ID_BUILD_REFINERY)

        if self.refinery_count <= 0 or worker_supply == 0 or (self.scv_in_vespene1 >= 3 and self.scv_in_vespene2 >= 3):
            # Si plus de minerai mais rafinerie > 0 et worker > 0 autoriser quand même l'action
            if self.mineral_restant <= 0 and worker_supply > 0 and self.refinery_count > 0:
                pass
            else:
                excluded_actions.append(ACTION_ID_SCV_TO_VESPENE)

        # Si pas de worker inactif
        if inactiv_worker <= 0 or self.mineral_restant <= 0:
            excluded_actions.append(ACTION_ID_SCV_INACTIV_TO_MINE)

        # Si pas de supply ou pas de barrack ou pas de worker ou usine déja construite
        if \
                supply_limit <= 15 or \
                self.barracks_count == 0 or \
                worker_supply == 0 or \
                self.engineering_bay_built is True or \
                self.mining_owned < 125:
            excluded_actions.append(ACTION_ID_BUILD_ENGINEERING_BAY)

        # Si pas de supply ou pas de barrack ou pas de worker ou tourelle missile construite ou usine pas construite
        if \
                supply_limit <= 15 or \
                self.barracks_count == 0 or \
                worker_supply == 0 or \
                self.missile_turret_count > 0 or \
                self.engineering_bay_built is False or \
                self.mining_owned < 100:
            excluded_actions.append(ACTION_ID_BUILD_MISSILE_TURRET)

        # Si déjà une usine ou pas de caserne ou pas au moins 150 minerais ou 100 vespene
        if self.factory_count > 0 or self.barracks_count == 0 or self.mining_owned < 150 or self.vespene_owned < 100:
            excluded_actions.append(ACTION_ID_BUILD_FACTORY)

        # Si pas d'usine ou si déjà un starport pas au moins 150 minerais
        if self.factory_count == 0 or \
                self.starport_count > 0 or \
                self.starport_techlab_count > 0 or \
                self.mining_owned < 100:
            excluded_actions.append(ACTION_ID_BUILD_STARPORT)

        # Si pas de starport ou pas au moins 150 minerais ou 150 vespene
        if self.starport_count == 0 or self.mining_owned < 150 or self.vespene_owned < 150:
            excluded_actions.append(ACTION_ID_BUILD_FUSION_CORE)

        # Si pas de noyau de fusion ou pas de starport au moins 50 minerais ou 25 vespene
        if self.fusion_core_count == 0 or \
                self.starport_count == 0 or \
                self.mining_owned < 50 or \
                self.vespene_owned < 25:
            excluded_actions.append(ACTION_ID_UPGRADE_STARPORT_TECHLAB)

        # Si pas de supply ou pas de barrack ou pas de worker ou pas d'armée ou usine pas construite
        if supply_limit <= 15 or \
                self.barracks_count == 0 or \
                worker_supply == 0 or \
                army_count <= 3 or \
                self.engineering_bay_built is False or \
                self.mining_owned < 200 or \
                self.mineral_restant <= 0:
            # for action in ACTION_ID_DEFEND_POSITION:
            #     excluded_actions.append(action)
            excluded_actions.append(ACTION_ID_DEFEND_POSITION)

        # Autoriser l'attaque si
        # armée > 0 ET plus de minerai ET argent restant < 200
        # armée >= 25
        # armée >= 10 ET hypérion >=3
        if (army_count > 0 and self.mineral_restant <= 0 and self.mining_owned < 200) or \
                army_count > 20 or self.battle_cruiser_built >= 3:
            pass
        else:
            # Sinon l'attaque est exclue
            excluded_actions.append(ACTION_ID_ATTACK)

        # Si les supplys dépots du mur ont étés baissés
        if not self.supply_downed:
            excluded_actions.append(ACTION_ID_SUPPLY_DEPOT_RAISE_QUICK)

        # Si pas d'ennemis à l'écran ou écran pas sur command center
        if self.target_enemis is None or self.cc_count == 0 or army_count > 0 or worker_supply <= 0:
            excluded_actions.append(ACTION_ID_DEFEND_VS_ENEMY)

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
            self.init_current_state(obs)
            return self.action_re_init_smart_action()

        # Initialisation de l'état actuelle des éléments du joueurs.
        self.init_current_state(obs)

        self.essai += 1
        if self.essai <= 1:
            return actions.FunctionCall(_NO_OP, [])
        else:
            self.essai = 0

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

        # smart action choisi par le qlearning ?
        if self.smart_action in ACTIONS_BUILD_BUILDING:
            return self.actions_build_building()
        elif self.smart_action in ACTIONS_TRAIN_UNIT:
            return self.actions_train_unit()
        elif self.smart_action == ACTION_DEFEND_POSITION:
            return self.action_defend_position()
        elif self.smart_action == ACTION_SCV_INACTIV_TO_MINE:
            return self.action_scv_inactiv_to_mine()
        elif self.smart_action == ACTION_SCV_TO_VESPENE:
            return self.action_scv_to_vespene()
        elif self.smart_action == ACTION_ATTACK:
            return self.action_attack()
        elif self.smart_action == ACTION_DO_NOTHING:
            return self.action_do_nothing()
        elif self.smart_action == ACTION_ECONOMISE:
            return self.action_economise()
        elif self.smart_action == ACTION_SUPPLY_DEPOT_RAISE_QUICK:
            return self.action_supply_depot_raise_quick()
        elif self.smart_action == ACTION_DEFEND_VS_ENEMY:
            return self.action_defend_vs_enemy()
        return self.action_re_init_smart_action()

    def select_unit(self, unit):
        # Séléctionner un SCV
        if unit == "SCV":
            unit_y, unit_x = (self.unit_type == _TERRAN_SCV).nonzero()
            if unit_y.any() and _SELECT_POINT in self.obs.observation['available_actions']:
                i = random.randint(0, abs(len(unit_y) - 1))
                target = [unit_x[i], unit_y[i]]
                self.unit_selected = "SCV"
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        # Séléctionner un SCV
        if unit == "ALLSCV":
            unit_y, unit_x = (self.unit_type == _TERRAN_SCV).nonzero()
            if unit_y.any() and _SELECT_POINT in self.obs.observation['available_actions']:
                i = random.randint(0, abs(len(unit_y) - 1))
                target = [unit_x[i], unit_y[i]]
                self.unit_selected = "ALLSCV"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
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
                i = random.randint(0, abs(len(self.bunker_y) - 1))
                target = [self.bunker_x[i], self.bunker_y[i]]
                self.unit_selected = "BUNKER"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        # Séléctionner une caserne
        elif unit == "BARRACK":
            if self.barracks_y.any() and _SELECT_POINT in self.obs.observation['available_actions']:
                i = random.randint(0, abs(len(self.barracks_y) - 1))
                target = [self.barracks_x[i], self.barracks_y[i]]
                self.unit_selected = "BARRACK"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
        # Séléctionner le command center
        elif unit == "COMMANDCENTER":
            if self.cc_count > 0 and _SELECT_POINT in self.obs.observation['available_actions']:
                target = [int(self.cc_x.mean()), int(self.cc_y.mean())]
                self.unit_selected = "COMMANDCENTER"
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        # Séléctionner le starport
        elif unit == "STARPORT":
            if self.starport_count > 0 and _SELECT_POINT in self.obs.observation['available_actions']:
                target = [int(self.starport_x.mean()), int(self.starport_y.mean())]
                self.unit_selected = "STARPORT"
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        # Séléctionner le starport amélioré
        elif unit == "STARPORTTECHLAB":
            if self.starport_techlab_count > 0 and _SELECT_POINT in self.obs.observation['available_actions']:
                target = [int(self.starport_techlab_x.mean()), int(self.starport_techlab_y.mean())]
                self.unit_selected = "STARPORTTECHLAB"
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        # Séléctionner une caserne
        elif unit == "SUPPLYDEPOT":
            if self.depot_y.any() and _SELECT_POINT in self.obs.observation['available_actions']:
                i = random.randint(0, abs(len(self.depot_y) - 1))
                target = [self.depot_x[i], self.depot_y[i]]
                self.unit_selected = "SUPPLYDEPOT"
                return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
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
        if self.move_number == 0:
            self.inc_move_number()
            return self.move_camera_to_base()
        else:
            if self.mining_owned < 200 and self.mineral_restant > 0:
                return actions.FunctionCall(_NO_OP, [])
            else:
                return self.action_do_nothing()

    def action_scv_inactiv_to_mine(self):
        if self.unit_selected != "IDLEWORKER":
            return self.select_unit("IDLEWORKER")

        if self.move_number == 0:
            self.inc_move_number()
            return self.move_camera_to_base()
        elif self.move_number == 1:
            self.inc_move_number()
            # Déplacer le worker vers le minerai s'il est disponible à l'écran
            if _HARVEST_GATHER in self.obs.observation['available_actions'] and self.cc_count > 0:
                unit_y, unit_x = (self.unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                if unit_y.any():
                    i = random.randint(0, abs(len(unit_y) - 1))
                    m_x = unit_x[i]
                    m_y = unit_y[i]

                    target = [int(m_x), int(m_y)]
                    return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
            # Sinon le déplacer vers la base
            elif _MOVE_SCREEN in self.obs.observation['available_actions']:
                target = [self.cc_minimap_x, self.cc_minimap_y]
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_scv_to_vespene(self):
        if self.unit_selected != "SCV":
            return self.select_unit("SCV")
        elif self.move_number == 0:
            if _HARVEST_GATHER in self.obs.observation['available_actions']:
                if self.refinery_count == 1 or self.scv_in_vespene1 < 3:
                    if version.parse(_PYSC2_VERSION) > version.parse('2.0.1'):
                        rand = random.randint(0, abs(len(self.vespene_y) - 1))
                        target = [int(self.vespene_x[rand]), int(self.vespene_y[rand])]
                    else:
                        target = [int(self.vespene_center[0][0]), int(self.vespene_center[0][1])]
                    self.scv_in_vespene1 += 1
                    self.inc_move_number()
                    return actions.FunctionCall(_HARVEST_GATHER, [_NOT_QUEUED, target])
                elif self.refinery_count == 2 or self.scv_in_vespene1 < 3:
                    if version.parse(_PYSC2_VERSION) > version.parse('2.0.1'):
                        rand = random.randint(0, abs(len(self.vespene_y) - 1))
                        target = [int(self.vespene_x[rand]), int(self.vespene_y[rand])]
                    else:
                        target = [int(self.vespene_center[1][0]), int(self.vespene_center[1][1])]
                    self.scv_in_vespene2 += 1
                    self.inc_move_number()
                    return actions.FunctionCall(_HARVEST_GATHER, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_defend_position(self):
        if self.unit_selected != "SCV" and self.move_number == 0:
            return self.select_unit("SCV")

        # Activer le mode réparation auto du SCV
        if self.move_number == 0:
            self.inc_move_number()
            if _EFFECT_REPAIR_SCV_AUTOCAST in self.obs.observation["available_actions"]:
                return actions.FunctionCall(_EFFECT_REPAIR_SCV_AUTOCAST, [])

        # Déplacer la caméra vers le zone choisi par la smart action
        if self.move_number == 1:
            if _MOVE_CAMERA in self.obs.observation["available_actions"]:
                self.inc_move_number()
                return actions.FUNCTIONS.move_camera(self.target_rally_unit_minimap)
        # Construire dans la zone à un point random jusqu'à 4 bunker et jusqu'à 4 tourelles
        elif self.move_number == 2:
            self.inc_move_number()
            if self.base_top_left:
                x = random.randint(30, 70)
                y = random.randint(10, 70)
            else:
                x = random.randint(30, 58)
                y = random.randint(22, 75)

            action = random.randint(0, 6)

            if 0 <= action < 5 and self.bunker_count < 4 and _BUILD_BUNKER in self.obs.observation['available_actions']:
                return actions.FunctionCall(_BUILD_BUNKER, [_NOT_QUEUED, [x, y]])
            elif self.missile_turret_count < 4 and _BUILD_MISSILE_TURRET in self.obs.observation['available_actions']:
                    return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, [x, y]])
            return self.action_do_nothing()

        elif self.move_number == 3:
            army_supply = self.obs.observation['player'][5]
            # Si l'armée n'est pas séléctionnée
            if self.unit_selected != "ARMY" and army_supply > 0:
                return self.select_unit("ARMY")
            # Si l'armée est séléctionnée
            self.inc_move_number()
            if self.bunker_y.any() and \
                    _MOVE_SCREEN in self.obs.observation['available_actions'] and \
                    self.unit_selected == "ARMY":
                i = random.randint(0, abs(len(self.bunker_y) - 1))
                target = [self.bunker_x[i], self.bunker_y[i]]
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
            elif _MOVE_MINIMAP in self.obs.observation['available_actions'] and self.unit_selected == "ARMY":
                return actions.FunctionCall(_MOVE_MINIMAP, [_NOT_QUEUED, self.target_rally_unit_minimap])

        elif self.move_number == 4:
            # Séléctionner le bunker pour y faire rentrer un marine
            if self.unit_selected != "BUNKER" and self.bunker_y.any():
                return self.select_unit("BUNKER")
            self.inc_move_number()
            if _LOAD_BUNKER_SCREEN in self.obs.observation['available_actions'] and self.marine_count > 0:
                target = [int(round(self.marine_x.mean())), int(round(self.marine_y.mean()))]
                return actions.FunctionCall(_LOAD_BUNKER_SCREEN, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_attack(self):
        if self.move_number == 0:
            if _MOVE_CAMERA in self.obs.observation["available_actions"]:
                self.inc_move_number()
                return actions.FUNCTIONS.move_camera(self.target_rally_unit_minimap)

        elif self.move_number == 1:
            if self.supply_downed:
                self.inc_move_number()
                return self.move_camera_to_base()
            elif self.unit_selected != "SUPPLYDEPOT":
                return self.select_unit("SUPPLYDEPOT")
            else:
                self.inc_move_number()
                if _MORPH_SUPPLYDEPOT_LOWER_QUICK in self.obs.observation['available_actions']:
                    self.supply_downed = True
                    return actions.FunctionCall(_MORPH_SUPPLYDEPOT_LOWER_QUICK, [_QUEUED])

        elif self.move_number == 2:
            if self.unit_selected != "ARMY":
                return self.select_unit("ARMY")
            self.inc_move_number()
            if _ATTACK_MINIMAP in self.obs.observation['available_actions']:
                # Si un enemie est présent sur la minimap alors qu'il reste presque plus d'ennemies visible
                if 0 < len(self.enemy_y) < 30:
                    random_choice = random.randint(0, abs(len(self.enemy_y) - 1))
                    target = [self.enemy_x[random_choice], self.enemy_y[random_choice]]
                # Sinon position stratégique pseudo-aléatoire
                else:
                    random_choice = random.randint(0, 1)
                    # Position de base principales
                    if random_choice == 0:
                        if self.base_top_left:
                            rand_x = random.randint(36, 42)
                            rand_y = random.randint(42, 48)
                            target = [rand_x, rand_y]
                        else:
                            rand_x = random.randint(18, 24)
                            rand_y = random.randint(21, 27)
                            target = [rand_x, rand_y]
                    # Position de base middle
                    else:
                        random_choice = random.randint(0, 1)
                        if random_choice == 0:
                            rand_x = random.randint(15, 21)
                            rand_y = random.randint(47, 53)
                            target = [rand_x, rand_y]
                        else:
                            rand_x = random.randint(36, 42)
                            rand_y = random.randint(17, 23)
                            target = [rand_x, rand_y]
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_supply_depot_raise_quick(self):
        if self.move_number == 0:
            if _MOVE_CAMERA in self.obs.observation["available_actions"]:
                self.inc_move_number()
                return actions.FUNCTIONS.move_camera(self.target_rally_unit_minimap)

        elif self.move_number == 1:
            if self.unit_selected != "SUPPLYDEPOT" and self.supply_depot_count > 0:
                return self.select_unit("SUPPLYDEPOT")
            else:
                self.inc_move_number()
                if _MORPH_SUPPLYDEPOT_RAISE_QUICK in self.obs.observation['available_actions']:
                    self.supply_downed = False
                    return actions.FunctionCall(_MORPH_SUPPLYDEPOT_RAISE_QUICK, [_QUEUED])
        return self.action_re_init_smart_action()

    def action_defend_vs_enemy(self):
        if self.unit_selected != "ALLSCV":
            return self.select_unit("ALLSCV")
        if self.move_number == 0:
            self.inc_move_number()
            if _ATTACK_SCREEN in self.obs.observation['available_actions'] and self.target_enemis is not None:
                return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, self.target_enemis])
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
        elif self.smart_action == ACTION_BUILD_FACTORY:
            return self.action_build_factory()
        elif self.smart_action == ACTION_BUILD_STARPORT:
            return self.action_build_starport()
        elif self.smart_action == ACTION_BUILD_FUSION_CORE:
            return self.action_build_fusion_core()
        elif self.smart_action == ACTION_UPGRADE_STARPORT_TECHLAB:
            return self.action_upgrade_starport_techlab()
        else:
            return self.action_re_init_smart_action()

    def actions_train_unit(self):
        if self.smart_action == ACTION_TRAIN_SCV:
            return self.action_train_scv()
        elif self.smart_action == ACTION_TRAIN_MARINE:
            return self.action_train_marine()
        elif self.smart_action == ACTION_TRAIN_BATTLE_CRUISER:
            return self.action_train_battle_cruiser()
        else:
            return self.action_re_init_smart_action()

    # ACTIONS BUILD
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def action_build_supply_depot(self):
        # Déplacer la caméra vers l'endroit à bloquer
        if self.move_number == 1:
            if _MOVE_CAMERA in self.obs.observation["available_actions"]:
                self.inc_move_number()
                return actions.FUNCTIONS.move_camera(self.target_rally_unit_minimap)
            else:
                return self.action_re_init_smart_action()

        # Construire le mur de dépôt de ravitaillement
        elif self.move_number == 2:
            if _BUILD_SUPPLY_DEPOT in self.obs.observation['available_actions']:
                coord_supply_y, coord_supply_x = (self.unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
                coord_supply = []
                for i, val in enumerate(coord_supply_x):
                    coord_supply.append([coord_supply_x[i], coord_supply_y[i]])
                if self.base_top_left:
                    # Si première fois ou mur cassé
                    if self.supply_depot_count == 0 or \
                            (self.anti_zerg_rush_wall is True and [46, 34] not in coord_supply):
                        self.testx += 1
                        target = [46, 34]
                    # Si première fois ou mur cassé
                    elif self.supply_depot_count == 1 or \
                            (self.anti_zerg_rush_wall is True and [50, 42] not in coord_supply):
                        target = [50, 42]
                    # Si première fois ou mur cassé
                    elif self.supply_depot_count == 2 or \
                            (self.anti_zerg_rush_wall is True and [58, 46] not in coord_supply):
                        target = [58, 46]
                    else:
                        self.anti_zerg_rush_wall = True
                        self.inc_move_number()
                        return self.move_camera_to_base()
                    self.move_number = 10
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
                else:
                    # Si première fois ou mur cassé
                    if self.supply_depot_count == 0 or \
                            (self.anti_zerg_rush_wall is True and [40, 56] not in coord_supply):
                        target = [40, 56]
                    # Si première fois ou mur cassé
                    elif self.supply_depot_count == 1 or \
                            (self.anti_zerg_rush_wall is True and [36, 48] not in coord_supply):
                        target = [36, 48]
                    # Si première fois ou mur cassé
                    elif self.supply_depot_count == 2 or \
                            (self.anti_zerg_rush_wall is True and [30, 44] not in coord_supply):
                        target = [30, 44]
                    else:
                        self.anti_zerg_rush_wall = True
                        self.inc_move_number()
                        return self.move_camera_to_base()
                    self.move_number = 10
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        # Si mur déja fait alors construire à côté de la base les supply depots
        elif self.move_number == 3:
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
                        self.move_number = 10
                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
                    else:
                        self.exclude_build_supply_depot = True
        # Repositionner la caméra à la base pour la fin d'action du mur
        elif self.move_number == 10:
            self.end_action()
            return self.move_camera_to_base()
        return self.action_re_init_smart_action()

    def action_build_barracks(self):
        if _BUILD_BARRACKS in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                if self.barracks_count == 0:
                    if self.base_top_left:
                        target = [59, 30]
                    else:
                        target = [24, 40]
                else:
                    if self.base_top_left:
                        target = [59, 40]
                    else:
                        target = [24, 30]
                self.end_action()
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_missile_turret(self):
        if _BUILD_MISSILE_TURRET in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                if self.base_top_left:
                    target = [48, 48]
                else:
                    target = [33, 18]
                self.end_action()
                return actions.FunctionCall(_BUILD_MISSILE_TURRET, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_engineering_bay(self):
        if _BUILD_ENGINEERING_BAY in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                if self.base_top_left:
                    target = [40, 50]
                else:
                    target = [42, 18]
                self.end_action()
                return actions.FunctionCall(_BUILD_ENGINEERING_BAY, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_refinery(self):
        if _BUILD_REFINERY in self.obs.observation['available_actions']:
            if version.parse(_PYSC2_VERSION) > version.parse('2.0.1'):
                rand = random.randint(0, abs(len(self.vespene_y) - 1))
                target = [int(self.vespene_x[rand]), int(self.vespene_y[rand])]
            else:
                if self.refinery_count <= 0:
                    target = [int(self.vespene_center[0][0]), int(self.vespene_center[0][1])]
                else:
                    target = [int(self.vespene_center[1][0]), int(self.vespene_center[1][1])]
            self.end_action()
            return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_factory(self):
        if _BUILD_FACTORY in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                if self.base_top_left:
                    target = [40, 60]
                else:
                    target = [42, 8]
                self.end_action()
                return actions.FunctionCall(_BUILD_FACTORY, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_starport(self):
        if _BUILD_STARPORT in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                if self.base_top_left:
                    target = [59, 50]
                else:
                    target = [16, 18]
                self.end_action()
                return actions.FunctionCall(_BUILD_STARPORT, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_build_fusion_core(self):
        if _BUILD_FUSION_CORE in self.obs.observation['available_actions']:
            if self.cc_screen_y.any():
                if self.base_top_left:
                    target = [20, 60]
                else:
                    target = [62, 8]
                self.end_action()
                return actions.FunctionCall(_BUILD_FUSION_CORE, [_NOT_QUEUED, target])
        return self.action_re_init_smart_action()

    def action_upgrade_starport_techlab(self):
        if self.unit_selected != "STARPORT":
            return self.select_unit("STARPORT")

        if _BUILD_TECHLAB_STARPORT in self.obs.observation['available_actions']:
            self.end_action()
            return actions.FunctionCall(_BUILD_TECHLAB_STARPORT, [_QUEUED])
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
            if self.rally_unit_barracks is False and _RALLY_UNITS_MINIMAP in self.obs.observation['available_actions']:
                self.rally_unit_barracks = True
                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, self.target_rally_unit_minimap])
            # Lancement construction du marine
            if _TRAIN_MARINE in self.obs.observation['available_actions']:
                self.end_action()
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        return self.action_re_init_smart_action()

    def action_train_battle_cruiser(self):
        if self.unit_selected != "STARPORT":
            return self.select_unit("STARPORT")
        else:
            if self.rally_unit_starport is False and _RALLY_UNITS_MINIMAP in self.obs.observation['available_actions']:
                self.rally_unit_starport = True
                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, self.target_rally_unit_minimap])
            # Lancement construction du marine
            if _TRAIN_BATTLE_CRUISER in self.obs.observation['available_actions']:
                self.battle_cruiser_built += 1
                self.end_action()
                return actions.FunctionCall(_TRAIN_BATTLE_CRUISER, [_QUEUED])
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


def save_score(score, steps, win):
    try:
        num_lines = sum(1 for _ in open(_SCORE_FILE)) + 1
    except:
        num_lines = 1
    with open(_SCORE_FILE, "a+") as scores_file:
        print("{};{};{};{}"
              # .format(num_lines, datetime.timedelta(seconds=(time.time() - start_time)), score),
              .format(num_lines, steps, score, win),
              file=scores_file)


def run(agent):
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
                    step_mul=1,
                    game_steps_per_episode=0,
                    visualize=False) as env:
                run_loop.run_loop(agent, env)


def main():
    agent = [SparseAgentDefensive()]
    run(agent)


dists = [d for d in pkg_resources.working_set]
for dist in dists:
    if 'pysc2' in dist.project_name.lower():
        _PYSC2_VERSION = dist.version

if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    main()
