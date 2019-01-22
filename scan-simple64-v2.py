import time

from pysc2.agents import base_agent
from pysc2.lib import actions

# simple scan
# optimisation camera centree de 24x24
# decallage de 20x20 pour des questions de couverture du scan
# permettant de faire 149 scans complets par mission Simple64


class Simple(base_agent.BaseAgent):
    fen_x = 64
    fen_y = 64

    MAP_X_SIZE = 64
    MAP_Y_SIZE = 64
    FEN_DX = 20
    FEN_DY = 20

    nb_scan = 0

    def step(self, obs):
        super(Simple, self).step(obs)
        time.sleep(0.5)
        if self.fen_x < self.MAP_X_SIZE - self.FEN_DX:
            self.fen_x = self.fen_x + self.FEN_DX
        else:
            if self.fen_y < self.MAP_Y_SIZE - self.FEN_DY:
                self.fen_x = 12
                self.fen_y = self.fen_y + self.FEN_DY
            else:
                self.fen_x = 12
                self.fen_y = 12
                self.nb_scan = self.nb_scan + 1
        print("[{}, {}] ".format(int(self.fen_x), int(self.fen_y)))
        return actions.FUNCTIONS.move_camera([int(self.fen_x), int(self.fen_y)])
