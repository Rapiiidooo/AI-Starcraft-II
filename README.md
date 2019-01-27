# AI-STARCRAFT-II

Artificial Intelligence based on [pysc2](https://github.com/deepmind/pysc2) library for Starcraft II game.

## move_camera.py

Simple AI to scan the entire map moving the camera.
Made after [Nicolas Jouandeau](https://github.com/n-ai-up8-edu/pysc2) lessons.

## collect_mineral.py

AI's trying to play to the CollectMineral mini-game.
Made after [Nicolas Jouandeau](https://github.com/n-ai-up8-edu/pysc2) lessons.

## defensive_simple.py

First tries to play with a defensive strategie, all hard-coded.
Based on [Building a Basic PySC2 Agent](https://chatbotslife.com/building-a-basic-pysc2-agent-b109cde1477c) tutorial.

## scan-simple64-v2.py

Made by [Nicolas Jouandeau](https://github.com/n-ai-up8-edu/pysc2). 
The next agent is using the location returned by this one, this agent is trying to scan the entire map from minimap.

## sparse_agent_defensive.py

Agent with qLearning table based on [Refine Your Sparse PySC2 Agent](https://itnext.io/refine-your-sparse-pysc2-agent-a3feb189bc68) tutorial.

Here are his differents actions implemented, then it can choose between : 

* ACTION_DO_NOTHING
***
Nothing is doing this step.
***
* ACTION_BUILD_SCV
***
The agent's will try to select the command center and build an SCV.
***
* ACTION_BUILD_SUPPLY_DEPOT
***
The agent's will try to select a SCV unit and build a supply depot with limit of 7 near the command center.
***
* ACTION_BUILD_BARRACKS
***
The agent's will try to select a SCV unit and build a barrack with limit of 2 near the command center.
***
* ACTION_BUILD_MARINE
***
The agent's will try to select a barrack and build a marine.
***
* ACTION_BUILD_MISSILE_TURRET
***
The agent's will try to select a SCV unit and build a missile_turret with limit of one.
This building is necessary to see invisible unit.
***
* ACTION_BUILD_ENGINEERING_BAY
***
The agent's will try to select a SCV unit and build an engineering bay with limit of one near the command center.
***
* ACTION_SCV_INACTIV_TO_MINE
***
The agent's will try to select an "AFK" *(funny for a bot :))* SCV unit and if a ressource is available on the screen redirect it to this one, otherwise move it to the command center.
***
* ACTION_DEFEND_POSITION_\<**x**>_\<**y**>
***
This action is the one of the most important trying to defend our base, this is in 5 important steps:
- trying to select a SCV
- trying to move the camera into the [**x**, **y**] position
- trying to build a bunker (or missile turret if limit is reached) to random position on the screen (limit of 2). If neither of them can be build because of the limit reached then select the all army. 
- trying to move all army at the center of the [**x**, **y**] minimap position.
- trying to get a bunker selected and load a marine.
***

The rewards for the qLearning table are simple, timer and score. I tried some differents configuration and the results are below.

In other word the sparse_agent_defensive.py is learning what is the best action to do in the prematurate game, and wich place is the best trying to lock and defend against the enemy.

Score :

* using speculate time elapsed to reward : `if time.get_time() > 200 then +1 else -1`  
![](/Results/v0_reward_speculate_time.png)

* using only the victory as reward :
![](/Results/v1_reward_victoire.png)

* using step spend until end of game : `if steps > 200 then +1 else -1`  
![](/Results/v2_reward_step_sup_200.png)

* using steps spend and score reached : `if steps > 5000 or score > 2000 then +1 else -1`  
![](/Results/v3_reward_step_sup_5000_score_sup_2000.png)
