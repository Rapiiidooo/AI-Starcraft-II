import pandas as pd

SMART_ACTIONS = [
    'ACTION_DO_NOTHING',
    'ACTION_TRAIN_SCV',
    'ACTION_TRAIN_MARINE',
    'ACTION_TRAIN_BATTLE_CRUISER',
    'ACTION_BUILD_SUPPLY_DEPOT',
    'ACTION_BUILD_BARRACKS',
    'ACTION_BUILD_MISSILE_TURRET',
    'ACTION_BUILD_ENGINEERING_BAY',
    'ACTION_BUILD_REFINERY',
    'ACTION_BUILD_FACTORY',
    'ACTION_BUILD_STARPORT',
    'ACTION_BUILD_FUSION_CORE',
    'ACTION_UPGRADE_STARPORT_TECHLAB',
    'ACTION_ECONOMISE',
    'ACTION_SCV_TO_VESPENE',
    'ACTION_SCV_INACTIV_TO_MINE',
    'ACTION_DEFEND_POSITION',
    'ACTION_DEFEND_VS_ENEMY',
    'ACTION_ATTACK',
    'ACTION_SUPPLY_DEPOT_RAISE_QUICK'
]


def main():
    output = 'qtable.txt'

    df = pd.read_pickle('data.gz', compression='gzip')
    desc = df.describe(include='all').to_string()
    index_end_line = desc.find('\n')
    columns = desc[:index_end_line]
    values = desc[index_end_line:]

    for i, action in enumerate(SMART_ACTIONS):
        columns = columns.replace(str(i), action, 1)

    with open(output, "w+") as output_file:
        print("{}".format(columns + values),
              file=output_file)


if __name__ == "__main__":
    main()
