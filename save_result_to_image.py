from matplotlib import pyplot


def main():
    name = 'scores.txt'
    nb_games = []
    timers = []
    scores = []
    rewards = []
    nb_wins = []
    nb_looses = []
    nb_draws = []
    win = 0
    loose = 0
    draw = 0
    with open(name, 'r') as f:
        for line in f:
            nb_game, time, score, reward = line.split(";")
            reward = int(reward.replace('\n', ''))
            nb_games.append(int(nb_game))
            timers.append(int(time))
            scores.append(int(score))
            rewards.append(reward)
            if reward == 1:
                win += 1
            elif reward == 0:
                draw += 1
            elif reward == -1:
                loose += 1
            nb_wins.append(win)
            nb_looses.append(loose)
            nb_draws.append(draw)

    try:
        pyplot.bar(nb_wins, timers, label='Durée avant victoire')
        pyplot.bar(nb_looses, timers, label='Durée avant défaite')
        pyplot.bar(nb_draws, timers, label="Durée avant égalité")
        pyplot.title('SC II Defensive AI TIMER')
        pyplot.ylabel('timer')
        pyplot.legend()
        pyplot.savefig('plot-timer.png')
        pyplot.clf()

        pyplot.plot(nb_games, nb_wins, label='Nombre de victoire')
        pyplot.plot(nb_games, nb_looses, label='Nombre de défaite')
        pyplot.plot(nb_games, nb_draws, label="Nombre d'égalité")
        pyplot.title('SC II Defensive AI REWARD')
        pyplot.xlabel('Nombre de partie jouées')
        pyplot.legend()
        pyplot.savefig('plot-reward.png')
        pyplot.clf()
    except:
        print('Something wrong. Image not created...')


if __name__ == "__main__":
    main()
