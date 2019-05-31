from matplotlib import pyplot


def main():
    name = 'scores.txt'
    nb_games = []
    timers = []
    scores = []
    wins = []
    with open(name, 'r') as f:
        for line in f:
            nb_game, time, score, win = line.split(";")
            nb_games.append(int(nb_game))
            timers.append(int(time))
            scores.append(int(score))
            wins.append(int(win.replace('\n', '')))

    try:
        pyplot.plot(nb_games, timers)
        pyplot.title('SC II Defensive AI TIMER')
        pyplot.ylabel('steps')
        pyplot.xlabel('nb_games')
        pyplot.savefig('plot-timer.png')
        pyplot.clf()

        pyplot.plot(nb_games, scores)
        pyplot.title('SC II Defensive AI SCORE')
        pyplot.xlabel('nb_games')
        pyplot.ylabel('score (score_cummulative)')
        pyplot.savefig('plot-score.png')
        pyplot.clf()

        pyplot.plot(nb_games, wins)
        pyplot.title('SC II Defensive AI SCORE')
        pyplot.xlabel('nb_games')
        pyplot.ylabel('win')
        pyplot.savefig('plot-win.png')
        pyplot.clf()
    except:
        print('Something wrong. Image not created...')


if __name__ == "__main__":
    main()
