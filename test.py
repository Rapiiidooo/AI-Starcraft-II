from math import sqrt


def get_nearest(unite, targets):
    closest = targets[0]
    unita, unitb = unite

    closesta, closestb = closest
    dclosest = distance_euclid(unita, unitb, closesta, closestb)
    for target, i in enumerate(targets):
        x, y = target
        dtmp = distance_euclid(unita, unitb, x, y)
        if dclosest > dtmp:
            dclosest = dtmp
            closest = targets[i]
    return closest


def distance_euclid(xa, xb, ya, yb):
    return sqrt((xb - xa) ** 2 + (yb - ya) ** 2)


shards = [
    (39, 33),
    (14, 24),
    (45, 44),
    (40, 8),
    (32, 19),
    (16, 38),
    (61, 17),
    (18, 34),
    (51, 9),
    (7, 28),
    (67, 26),
    (8, 46),
    (64, 15),
    (49, 29),
    (75, 40),
    (26, 22),
    (56, 51),
    (23, 26),
    (59, 52),
    (37, 23)
]
unit = (49, 20)

print(get_nearest(unit, shards))
