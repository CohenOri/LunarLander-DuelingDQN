main = [0, 0.2, 0.4, 0.6, 0.8, 1] # all the way to 1
sides = [-1, -0.9, -0.8, -0.7, -0.6 ,-0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

combinations = []
for x in main:
    for y in sides:
        combinations.append([x, y])



def discrete_to_continuous(value):
    return combinations[value]
