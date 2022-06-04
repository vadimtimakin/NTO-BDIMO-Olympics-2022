from nis import cat


with open('sub.txt', 'r') as file:
    catboost = file.readlines()
    catboost = [float(i[:-1]) for i in catboost]

with open('prophet.txt', 'r') as file:
    prophet = file.readlines()
    prophet = [float(i[:-1]) for i in prophet]

final = [c * 0.95 + p * 0.05 for c, p in zip(catboost, prophet)]

with open('ensemble.txt', 'w') as file:
    for i in final:
        file.write(str(i) + '\n')