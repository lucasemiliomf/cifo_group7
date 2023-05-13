from charles.charles import Population, Individual
from copy import deepcopy
from data.stigler_diet import nutrients, data
from charles.selection import fps, tournament_sel
from charles.mutation import binary_mutation
from charles.crossover import single_point_co
from random import random
from operator import attrgetter


def get_fitness(self):
    """A function to calculate the total weight of the bag if the capacity is not exceeded
    If the capacity is exceeded, it will return a negative fitness
    Returns:
        int: Total weight
    """
    price = 0  # Fitness
    calories = 0
    protein = 0
    calcium = 0
    iron = 0
    vitaminA = 0
    vitaminB1 = 0
    vitaminB2 = 0
    niacin = 0
    vitaminC = 0
    for bit in range(len(self.representation)):
        if self.representation[bit] == 1:
            price += data[bit][2]
            calories += data[bit][3]
            protein += data[bit][4]
            calcium += data[bit][5]
            iron += data[bit][6]
            vitaminA += data[bit][7]
            vitaminB1 += data[bit][8]
            vitaminB2 += data[bit][9]
            niacin += data[bit][10]
            vitaminC += data[bit][11]
    if calories < nutrients[0][1]:
        return 9999999
    if protein < nutrients[1][1]:
        return 9999999
    if calcium < nutrients[2][1]:
        return 9999999
    if iron < nutrients[3][1]:
        return 9999999
    if vitaminA < nutrients[4][1]:
        return 9999999
    if vitaminB1 < nutrients[5][1]:
        return 9999999
    if vitaminB2 < nutrients[6][1]:
        return 9999999
    if niacin < nutrients[7][1]:
        return 9999999
    if vitaminC < nutrients[8][1]:
        return 9999999
    return price


def get_neighbours(self):
    """A neighbourhood function for the knapsack problem,
    for each neighbour, flips the bits
    Returns:
        list: a list of individuals
    """
    n = [deepcopy(self.representation) for i in range(len(self.representation))]

    for index, neighbour in enumerate(n):
        if neighbour[index] == 1:
            neighbour[index] = 0
        elif neighbour[index] == 0:
            neighbour[index] = 1

    n = [Individual(i) for i in n]
    return n


# Monkey Patching
Individual.get_fitness = get_fitness
Individual.get_neighbours = get_neighbours

pop = Population(size=50, optim="min", sol_size=len(data), valid_set=[0, 1], replacement=True)

pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.2, select=tournament_sel,
           mutate=binary_mutation, crossover=single_point_co,
           elitism=False)

elite = deepcopy(min(pop.individuals, key=attrgetter("fitness")))
for index, bit in enumerate(elite.representation):
    if bit == 1:
        print(data[index])
