from charles.charles import Population, Individual
from copy import deepcopy
from data.stigler_diet import nutrients, data
from charles.selection import fps, tournament_sel
from charles.mutation import binary_mutation, swap_mutation
from charles.crossover import single_point_co
from random import random
from operator import attrgetter


def get_values(individual):
    price = 0
    calories = 0
    protein = 0
    calcium = 0
    iron = 0
    vitaminA = 0
    vitaminB1 = 0
    vitaminB2 = 0
    niacin = 0
    vitaminC = 0
    for bit in range(len(individual)):
        price += individual[bit]*data[bit][2]
        calories += individual[bit]*data[bit][3]
        protein += individual[bit]*data[bit][4]
        calcium += individual[bit]*data[bit][5]
        iron += individual[bit]*data[bit][6]
        vitaminA += individual[bit]*data[bit][7]
        vitaminB1 += individual[bit]*data[bit][8]
        vitaminB2 += individual[bit]*data[bit][9]
        niacin += individual[bit]*data[bit][10]
        vitaminC += individual[bit]*data[bit][11]
    return [price, calories, protein, calcium, iron, vitaminA, vitaminB1, vitaminB2, niacin, vitaminC]


def get_fitness(self):
    """A function to calculate the total weight of the bag if the capacity is not exceeded
    If the capacity is exceeded, it will return a negative fitness
    Returns:
        int: Total weight
    """
    list_values = get_values(self.representation)
    fitness = list_values[0]
    if list_values[1] < 365 * nutrients[0][1]:
        fitness += 100000*(365 * nutrients[0][1] - list_values[1])/(365 * nutrients[0][1])
    if list_values[2] < 365 * nutrients[1][1]:
        fitness += 100000*(365 * nutrients[1][1] - list_values[2])/(365 * nutrients[1][1])
    if list_values[3] < 365 * nutrients[2][1]:
        fitness += 100000*(365 * nutrients[2][1] - list_values[3])/(365 * nutrients[2][1])
    if list_values[4] < 365 * nutrients[3][1]:
        fitness += 100000*(365 * nutrients[3][1] - list_values[4])/(365 * nutrients[3][1])
    if list_values[5] < 365 * nutrients[4][1]:
        fitness += 100000*(365 * nutrients[4][1] - list_values[5])/(365 * nutrients[4][1])
    if list_values[6] < 365 * nutrients[5][1]:
        fitness += 100000*(365 * nutrients[5][1] - list_values[6])/(365 * nutrients[5][1])
    if list_values[7] < 365 * nutrients[6][1]:
        fitness += 100000*(365 * nutrients[6][1] - list_values[7])/(365 * nutrients[6][1])
    if list_values[8] < 365 * nutrients[7][1]:
        fitness += 100000*(365 * nutrients[7][1] - list_values[8])/(365 * nutrients[7][1])
    if list_values[9] < 365 * nutrients[8][1]:
        fitness += 100000*(365 * nutrients[8][1] - list_values[9])/(365 * nutrients[8][1])
    return fitness


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

pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)

pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel,
           mutate=swap_mutation, crossover=single_point_co,
           elitism=True)

elite = deepcopy(min(pop.individuals, key=attrgetter("fitness")))

for index, bit in enumerate(elite.representation):
    if bit != 0:
        print(f"{data[index][0]}, {bit} times {data[index][1]}")

list_values_elite = get_values(elite)

for i in range(len(nutrients)):
    print(f"Consumed {list_values_elite[i+1]} {nutrients[i][0]} in a year (Required {nutrients[i][1]*365})")

print(f"Cost was ${list_values_elite[0]:.2f}")
