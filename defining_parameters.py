from charles.charles import Population, Individual
from data.stigler_diet import nutrients, data
from charles.selection import tournament_sel
from charles.mutation import swap_mutation
from charles.crossover import single_point_co


def get_values(individual):
    """A function to calculate the values for the individuals (strings containing the amount requested for
    each food on data). It aggregates the values of price and each nutrient according to the quantity
    requested for each food.
    Returns:
        list of floats: price, calories, protein, calcium, iron, vitaminA, vitaminB1, vitaminB2, niacin, vitaminC
    """

    # Initialize the variables.
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

    # For each food in our data (length of individual) we multiply the amount (represented on the individual)
    # with each of the parameters defined on data (price and each nutrient for that specific food).
    for bit in range(len(individual)):
        price += individual[bit] * data[bit][2]
        calories += individual[bit] * data[bit][3]
        protein += individual[bit] * data[bit][4]
        calcium += individual[bit] * data[bit][5]
        iron += individual[bit] * data[bit][6]
        vitaminA += individual[bit] * data[bit][7]
        vitaminB1 += individual[bit] * data[bit][8]
        vitaminB2 += individual[bit] * data[bit][9]
        niacin += individual[bit] * data[bit][10]
        vitaminC += individual[bit] * data[bit][11]

    # Returns a list with the total price and total amount of nutrients.
    return [price, calories, protein, calcium, iron, vitaminA, vitaminB1, vitaminB2, niacin, vitaminC]


def get_fitness(self):
    """A function to calculate the total price of the individual (list of foods) if the nutrients constraint is met.
    If the individual does not meet the nutrients constraint, we add a punishment to the fitness.
    The punishment is proportional to the amount of the constraint that was not met and can be added for each nutrient.
    Returns:
        int: Total price
    """

    # Initialize our function getting the values from the individual
    list_values = get_values(self.representation)

    # Initially, the fitness is the price for the list of individuals
    fitness = list_values[0]

    # If the amount of nutrients in a year is not met, we add a punishment on fitness.
    # The punishment is proportional to the difference between the total and the one obtained.
    # The punishment is divided by the total to not prioritize any nutrient over the other.

    # Calories
    if list_values[1] < 365 * nutrients[0][1]:
        fitness += 10000 * (365 * nutrients[0][1] - list_values[1]) / (365 * nutrients[0][1])
    # Protein
    if list_values[2] < 365 * nutrients[1][1]:
        fitness += 10000 * (365 * nutrients[1][1] - list_values[2]) / (365 * nutrients[1][1])
    # Calcium
    if list_values[3] < 365 * nutrients[2][1]:
        fitness += 10000 * (365 * nutrients[2][1] - list_values[3]) / (365 * nutrients[2][1])
    # Iron
    if list_values[4] < 365 * nutrients[3][1]:
        fitness += 10000 * (365 * nutrients[3][1] - list_values[4]) / (365 * nutrients[3][1])
    # Vitamin A
    if list_values[5] < 365 * nutrients[4][1]:
        fitness += 10000 * (365 * nutrients[4][1] - list_values[5]) / (365 * nutrients[4][1])
    # Vitamin B1
    if list_values[6] < 365 * nutrients[5][1]:
        fitness += 10000 * (365 * nutrients[5][1] - list_values[6]) / (365 * nutrients[5][1])
    # Vitamin B2
    if list_values[7] < 365 * nutrients[6][1]:
        fitness += 10000 * (365 * nutrients[6][1] - list_values[7]) / (365 * nutrients[6][1])
    # Niacin
    if list_values[8] < 365 * nutrients[7][1]:
        fitness += 10000 * (365 * nutrients[7][1] - list_values[8]) / (365 * nutrients[7][1])
    # Vitamin C
    if list_values[9] < 365 * nutrients[8][1]:
        fitness += 10000 * (365 * nutrients[8][1] - list_values[9]) / (365 * nutrients[8][1])

    # If the constraints were met, it returns the original price of individual.
    # Otherwise, it returns the price added the punishments.
    return fitness


# Monkey Patching
Individual.get_fitness = get_fitness

# Those parameters were executed separated and results were checked manually.
# Ideally, it would be performed a grid search over all parameter to get best result.
# Since it would take a lot of computational resources and time, the test were made independent.
# The best parameters chosen were used on the main file



# Defining valid_set
pop = Population(size=100, optim="min", sol_size=len(data), valid_set=[0, 1], replacement=True)
pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=inversion_mutation,
           crossover=single_point_co, elitism=True)
best_result = pop.get_elite()
print(f"{best_result.representation}")
print(f"{get_values(best_result)[0]}")
print(f"{best_result.get_fitness()}")
print("Since the best representation get all the individuals filled and cannot meet the nutrients requested over a "
      "year, the representation cannot be binary.")
print("It was decided to follow up with a representation quaternary (only characters in the valid set [0,1,2,3]")



# Defining Elitism
avg_fitness_elitism = 0
executions = 30
for i in range(executions):
    pop = Population(size=100, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=True)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_elitism += fitness
avg_fitness_elitism = avg_fitness_elitism / executions

avg_fitness_wo_elitism = 0
for i in range(executions):
    pop = Population(size=100, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_wo_elitism += fitness
avg_fitness_wo_elitism = avg_fitness_wo_elitism / executions
print(f"The average fitness with elitism was {avg_fitness_elitism}")
print(f"The average fitness without elitism was {avg_fitness_wo_elitism}")
print("Since the average fitness without elitism was lower, the best choice (for minimization) is to not have elitism")
print("However, the difference was small and it was different in different execution, so it can be not statistically "
      "significant")



# Defining population size
avg_fitness_pop100 = 0
executions = 30
for i in range(executions):
    pop = Population(size=100, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pop100 += fitness
avg_fitness_pop100 = avg_fitness_pop100 / executions

avg_fitness_pop200 = 0
for i in range(executions):
    pop = Population(size=200, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pop200 += fitness
avg_fitness_pop200 = avg_fitness_pop200 / executions

avg_fitness_pop300 = 0
for i in range(executions):
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pop300 += fitness
avg_fitness_pop300 = avg_fitness_pop300 / executions
print(f"The average fitness with pop_size = 100 was {avg_fitness_pop100}")
print(f"The average fitness with pop_size = 200 was {avg_fitness_pop200}")
print(f"The average fitness with pop_size = 300 was {avg_fitness_pop300}")
print("Since the best result was with population of 300, we decided to follow up on this solution.")
print("Apparently, the bigger the population size, the better the result. However, to save computation resources, "
      "the population will be limited to maximum of 300.")



# Defining crossover probability (by default, high)
avg_fitness_pc_90 = 0
executions = 30
for i in range(executions):
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pc_90 += fitness
avg_fitness_pc_90 = avg_fitness_pc_90 / executions

avg_fitness_pc_80 = 0
for i in range(executions):
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.8, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pc_80 += fitness
avg_fitness_pc_80 = avg_fitness_pc_80 / executions

avg_fitness_pc_95 = 0
for i in range(executions):
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.95, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pc_95 += fitness
avg_fitness_pc_95 = avg_fitness_pc_95 / executions
print(f"The average fitness with pc = 0.8 was {avg_fitness_pc_80}")
print(f"The average fitness with pc = 0.9 was {avg_fitness_pc_90}")
print(f"The average fitness with pc = 0.95 was {avg_fitness_pc_95}")
print("Since the best result was with pc of 0.9, we decided to follow up on this solution.")
print("It is important to highlight that 0.95 was not very different result.")



# Defining mutation probability (by default, low)
avg_fitness_pm_10 = 0
executions = 30
for i in range(executions):
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.1, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pm_10 += fitness
avg_fitness_pm_10 = avg_fitness_pm_10 / executions

avg_fitness_pm_20 = 0
for i in range(executions):
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.2, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pm_20 += fitness
avg_fitness_pm_20 = avg_fitness_pm_20 / executions

avg_fitness_pm_5 = 0
for i in range(executions):
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    pop.evolve(gens=100, xo_prob=0.9, mut_prob=0.05, select=tournament_sel, mutate=swap_mutation,
               crossover=single_point_co, elitism=False)
    best_result = pop.get_elite()
    fitness = best_result.get_fitness()
    avg_fitness_pm_5 += fitness
avg_fitness_pm_5 = avg_fitness_pm_5 / executions
print(f"The average fitness with pm = 0.1 was {avg_fitness_pm_10}")
print(f"The average fitness with pm = 0.2 was {avg_fitness_pm_20}")
print(f"The average fitness with pm = 0.05 was {avg_fitness_pm_5}")
print("Since the best result was with pm of 0.2, we decided to follow up on this solution.")
