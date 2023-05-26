from charles.charles import Population, Individual
from data.stigler_diet import nutrients, data
from charles.selection import fps, tournament_sel
from charles.mutation import swap_mutation, inversion_mutation, quaternary_mutation
from charles.crossover import single_point_co, indexes_cycle_xo, uniform_xo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

# Defining best_individual as a null solution to get improved
best_individual = Individual(
    size=len(data),
    replacement=True,
    valid_set=[0],
)


# Selection, Crossover and Mutation methods tuning:
#    It will be performed a grid search with multiple methods of selection, crossover and mutation.
#    Ideally, all the parameters should be tested.
#    Since it would require time and computational resources, it was performed only over those methods.
#    Other parameters were defined manually and empirically in the defining_parameters.py
def execute_ga(selection, xo, mutation):
    """A function to execute the Genetic Algorithm, given a selection, crossover and mutation method as parameters.

    Args:
        selection (function): selection function from charles/selection.py
        xo (function): crossover function from charles/crossover.py
        mutation (function): mutation function from charles/mutation.py

    Returns:
        float: fitness from the best individual
        dictionary: fitness_stats from each generation over the population
    """
    pop = Population(size=300, optim="min", sol_size=len(data), valid_set=[0, 1, 2, 3], replacement=True)
    fitness_stats = {'min': [], 'max': [], 'avg': []}

    for gen in range(100):
        pop.evolve(
            gens=1,
            xo_prob=0.9,
            mut_prob=0.2,
            select=selection,
            mutate=mutation,
            crossover=xo,
            elitism=False)

        # Get fitness stats for this generation
        fitness_values = [ind.get_fitness() for ind in pop.individuals]
        fitness_stats['min'].append(min(fitness_values))
        fitness_stats['max'].append(max(fitness_values))
        fitness_stats['avg'].append(np.mean(fitness_values))

    elite = pop.get_elite()
    return elite, fitness_stats


# Defining our parameters for selection, crossover and mutation to perform a grid search.
selection_methods = [fps, tournament_sel]
xo_methods = [single_point_co, indexes_cycle_xo, uniform_xo]
mutation_methods = [quaternary_mutation, swap_mutation, inversion_mutation]

results_df = pd.DataFrame(columns=['selection_method', 'mutation_method', 'crossover_method', 'average_fitness'])

fitness_data = {}

# Grid Search
# Repeat over each selection method
for selection in selection_methods:
    # Repeat over each crossover method
    for xo in xo_methods:
        # Repeat over each mutation method
        for mutation in mutation_methods:
            # Defining parameters to calculate average fitness over a certain number of executions
            avg_fitness = 0
            executions = 30
            # Carry out GA with the current parameters several times to calculate average fitness
            for i in range(executions):
                # Executing GA with current parameters and getting fitness
                execution_elite, fitness_stats = execute_ga(selection, xo, mutation)
                fitness = execution_elite.get_fitness()
                avg_fitness += fitness

                # If our new solution is better from previous, we change
                if best_individual.get_fitness() > fitness:
                    best_individual = execution_elite

                # Add fitness_stats to fitness_data
                key = (selection.__name__, xo.__name__, mutation.__name__)
                if key not in fitness_data:
                    fitness_data[key] = {'min': [], 'max': [], 'avg': []}
                fitness_data[key]['min'].extend(fitness_stats['min'])
                fitness_data[key]['max'].extend(fitness_stats['max'])
                fitness_data[key]['avg'].extend(fitness_stats['avg'])

            # Calculating average fitness
            avg_fitness = avg_fitness / executions

            # Saving parameters in a dataframe to check best solution
            row = {'selection_method': selection.__name__, 'mutation_method': mutation.__name__,
                   'crossover_method': xo.__name__, 'average_fitness': avg_fitness}
            results_df = pd.concat([results_df, pd.DataFrame(row, index=[0])], ignore_index=True)

# Sort the dataframe by average_fitness in ascending order (lowest values first)
results_df = results_df.sort_values(by='average_fitness')

# Saving the dataframe in a csv
results_df.to_csv('output.csv', index=False)

# Get the top 3 combinations
top_combinations_df = results_df[:3]

# Plot Average Fitness Over Generations for Top 3 Combinations

# Reset index for top_combinations_df
top_combinations_df = top_combinations_df.reset_index(drop=True)

fig, ax = plt.subplots()
colors = ['blue', 'green', 'red']

for i, row in top_combinations_df.iterrows():
    selection_method = row['selection_method']
    crossover_method = row['crossover_method']
    mutation_method = row['mutation_method']
    key = (selection_method, crossover_method, mutation_method)

    # Extract the corresponding fitness data
    values = fitness_data[key]

    # Get generation range
    generations = range(100)  # 100 generations

    # Plot average fitness over generations
    ax.plot(generations, values['avg'][:100], label=f'Average for {key[0]}, {key[1]}, {key[2]}', color=colors[i])
    ax.fill_between(generations, values['min'][:100], values['max'][:100], color=colors[i], alpha=0.1)

# Add labels and legend
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
ax.legend()

# Set title
ax.set_title('Average Fitness Over Generations for Top 3 Combinations')

plt.tight_layout()
plt.show()

#### 3 separate plots

# Reset index for top_combinations_df
top_combinations_df = top_combinations_df.reset_index(drop=True)

colors = ['blue', 'green', 'red']

fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)  # Create a 1x3 grid of subplots with shared y-axis

for i, row in top_combinations_df.iterrows():
    selection_method = row['selection_method']
    crossover_method = row['crossover_method']
    mutation_method = row['mutation_method']
    key = (selection_method, crossover_method, mutation_method)

    # Extract the corresponding fitness data
    values = fitness_data[key]

    # Get generation range
    generations = range(100)  # 100 generations

    # Plot average fitness over generations in the corresponding subplot
    ax = axs[i]
    ax.plot(generations, values['avg'][:100], label=f'Average for {key[0]}, {key[1]}, {key[2]}', color=colors[i])
    ax.fill_between(generations, values['min'][:100], values['max'][:100], color=colors[i], alpha=0.1)

    # Add labels and legend
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.legend()

    # Set title for each subplot
    # ax.set_title(f'Average Fitness Over Generations\n{key[0]}, {key[1]}, {key[2]}')

# Add a single title above all three subplots
plt.suptitle('Average Fitness Over Generations for Top 3 Combinations', fontsize=16)

plt.tight_layout()
plt.show()

with open("best_individual.txt", "w") as file:
    file.write("List of foods found for optimal solution:\n")
    for index, bit in enumerate(best_individual.representation):
        if bit != 0:
            file.write(f"{data[index][0]}, {bit} times {data[index][1]}\n")

    list_values_elite = get_values(best_individual)

    file.write("\nList of nutrients consumed in a year for this set of foods")
    for i in range(len(nutrients)):
        file.write(
            f"Consumed {list_values_elite[i + 1]} {nutrients[i][0]} in a year (Required {nutrients[i][1] * 365})\n")

    file.write(f"\nCost for this solution was {best_individual.get_fitness():.0f} cents\n")

print("Output saved to best_individual.txt")
