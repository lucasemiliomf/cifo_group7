from random import sample, randint


def quaternary_mutation(individual):
    """Binary mutation for a GA individual. Flips the bits.

    Args:
        individual (Individual): A GA individual from charles.py

    Raises:
        Exception: When individual is not binary encoded.py

    Returns:
        Individual: Mutated Individual
    """
    mut_index = randint(0, len(individual) - 1)

    if individual[mut_index] == 0:
        individual[mut_index] = sample([1, 2, 3], 1)[0]
    elif individual[mut_index] == 1:
        individual[mut_index] = sample([0, 2, 3], 1)[0]
    elif individual[mut_index] == 2:
        individual[mut_index] = sample([0, 1, 3], 1)[0]
    elif individual[mut_index] == 3:
        individual[mut_index] = sample([0, 1, 2], 1)[0]
    else:
        raise Exception(
            f"Trying to do a quaternary mutation on {individual}. But it's not quaternary.")
    return individual


def swap_mutation(individual):
    """Swap mutation for a GA individual. Swaps the bits.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    individual[mut_indexes[0]], individual[mut_indexes[1]] = individual[mut_indexes[1]], individual[mut_indexes[0]]
    return individual


def inversion_mutation(individual):
    """Inversion mutation for a GA individual. Reverts a portion of the representation.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    mut_indexes.sort()
    individual[mut_indexes[0]:mut_indexes[1]] = individual[mut_indexes[0]:mut_indexes[1]][::-1]
    return individual


if __name__ == '__main__':
    test = [1, 2, 3, 4, 5, 6]
    test = inversion_mutation(test)
    print(test)
