from random import sample


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
