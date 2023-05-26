from random import randint, uniform, random


def single_point_co(p1, p2):
    """Implementation of single point crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    co_point = randint(1, len(p1) - 2)

    offspring1 = p1[:co_point] + p2[co_point:]
    offspring2 = p2[:co_point] + p1[co_point:]

    return offspring1, offspring2


def cycle_xo(p1, p2):
    """Implementation of cycle crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # offspring placeholders
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    while None in offspring1:
        index = offspring1.index(None)
        val1 = p1[index]
        val2 = p2[index]

        # copy the cycle elements
        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.index(val2)

        # copy the rest
        for element in offspring1:
            if element is None:
                index = offspring1.index(None)
                if offspring1[index] is None:
                    offspring1[index] = p2[index]
                    offspring2[index] = p1[index]

    return offspring1, offspring2


def indexes_cycle_xo(p1, p2):
    """Indexing individuals to perform a cycle_xo.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # Indexing the parents for crossover
    index_o1, index_o2 = cycle_xo(range(len(p1)), range(len(p2)))

    # Offspring placeholders
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    # Associating each bit from index to offspring
    for index, bit in enumerate(index_o1):
        offspring1[index] = p1[bit]
    for index, bit in enumerate(index_o2):
        offspring2[index] = p2[bit]

    return offspring1, offspring2


def uniform_xo(p1, p2, pc=0.5):
    """Implementation of uniform crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    for index in range(len(p1)):
        if random() < pc:  # Perform crossover with a certain probability
            offspring1[index] = p2[index]
            offspring2[index] = p1[index]
        else:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]

    return offspring1, offspring2


if __name__ == '__main__':
    p1, p2 = [9, 8, 4, 5, 6, 7, 1, 3, 2, 10], [8, 7, 1, 2, 3, 10, 9, 5, 4, 6]
    o1, o2 = cycle_xo(p1, p2)
    print(o1, o2)
