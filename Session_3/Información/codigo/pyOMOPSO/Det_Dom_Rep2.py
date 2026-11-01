import numpy as np
from Dominate_rep import DominateRepository
 


def DetermineDominationRepository(e_file,e):

    repository = e_file # repositorio actual
    nPop = len(repository)

    # Create a mapping from index to object
    repo_dict = {i: repo for i, repo in enumerate(repository)}

    # Initialize a set to keep track of dominated particles
    dominated_set = set()

    # Iterate over the repository only once
    for i in range(nPop):
        for j in range(i+1, nPop):
            # Check if i dominates j
            if DominateRepository(repo_dict[i], repo_dict[j], e):
                dominated_set.add(j)
            # Check if j dominates i
            elif DominateRepository(repo_dict[j], repo_dict[i], e):
                dominated_set.add(i)

    # Filter out the dominated particles
    repository = [repo for i, repo in enumerate(repository) if i not in dominated_set]

    return repository