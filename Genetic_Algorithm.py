import random
import numpy as np
import matplotlib.pyplot as plt

# Setting up the genotype
genes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
target_word = "HELLO WORLD"

# Function for evaluating the fitness of the individuals
def fitness(target_word, candidate):
    score = 0
    evaluated_cand = {}
    for i in range (len(target_word)):
        if (target_word[i] == candidate[i]):
            score += 2
        elif (target_word[i] in candidate):
            score += 1
        evaluated_cand.update({candidate : score})
    return evaluated_cand

# Sorting through candidates in order to pick the ones for breeding
def best_fitness(evaluated_cand, choose_best, choose_lucky):
    chosen_ones = []
    best_pick = sorted(evaluated_cand.items(), key=lambda x: x[1])
    best_tuple_list = []
    for i in range (choose_best):
        best_tuple_list.append(best_pick[len(best_pick)-(i+1)])

    best_list = [x[0] for x in best_tuple_list]
    for i in range (len(best_list)):
        chosen_ones.append(best_list[i])

    iter=0
    while (iter < choose_lucky):
        random_pick = random.choice(list(evaluated_cand.keys()))
        chosen_ones.append(random_pick)
        iter+=1
        for j in range (len(chosen_ones)-1):
            if (random_pick == chosen_ones[j]):
                del chosen_ones[-1]
                iter-=1
            else:
                continue

    return chosen_ones

# Creating the first population - the originals
def first_population(population_size):
    originals = []
    for i in range (0, population_size):
        word = ""
        for j in range (0, len(target_word)):
            gene_set = random.randint(0, len (genes)-1)
            word += (str(genes[gene_set]))
        originals.insert(i, word)
    return originals

# Separate function for the first crossover
def initial_crossover(best_individuals_pop, prev_generation):
    children = []

    parent1 = ""
    parent2 = ""

    # create child
    for i in range (len(best_individuals_pop)):
        parent1 = best_individuals_pop[random.randint(0, len(best_individuals_pop)-1)]
        parent2 = best_individuals_pop[random.randint(0, len(best_individuals_pop)-1)]
        if (parent1 == parent2):
            parent2 = best_individuals_pop[random.randint(0, len(best_individuals_pop) - 1)]
        else:
            child = parent1[:4] + parent2[4:6] + parent1[6:8] + parent2[8:]
            children.append(child)

    check_duplicates(children, parent1, parent2, prev_generation)

    return children

# Crossover function for future generations
def crossover(best_individuals_pop, prev_generation):
    children = []

    parent1 = ""
    parent2 = ""

    # create child
    for i in range (len(best_individuals_pop)):
        parent1 = best_individuals_pop[random.randint(0, len(best_individuals_pop)-1)]
        parent2 = best_individuals_pop[random.randint(0, len(best_individuals_pop)-1)]
        if (parent1 == parent2):
            parent2 = best_individuals_pop[random.randint(0, len(best_individuals_pop) - 1)]
        if (random.random()*100 < chance_mutation):
            parent_mutated = add_mutated_parent(parent1)
            child = parent_mutated[:4] + parent2[4:6] + parent_mutated[6:8] + parent2[8:]
            children.append(child)
        else:
            child = parent1[:4] + parent2[4:6] + parent1[6:8] + parent2[8:]
            children.append(child)

    check_duplicates(children, parent1, parent2, prev_generation)

    return children

# Adding the children to the generation
def next_generation(sorted, children):
    next_list = [x[0] for x in sorted]
    for i in range (len(children)):
        next_list.append(children[i])

    return next_list

# Separate function to check for duplicates when creating children.
# Checks if child is the same as another individual from population. if yes - delete current child and create another one.
# Also checks if the new child is the same as the second child
def check_duplicates(children, parent1, parent2, prev_generation):
    iter=0
    while(iter < len(children)):
        for j in range(len(prev_generation)):
            if (children[iter] == prev_generation[j]):
                del children[iter]
                random1 = random.randint(0, len(target_word))
                child = parent1[:random1] + parent2[random1:]
                children.append(child)
        iter+=1

    for i in range (len(children)-1):
        if (children[i] == children[i+1]):
            del children[-1]
            random1 = random.randint(0, len(genes) - 1)
            child = parent1[:random1] + parent2[random1:]
            children.append(child)

    return children

# Adding the mutation function for the parents only
def mutation(parent):
    index_modification = int(random.random() * len(parent))
    new_gene = random.randint(0, len(genes) - 1)
    parent = parent.replace(parent[index_modification], genes[new_gene], 1)

    return parent

# Adding successfully mutated parents to the generation
def add_mutated_parent(parent):
    for i in range(len(next_gen)):
        if(parent == next_gen[i]):
            del next_gen[i]
            break
    parent = mutation(parent)
    next_gen[-1] = parent
    return parent

# A function to check if the target word's been reached
def goal_check(children):
    for i in range (len(children)):
        if (children[i] == target_word):
            print("HEUREKA, ""HELLO WORLD"" FOUND! ")
            return True
        else:
            return False

# Plotting the Population Average Fitness over time
def evolutionAverageFitness(gen_counter, target_word, evolutionFitness):
    plt.axis([0, gen_counter, 0, 30])
    plt.title(target_word)
    plt.plot(evolutionFitness)
    plt.ylabel('Average fitness')
    plt.xlabel('Number of generations')
    plt.show()

# Plotting the Population Size over time
def populationSize(number_generations, generation_size):
    plt.axis([0, number_generations, 0, 5000])
    plt.title("Population over time")
    plt.plot(generation_size)
    plt.ylabel('Size of population')
    plt.xlabel('Number of generations')
    plt.show()

originals = first_population(500)  # creating first population of size 500


# choose 26% of the initial population for crossover this goes for every other generation in order to keep it from exploding in numbers.
choose_best = (int)((13*len(originals))/100)
choose_lucky = (int)((13*len(originals))/100)
check_fitness = {}
chance_mutation = 8

for i in range(len(originals)):
    check_fitness.update(fitness(target_word, originals[i]))

sort_fitness = sorted(check_fitness.items(), key=lambda x: x[1])
best_individuals_pop = best_fitness(check_fitness, choose_best, choose_lucky)
children = initial_crossover(best_individuals_pop, originals)
next_gen = next_generation(sort_fitness, children)

# Loop for testing generations
num_of_gen = 1000
gen_counter = 1  # Counter for the number of generations
ev_list = []  # A list to store all the fitness values for a single population.
evolutionFitness = []  # A list to store the average fitness values for all generations.
gen_size_history = []  # List to store the population size of each generation.

while (num_of_gen > 0):
    gen_counter+=1
    print("Loading Generation...", gen_counter)
    print(len(next_gen), " individuals", "\n")
    gen_size_history.append(len(next_gen))
    for key in check_fitness.keys():
        ev_list.append(check_fitness[key])
    evolutionFitness.append(np.mean(ev_list))
    check_fitness = {}
    for i in range(len(next_gen)):
        check_fitness.update(fitness(target_word, next_gen[i]))
    sort_fitness = sorted(check_fitness.items(), key=lambda x: x[1])
    best_individuals_pop = best_fitness(check_fitness, choose_best, choose_lucky)
    children = crossover(best_individuals_pop, next_gen)

    if (goal_check(children)):  # check if target is reached
        break
    next_gen = next_generation(sort_fitness, children)

    death_percentage = (int)((4*len(next_gen))/100)  # random deaths per generation - set to 4% the size of the population
    for i in range(death_percentage):
        del next_gen[random.randint(0, len(next_gen)-1)]
    num_of_gen-=1

# Print the last sorted generation with their fitness value
print(sorted(check_fitness.items(), key=lambda x: x[1]))
evolutionAverageFitness(gen_counter, target_word, evolutionFitness) # Plot the values for Average Fitness
populationSize(gen_counter, gen_size_history) # Plot the values for population size.
print("Generation number ", gen_counter)