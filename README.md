# Genetic_Algorithm_1

Simple genetic algorithm that creates new generations, where each individual is unique (meaning there aren’t two phenotypes, which share the exact same sequence of genes). 

Mutation rate is set to 8% chance and is introduced only to parents.

Population starts with 500 individuals, which are sorted based on fitness, using the built-in Timsort algorithm in Python. A maximum of 1000 generations/iterations is set. Candidates are evaluated based on two factors – if they have a correct gene (1 point) and if they have a correct gene in the correct spot, compared to the target phenotype (2 points). I’m using a mix between Elitist and Tournament selection for crossover. 26% of the size of the original population are chosen – half with the best fitness and half at random. Once the individuals are chosen, they are added to a list, from which they are randomly picked for breeding. 

Additional condition is added, which introduces 4% mortality of the population size after every iteration. 

There is also a plot function included (matplotlib), which would visualize the following after each run of the program:
•	Population size over time.
•	Average fitness over time. 

90% of the time it would take between 100 and 400 generations in order to produce the correct individual. The average fitness would be between 15 and 20 and the population size between 1400-2000. 
