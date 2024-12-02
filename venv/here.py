import os
import sys
import math
import random
import time
from multiprocessing import Pool




# Utility Functions
def calculate_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_tour_distance(tour, points):
    return sum(
        calculate_distance(points[tour[i]], points[tour[(i + 1) % len(tour)]])
        for i in range(len(tour))
    )

def read_tsp_file(filename):
    points = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        node_section = False
        for line in lines:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                node_section = True
                continue
            if line == "EOF":
                break
            if node_section:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x = float(parts[1])
                    y = float(parts[2])
                    points.append((x, y))
    return points

def save_solution(filename, distance, tour):
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, 'w') as file:
        file.write(f"{distance}\n")
        file.write(','.join(map(str, tour)))

# Genetic Algorithm Components
def create_individual(num_points):
    individual = list(range(num_points))
    random.shuffle(individual)
    return individual

def create_population(population_size, num_points):
    return [create_individual(num_points) for _ in range(population_size)]

def calculate_fitness(individual, points):
    distance = calculate_tour_distance(individual, points)
    return 1.0 / distance

def rank_population(population, points):
    fitness_results = [(i, calculate_fitness(ind, points)) for i, ind in enumerate(population)]
    return sorted(fitness_results, key=lambda x: x[1], reverse=True)

def selection(ranked_population, elite_size):
    selection_results = [ranked_population[i][0] for i in range(elite_size)]
    df = sum(fitness for _, fitness in ranked_population)
    cumulative_fitness = [sum(fitness for _, fitness in ranked_population[:i + 1]) for i in range(len(ranked_population))]
    
    for _ in range(len(ranked_population) - elite_size):
        pick = random.uniform(0, df)
        for i, cum_fit in enumerate(cumulative_fitness):
            if pick <= cum_fit:
                selection_results.append(ranked_population[i][0])
                break
    return selection_results

def mating_pool(population, selection_results):
    return [population[i] for i in selection_results]

def partially_mapped_crossover(parent1, parent2):
    child = [-1] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]
    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child

def breed_population(matingpool, elite_size):
    children = matingpool[:elite_size]
    pool = random.sample(matingpool, len(matingpool))
    for i in range(len(matingpool) - elite_size):
        child = partially_mapped_crossover(pool[i], pool[len(pool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

def mutate_population(population, mutation_rate):
    return [mutate(ind.copy(), mutation_rate) for ind in population]

def two_opt(individual, points):
    best_distance = calculate_tour_distance(individual, points)
    for i in range(len(individual) - 1):
        for j in range(i + 2, len(individual)):
            if j - i == 1:
                continue
            new_individual = individual[:i] + individual[i:j][::-1] + individual[j:]
            new_distance = calculate_tour_distance(new_individual, points)
            if new_distance < best_distance:
                return new_individual
    return individual

def next_generation(current_gen, elite_size, mutation_rate, points):
    ranked_pop = rank_population(current_gen, points)
    selection_results = selection(ranked_pop, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return [two_opt(ind, points) for ind in next_gen]

def simulated_annealing(points, initial_solution, initial_temp=1000, cooling_rate=0.995):
    current_solution = initial_solution
    best_solution = initial_solution
    current_temp = initial_temp

    while current_temp > 1:
        new_solution = current_solution[:]
        idx1, idx2 = random.sample(range(len(new_solution)), 2)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]

        current_distance = calculate_tour_distance(current_solution, points)
        new_distance = calculate_tour_distance(new_solution, points)

        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / current_temp):
            current_solution = new_solution

        if calculate_tour_distance(current_solution, points) < calculate_tour_distance(best_solution, points):
            best_solution = current_solution

        current_temp *= cooling_rate

    return best_solution

# Genetic Algorithm with Simulated Annealing
def genetic_algorithm_with_sa(points, population_size=100, elite_size=20, mutation_rate=0.01, generations=5000, cutoff_time=600):
    num_points = len(points)
    population = create_population(population_size, num_points)
    start_time = time.time()
    best_distance = float('inf')
    best_route = None

    for generation in range(generations):
        if time.time() - start_time > cutoff_time:
            print(f"Cutoff time reached at generation {generation}")
            break

        population = next_generation(population, elite_size, mutation_rate, points)
        current_best_index = rank_population(population, points)[0][0]
        current_best_route = population[current_best_index]
        current_best_route = simulated_annealing(points, current_best_route)
        current_best_distance = calculate_tour_distance(current_best_route, points)

        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route

        if generation % 100 == 0:
            print(f"Generation {generation}: Best Distance = {best_distance}")

    return best_distance, best_route

# Main Function
def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <datafile> <method> <cutoff_time>")
        sys.exit(1)

    datafile = sys.argv[1]
    method = sys.argv[2]
    try:
        cutoff_time = int(sys.argv[3])
    except ValueError:
        print("Error: <cutoff_time> must be an integer.")
        sys.exit(1)

    try:
        points = read_tsp_file(datafile)
    except FileNotFoundError:
        print(f"Error: File '{datafile}' not found.")
        sys.exit(1)

    if method == "GA_SA":
        population_size = 250
        elite_size = 50
        mutation_rate = 0.01
        generations = 10000
        distance, tour = genetic_algorithm_with_sa(points, population_size, elite_size, mutation_rate, generations, cutoff_time)
    else:
        print("Error: Supported method is 'GA_SA'.")
        sys.exit(1)

    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)
    output_filename = f"{os.path.basename(datafile).split('.')[0]}_{method}_{cutoff_time}.sol"
    output_file = os.path.join(output_directory, output_filename)

    save_solution(output_file, distance, tour)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
