import os
import sys
import math
import random

def set_seed(seed):
    random.seed(seed)
    
# Utility Functions
def calculate_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def calculate_tour_distance(tour, points):
    """
    Calculate the total distance of a tour.
    """
    total_distance = 0
    for i in range(len(tour)):
        total_distance += calculate_distance(points[tour[i]], points[tour[(i + 1) % len(tour)]])
    return total_distance

def generate_initial_solution(num_points):
    """
    Generate a random initial solution.
    """
    solution = list(range(num_points))
    random.shuffle(solution)
    return solution

def two_opt_swap(tour, i, k):
    """
    Perform a 2-opt swap on the tour.
    """
    return tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]

def local_search(points, max_iterations=1000):
    """
    Perform local search to solve TSP.
    """
    num_points = len(points)
    current_solution = generate_initial_solution(num_points)
    current_distance = calculate_tour_distance(current_solution, points)

    for _ in range(max_iterations):
        improved = False
        for i in range(num_points - 1):
            for k in range(i + 1, num_points):
                new_solution = two_opt_swap(current_solution, i, k)
                new_distance = calculate_tour_distance(new_solution, points)
                if new_distance < current_distance:
                    current_solution = new_solution
                    current_distance = new_distance
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return current_solution, current_distance

def read_tsp_file(filename):
    """
    Read a TSP file and return a list of points.
    """
    points = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        node_section = False
        for line in lines:
            if line.startswith("NODE_COORD_SECTION"):
                node_section = True
                continue
            if line.startswith("EOF"):
                break
            if node_section:
                parts = line.strip().split()
                if len(parts) == 3:
                    _, x, y = parts
                    points.append((float(x), float(y)))
    return points

def save_solution(filename, distance, tour):
    """
    Save the solution to a file.
    """
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(filename, 'w') as file:
        file.write(f"{distance}\n")
        file.write(','.join(map(str, tour)))

# Approximation Algorithm
def mst_approximation(points):
    """
    Perform MST-based 2-approximation for TSP.
    """
    import networkx as nx

    # Create a graph with distances as weights
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = calculate_distance(points[i], points[j])
            G.add_edge(i, j, weight=distance)

    # Compute the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(G, weight='weight')

    # Perform Depth-First Search (DFS) to generate the tour
    preorder_nodes = list(nx.dfs_preorder_nodes(mst, source=0))
    tour = preorder_nodes + [preorder_nodes[0]]

    # Calculate the total distance
    total_distance = sum(
        calculate_distance(points[tour[i]], points[tour[i + 1]]) for i in range(len(tour) - 1)
    )

    return total_distance, tour

# Main Function
def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <datafile> Approx <cutoff_time>")
        sys.exit(1)

    datafile = sys.argv[1]
    method = sys.argv[2]
    try:
        cutoff_time = int(sys.argv[3])
    except ValueError:
        print("Error: <cutoff_time> must be an integer.")
        sys.exit(1)

    if method != "Approx":
        print("Error: This program only supports the 'Approx' method.")
        sys.exit(1)

    try:
        points = read_tsp_file(datafile)
    except FileNotFoundError:
        print(f"Error: File '{datafile}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{datafile}': {e}")
        sys.exit(1)

    try:
        distance, tour = mst_approximation(points)
    except Exception as e:
        print(f"Error running MST Approximation: {e}")
        sys.exit(1)

    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)
    output_filename = f"{os.path.basename(datafile).split('.')[0]}_{method}_{cutoff_time}.sol"
    output_file = os.path.join(output_directory, output_filename)

    try:
        save_solution(output_file, distance, tour)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving solution to file '{output_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
