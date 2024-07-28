# Imports:
import numpy as np
import matplotlib.pyplot as plt

# Helper functions:
def read_adjacency_matrix(file_path):
    """
    Read adjacency matrix from txt file.
    """
    return np.loadtxt(file_path, delimiter=',')

# ACO Algorithm:
def aco_algorithm(adjacency_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, Q):
    """
    ACO Implementation.
    """
    num_points = len(adjacency_matrix) # Number of points in adjacency matrix
    pheromone = np.ones((num_points, num_points)) # Initialize pheromone levels
    shortest_path = None # Initialize shortest path
    shortest_path_length = None  # Initialize shortest path length
    
    for iteration in range(num_iterations):
        paths = [] # List to store paths taken by ants in this iteration
        path_lengths = [] # List to store lengths of each of the paths taken in this iteration
        
        # Iterate over each ant and find a path it takes
        for ant in range(num_ants):
            visited = [False]*num_points # Initialize list with length nump_points
            current_point = np.random.randint(num_points) # Choose a random starting point
            visited[current_point] = True
            path = [current_point]
            path_length = 0
            
            # Repeat until all points are visited once
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0] # Unvisited points
                probabilities = [] # List to store probabilities
                for next_point in unvisited: # Update probabilities for each unvisited point
                    tau = pheromone[current_point][next_point] ** alpha # Account for pheromone levels
                    heuristic = (1.0 / adjacency_matrix[current_point][next_point]) ** beta # Heuristic function
                    probabilities.append(tau * heuristic) # Update probability for current point 
                probabilities = probabilities / np.sum(probabilities) # Normalize
                next_point = np.random.choice(unvisited, p=probabilities) # Choose next point based on probabilities
                path.append(next_point) # Add next point to path
                path_length += adjacency_matrix[current_point][next_point]
                current_point = next_point # Iterate to the next point
                visited[current_point] = True # Update visited list
            
            path_length += adjacency_matrix[current_point][path[0]]  # Return to start
            path.append(path[0]) # Add starting point to complete the loop
            paths.append(path) # Add path to the list of paths
            path_lengths.append(path_length) # Add path length to the list of path lengths
        
        # Evaporation of pheromone levels
        for i in range(num_points):
            for j in range(num_points):
                pheromone[i][j] *= (1 - evaporation_rate)
        
        # Update pheromone levels based on paths taken by the ants
        for path, path_length in zip(paths, path_lengths):
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i+1]] += Q / path_length
        
        # Determine shortest path and shortest path length
        min_path_length = min(path_lengths)
        if shortest_path_length is None or min_path_length < shortest_path_length:
            shortest_path_length = min_path_length
            shortest_path = paths[np.argmin(path_lengths)]
    
    # Return results
    shortest_path = [int(node) for node in shortest_path] # Convert to int for readability
    return shortest_path, shortest_path_length

def aco_algorithm_with_tracking(adjacency_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, Q):
    global_best_cost = float('inf')
    global_best_path = None
    global_best_costs = []

    for iteration in range(num_iterations):
        shortest_path, shortest_path_length = aco_algorithm(adjacency_matrix, num_ants, 1, alpha, beta, evaporation_rate, Q)
        
        if shortest_path_length < global_best_cost:
            global_best_cost = shortest_path_length
            global_best_path = shortest_path
        
        global_best_costs.append(global_best_cost)
    
    return global_best_path, global_best_cost, global_best_costs

if __name__ == "__main__":
    adjacency_matrix = read_adjacency_matrix('adjacency.txt')
    num_ants = 20 # number of ants used
    num_iterations = 30 # number of iterations
    alpha = 0.5
    beta = 0.5
    evaporation_rate = 0.1 # rate of evaporation
    Q = 50 # Amount of pheromone deposited by ants
    
    best_path, best_path_length, global_best_costs = aco_algorithm_with_tracking(adjacency_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, Q)
    print("Shortest path:", best_path)
    print("Shortest path length:", best_path_length)
    
    # Plot global best cost vs iterations
    plt.plot(range(num_iterations), global_best_costs)
    plt.xlabel('Iterations')
    plt.ylabel('Global Best Cost')
    plt.title('Global Best Cost vs Iterations')
    plt.show()
    
    # Print the traversal sequence of the single best solution
    print("Traversal sequence of the single best solution:", best_path)

# if __name__ == "__main__":
#     adjacency_matrix = read_adjacency_matrix('adjacency.txt')

#     # Define ranges for parameters
#     num_ants_range = range(5, 11)  # From 5 to 10 inclusive
#     num_iterations_range = range(50, 151, 50)  # From 50 to 150 in steps of 30
#     alpha_range = [x / 10.0 for x in range(5, 31, 5)]  # From 0.1 to 3.0 in steps of 0.3
#     beta_range = [x / 10.0 for x in range(5, 31, 5)]  # From 0.5 to 3.0 in steps of 0.5
#     evaporation_rate_range = [x / 10.0 for x in range(1, 11, 2)]  # From 0.3 to 0.7 in steps of 0.2
#     Q_range = range(50, 151, 50)  # From 50 to 150 in steps of 50

#     with open('output.txt', 'w') as file:
#         for num_ants in num_ants_range:
#             for num_iterations in num_iterations_range:
#                 for alpha in alpha_range:
#                     for beta in beta_range:
#                         for evaporation_rate in evaporation_rate_range:
#                             for Q in Q_range:
#                                 file.write(f"Testing with parameters: num_ants={num_ants}, num_iterations={num_iterations}, alpha={alpha}, beta={beta}, evaporation_rate={evaporation_rate}, Q={Q}\n")

#                                 shortest_path, shortest_path_length = aco_algorithm(adjacency_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, Q)
#                                 file.write(f"Shortest path: {shortest_path}\n")
#                                 file.write(f"Shortest path length: {shortest_path_length}\n")
                            
#     print("Done!")

# if __name__ == "__main__":
#     adjacency_matrix = read_adjacency_matrix('adjacency.txt')
#     num_ants = 20
#     num_iterations = 30
#     alpha = 0.5
#     beta = 0.5
#     evaporation_rate = 0.1
#     Q = 50
    
#     shortest_path, shortest_path_length = aco_algorithm(adjacency_matrix, num_ants, num_iterations, alpha, beta, evaporation_rate, Q)
#     print("Shortest path:", shortest_path)
#     print("Shortest path length:", shortest_path_length)