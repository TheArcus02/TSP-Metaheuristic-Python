from instance import Instance
from tsp import TSP

if __name__ == '__main__':
    file_path = 'data/berlin52.txt'

    instance = Instance()
    instance.get_from_file(file_path)

    tsp = TSP(instance.cities)

    lower_bound = tsp.calculate_lower_bound()
    tour, total_distance = tsp.run_greedy()
    print(tsp)
    tsp.plot_solution()
