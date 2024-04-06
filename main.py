from greedy import greedy_tsp
from utils import (calculate_lower_bound, generate_tsp_instance_from_file, plot_tsp_solution)

if __name__ == '__main__':
    file_path = 'data/tsp1000.txt'
    cities = generate_tsp_instance_from_file(file_path)
    # cities = generate_tsp_instance(9, 0, 10)
    lower_bound = calculate_lower_bound(cities)
    tour, total_distance = greedy_tsp(cities)
    print(f'The tour is {tour}\nThe tour distance is {total_distance}\nThe lower bound is {lower_bound}')
    plot_tsp_solution(cities, tour)
