from greedy import greedy_tsp
from instance import Instance
from utils import (calculate_lower_bound, plot_tsp_solution)

if __name__ == '__main__':
    file_path = 'data/berlin52.txt'

    instance = Instance()
    instance.get_from_file(file_path)
    cities = instance.get_cities()

    lower_bound = calculate_lower_bound(cities)
    tour, total_distance = greedy_tsp(cities)
    print(
        f'The tour is {list(map(lambda x: x + 1, tour))}\nThe tour distance is {total_distance}\nThe lower bound is {lower_bound}')
    plot_tsp_solution(cities, tour)
