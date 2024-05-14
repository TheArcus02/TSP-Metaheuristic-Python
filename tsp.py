import itertools
import time
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from aco import ACO
from utils import calculate_distance


class TSP:
    def __init__(self, cities: List[Tuple[int, int]]):
        self._cities = cities
        self._tour = None
        self._distance = None
        self._lower_bound = None
        self._last_used_algorithm = None

    def __repr__(self):
        return f'The tour is {list(map(lambda x: x + 1, self.tour))}\nThe tour distance is {self.distance}'

    def run_greedy(self):
        self.last_used_algorithm = 'Greedy'
        points = self.cities
        num_points = len(points)
        remaining_points = set(range(num_points))
        tour = []

        current_point = 0
        tour.append(current_point)
        remaining_points.remove(current_point)

        total_distance = 0

        while remaining_points:
            nearest_point = min(remaining_points,
                                key=lambda x: calculate_distance(points[current_point], points[x]))
            tour.append(nearest_point)
            remaining_points.remove(nearest_point)
            total_distance += calculate_distance(points[current_point], points[nearest_point])
            current_point = nearest_point

        # Add return to the starting point
        tour.append(0)
        # Add distance to return to the starting point
        total_distance += calculate_distance(points[tour[-2]], points[0])

        self.distance = total_distance
        self.tour = tour

        return tour, total_distance

    def run_aco(self, num_ants: int, num_iterations: int, alpha: float, beta: float, rho: float):
        self.last_used_algorithm = 'ACO'
        aco = ACO(
            cities=self.cities,
            num_ants=num_ants,
            num_iterations=num_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho
        )
        tour, tour_len = aco.run()
        self.tour = tour
        self.distance = tour_len

        return tour, tour_len

    def tune_aco_parameters(self, parameter_values: Dict[str, List], num_trials: int, log=False):
        start_time = time.time()

        best_parameters = None
        best_tour_length = float('inf')

        param_combinations = list(itertools.product(*parameter_values.values()))
        np.random.shuffle(param_combinations)

        if log is True:
            print('Tuning ACO parameters...')
        for i, params in enumerate(param_combinations):
            if i is num_trials:
                break
            if log is True:
                print(f'Iteration {i + 1} of {num_trials}')

            kwargs = {param_name: param_value for param_name, param_value in zip(parameter_values.keys(), params)}
            if log is True:
                print(f'trying {kwargs}')
            _, performance = self.run_aco(**kwargs)

            if performance < best_tour_length:
                best_tour_length = performance
                best_parameters = kwargs

        end_time = time.time()
        if log is True:
            print('Tuning completed')
            print(f'tuning time: {end_time - start_time}s')
            print(f'best parameters: {best_parameters}')
        return best_parameters, best_tour_length

    def calculate_lower_bound(self):
        points = self.cities
        G = nx.Graph()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = calculate_distance(points[i], points[j])
                G.add_edge(i, j, weight=distance)

        MST = nx.minimum_spanning_tree(G)

        # Calculate lower bound as the sum of weights of edges in MST
        lower_bound = sum(MST[i][j]['weight'] for i, j in MST.edges())

        self.lower_bound = lower_bound
        return lower_bound

    def plot_solution(self):
        points = self.cities
        tour = self.tour

        x = [point[0] for point in points]
        y = [point[1] for point in points]

        # Plot cities
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue')

        # Plot tour
        for i in range(len(tour) - 1):
            city1 = tour[i]
            city2 = tour[i + 1]

            plt.plot([points[city1][0], points[city2][0]], [points[city1][1], points[city2][1]], color='red')

        # Connect last city to the starting city
        plt.plot([points[tour[-1]][0], points[tour[0]][0]], [points[tour[-1]][1], points[tour[0]][1]], color='red')

        plt.title(f'TSP {self.last_used_algorithm} Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

    @property
    def cities(self):
        return self._cities

    @cities.setter
    def cities(self, value: List[Tuple[int, int]]):
        self._cities = value

    @property
    def tour(self):
        return self._tour

    @tour.setter
    def tour(self, value):
        self._tour = value

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = value

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value):
        self._lower_bound = value

    @property
    def last_used_algorithm(self):
        return self._last_used_algorithm

    @last_used_algorithm.setter
    def last_used_algorithm(self, value):
        self._last_used_algorithm = value
