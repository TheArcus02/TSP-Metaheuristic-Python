import time
from typing import List, Tuple

import numpy as np

from ant import Ant


class ACO:
    def __init__(self, cities: List[Tuple[int, int]], num_ants: int,
                 num_iterations: int, alpha: float, beta: float, rho: float):
        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.pheromones = np.ones((self.num_cities, self.num_cities)) / self.num_cities

    def run(self, max_time=3 * 60) -> Tuple[list, float]:
        # Initialize best tour and best tour length
        best_tour = None
        best_tour_length = float('inf')
        start_time = time.perf_counter()

        # Run ACO for a given number of iterations
        for _ in range(self.num_iterations):

            if time.perf_counter() - start_time > max_time:
                print("Time limit exceeded. Stopping the algorithm")
                break

            # Create ant instances
            ants = [Ant(self.cities, self.pheromones, self.alpha, self.beta) for _ in range(self.num_ants)]

            # Each ant constructs a tour
            for ant in ants:
                ant.construct_tour()

                # Update best tour if a shorter tour is found
                if ant.tour_length < best_tour_length:
                    best_tour_length = ant.tour_length
                    best_tour = ant.tour

            # Update pheromone levels
            self._update_pheromones(ants)

        return best_tour, best_tour_length

    def _update_pheromones(self, ants: List) -> None:
        # Evaporate pheromones
        self.pheromones *= (1 - self.rho)

        # Update pheromones based on ant tours
        for ant in ants:
            for i in range(len(ant.tour) - 1):
                city1, city2 = ant.tour[i], ant.tour[i + 1]
                self.pheromones[city1][city2] += 1 / ant.tour_length
