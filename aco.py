import time
from typing import List, Tuple

import numpy as np

from ant import Ant
from utils import calculate_distance


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
        self.distances = self._calculate_distance_matrix()

    def run(self, max_time=3 * 60, use_multiprocessing=False, verbose=False) -> Tuple[List, float]:
        if use_multiprocessing:
            return self._run_with_multiprocessing(max_time, verbose)
        return self._run_sequential(max_time, verbose)

    def _run_sequential(self, max_time, verbose) -> Tuple[list, float]:
        # Initialize best tour and best tour length
        best_tour = None
        best_tour_length = float('inf')
        start_time = time.perf_counter()

        # Run ACO for a given number of iterations
        for iteration in range(self.num_iterations):
            iteration_start_time = time.perf_counter()
            if time.perf_counter() - start_time > max_time:
                print("Time limit exceeded. Stopping the algorithm")
                break

            # Create ant instances
            ants = [Ant(self.cities, self.pheromones, self.alpha, self.beta, self.distances) for _ in
                    range(self.num_ants)]

            # Each ant constructs a tour
            for ant in ants:
                ant.construct_tour()

                # Update best tour if a shorter tour is found
                if ant.tour_length < best_tour_length:
                    best_tour_length = ant.tour_length
                    best_tour = ant.tour

            # Update pheromone levels
            self._update_pheromones(ants)
            if verbose:
                print(
                    f"Iteration {iteration + 1}/{self.num_iterations}: Best tour length so far is {best_tour_length},"
                    f" took: {time.perf_counter() - iteration_start_time}")

        return best_tour, best_tour_length

    def _run_with_multiprocessing(self, max_time, verbose) -> Tuple[list, float]:
        from multiprocessing import Pool

        best_tour = None
        best_tour_length = float('inf')
        start_time = time.perf_counter()

        for iteration in range(self.num_iterations):
            iteration_start_time = time.perf_counter()
            if time.perf_counter() - start_time > max_time:
                print("Time limit exceeded. Stopping the algorithm")
                break

            with Pool() as pool:
                ants_per_process = self.num_ants // pool._processes
                remainder = self.num_ants % pool._processes

                batch_sizes = [ants_per_process + (1 if i < remainder else 0) for i in range(pool._processes)]

                ant_batches = pool.map(self._create_and_construct_ants_batch, batch_sizes)

                ants = [ant for batch in ant_batches for ant in batch]

            for ant in ants:
                if ant.tour_length < best_tour_length:
                    best_tour_length = ant.tour_length
                    best_tour = ant.tour
            self._update_pheromones(ants)

            if verbose:
                print(
                    f"Iteration {iteration + 1}/{self.num_iterations}: Best tour length so far is {best_tour_length},"
                    f" took: {time.perf_counter() - iteration_start_time}")

        return best_tour, best_tour_length

    def _create_and_construct_ants_batch(self, num_ants: int) -> List[Ant]:
        ants = [Ant(self.cities, self.pheromones, self.alpha, self.beta, self.distances) for _ in range(num_ants)]
        for ant in ants:
            ant.construct_tour()
        return ants

    def _update_pheromones(self, ants: List) -> None:
        # Evaporate pheromones
        self.pheromones *= (1 - self.rho)

        # Update pheromones based on ant tours
        for ant in ants:
            for i in range(len(ant.tour) - 1):
                city1, city2 = ant.tour[i], ant.tour[i + 1]
                self.pheromones[city1][city2] += 1 / ant.tour_length

    def _calculate_distance_matrix(self) -> np.ndarray:
        num_cities = self.num_cities
        distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                distance = calculate_distance(self.cities[i], self.cities[j])
                distances[i, j] = distances[j, i] = distance
        return distances
