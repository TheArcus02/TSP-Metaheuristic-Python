import time
from multiprocessing import Pool
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ant import Ant
from utils import calculate_distance


class ACO:
    def __init__(self, cities: List[Tuple[int, int]], num_ants: int,
                 num_iterations: int, alpha: float, beta: float, rho: float,
                 max_time: int, optimum: int = None, verbose: bool = False):
        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.pheromones = self._calculate_initial_pheromones()
        self.distances = self._calculate_distance_matrix()
        self.optimum = optimum
        self.verbose = verbose

        self.best_tour_lengths = []
        self.start_time = None
        self.max_time = max_time

        self.best_tour = None
        self.best_tour_length = float('inf')
        self.intensified_search = False

    def _init_start_time(self):
        self.start_time = time.perf_counter()

    def run(self, use_multiprocessing=False) -> Tuple[List, float]:
        self._init_start_time()
        if use_multiprocessing:
            return self._run_with_multiprocessing()
        return self._run_sequential()

    def _run_sequential(self) -> Tuple[List, float]:
        stagnation_count = 0
        stagnation_threshold = 6
        reset_counter = 0
        reset_threshold = 15

        iteration = 0
        while iteration < self.num_iterations and time.perf_counter() - self.start_time < self.max_time:
            iteration_start_time = time.perf_counter()
            new_best_tour = False

            ants = [Ant(self.cities, self.pheromones, self.alpha, self.beta, self.distances, i) for i in
                    range(self.num_ants)]
            local_best_tour = None

            for ant in ants:
                if self._check_if_time_limit_exceeded():
                    return self.best_tour, self.best_tour_length

                ant.construct_tour()
                if ((ant.tour_length < self.best_tour_length or self.intensified_search)
                        and (local_best_tour is None or ant.tour_length < local_best_tour)):
                    ant.two_opt(self._check_if_time_limit_exceeded)

                    if local_best_tour is None or ant.tour_length < local_best_tour:
                        local_best_tour = ant.tour_length

                    if ant.tour_length < self.best_tour_length:
                        if self.intensified_search:
                            self.intensified_search = False
                        self.best_tour_length = ant.tour_length
                        self.best_tour = ant.tour
                        new_best_tour = True

                self._update_pheromones(ant, self.intensified_search)

                if self.optimum and int(self.best_tour_length) <= self.optimum:
                    if self.verbose:
                        print(f"Optimum {int(self.best_tour_length)} found")
                    self.best_tour_lengths.append(self.best_tour_length)
                    return self.best_tour, self.best_tour_length

            stagnation_count, reset_counter = self._update_stagnation_and_reset(
                new_best_tour, stagnation_count, stagnation_threshold, reset_counter, reset_threshold)

            if self.verbose:
                print(
                    f"Iteration {iteration + 1}/{self.num_iterations}: Best tour length so far is {self.best_tour_length},"
                    f" took: {time.perf_counter() - iteration_start_time}")

            self.best_tour_lengths.append(self.best_tour_length)

            iteration += 1
        return self.best_tour, self.best_tour_length

    def _run_with_multiprocessing(self):
        stagnation_count = 0
        stagnation_threshold = 6
        reset_counter = 0
        reset_threshold = 15

        num_processes, batch_sizes = self._init_multiprocessing()

        iteration = 0
        while iteration < self.num_iterations and time.perf_counter() - self.start_time < self.max_time:
            iteration_start_time = time.perf_counter()
            new_best_tour = False

            with Pool(processes=num_processes) as pool:
                new_best_tour = self._process_batches(pool, batch_sizes)

                if self._check_if_optimum_found():
                    break

            stagnation_count, reset_counter = self._update_stagnation_and_reset(
                new_best_tour, stagnation_count, stagnation_threshold, reset_counter, reset_threshold)

            if self.verbose:
                print(
                    f"Iteration {iteration + 1}/{self.num_iterations}: Best tour length so far is {self.best_tour_length},"
                    f" took: {time.perf_counter() - iteration_start_time}")

            self.best_tour_lengths.append(self.best_tour_length)

            iteration += 1
        return self.best_tour, self.best_tour_length

    def _check_if_optimum_found(self):
        if self.optimum and int(self.best_tour_length) == self.optimum:
            if self.verbose:
                print(f"Optimum {int(self.best_tour_length)} found")
            return True
        return False

    def _init_multiprocessing(self) -> Tuple[int, list]:
        num_processes = min(self.num_ants, Pool()._processes)
        ants_per_process = self.num_ants // num_processes
        remainder = self.num_ants % num_processes
        batch_sizes = [ants_per_process + (1 if i < remainder else 0) for i in range(num_processes)]
        return num_processes, batch_sizes

    def _process_batches(self, pool, batch_sizes):
        new_best_tour = False
        for batch_size in batch_sizes:
            ant_batches = pool.map(self._construct_ant_and_process, [batch_size] * len(batch_sizes))

            for ants in ant_batches:
                if ants is None:
                    break

                for ant in ants:
                    self._update_pheromones(ant, self.intensified_search)

                    if ant.tour_length < self.best_tour_length:
                        self.best_tour_length = ant.tour_length
                        self.best_tour = ant.tour
                        new_best_tour = True
                    if self._check_if_optimum_found():
                        break

        return new_best_tour

    def _construct_ant_and_process(self, batch_size):
        ants = [Ant(self.cities, self.pheromones, self.alpha, self.beta, self.distances, i) for i in range(batch_size)]
        local_best_tour = None
        for ant in ants:
            if self._check_if_time_limit_exceeded():
                return ants
            ant.construct_tour()
            if ((ant.tour_length < self.best_tour_length or self.intensified_search)
                    and (local_best_tour is None or ant.tour_length < local_best_tour)):
                ant.two_opt(self._check_if_time_limit_exceeded)
                if local_best_tour is None or ant.tour_length < local_best_tour:
                    local_best_tour = ant.tour_length
                if ant.tour_length < self.best_tour_length and self.intensified_search:
                    self.intensified_search = False
        return ants

    def _update_stagnation_and_reset(self, new_best_tour, stagnation_count, stagnation_threshold, reset_counter,
                                     reset_threshold):
        if new_best_tour:
            stagnation_count = 0
            reset_counter = 0
        else:
            stagnation_count += 1
            reset_counter += 1

        if reset_counter >= reset_threshold:
            if self.verbose:
                print(f'No improvements in {reset_threshold} iterations, resetting the pheromone matrix')
            self.pheromones = self._calculate_initial_pheromones()
            reset_counter = 0

        if stagnation_count >= stagnation_threshold:
            if self.verbose:
                print(f'No improvements in {stagnation_threshold} iterations, applying intensified search')
            self.intensified_search = True

        return stagnation_count, reset_counter

    def _check_if_time_limit_exceeded(self):
        if time.perf_counter() - self.start_time > self.max_time:
            print('Time limit exceeded. Stopping the algorithm')
            return True
        return False

    def _create_and_construct_ants_batch(self, num_ants: int) -> List[Ant]:
        ants = [Ant(self.cities, self.pheromones, self.alpha, self.beta, self.distances, i) for i in range(num_ants)]
        for ant in ants:
            ant.construct_tour()
        return ants

    def _update_pheromones(self, ant: Ant, intensify=False) -> None:
        rho = self.rho * 1.5 if intensify else self.rho
        self.pheromones *= (1 - rho)
        # Update pheromones based on ant tours
        for i in range(len(ant.tour) - 1):
            city1, city2 = ant.tour[i], ant.tour[i + 1]
            self.pheromones[city1][city2] += 1 / ant.tour_length
            self.pheromones[city2][city1] += 1 / ant.tour_length

    def _calculate_distance_matrix(self) -> np.ndarray:
        num_cities = self.num_cities
        distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i + 1, num_cities):
                distance = calculate_distance(self.cities[i], self.cities[j])
                distances[i, j] = distances[j, i] = distance
        return distances

    def _calculate_initial_pheromones(self) -> np.ndarray:
        return np.ones((self.num_cities, self.num_cities)) * 3

    def plot_best_tour_lengths(self) -> None:
        if not self.best_tour_lengths:
            print("No tour lengths to plot.")
            return

        iterations = range(1, len(self.best_tour_lengths) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.best_tour_lengths, marker='o', linestyle='-', color='b')
        plt.title('Best Tour Lengths Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Tour Length')
        plt.grid(True)
        plt.show()

    def _plot_tour(self, title: str) -> None:
        points = self.cities
        tour = self.best_tour

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

        plt.title(title)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()
