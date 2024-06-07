from typing import List, Tuple, Set, Dict

import numpy as np


class Ant:
    def __init__(self, cities: List[Tuple[int, int]], pheromones: np.ndarray, alpha: float, beta: float,
                 distances: np.ndarray, key: int):
        self.cities = cities
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.distances = distances
        self.tour = []
        self.tour_length = float('inf')
        self.key = key

    def construct_tour(self) -> None:
        num_cities = len(self.cities)
        current_city = self.key % num_cities  # Start from a random city
        self.tour.append(current_city)
        visited = {current_city}

        # Build the tour by choosing the next city until all cities are visited
        while len(self.tour) < num_cities:
            next_city = self._choose_next_city(current_city, visited)
            self.tour.append(next_city)
            visited.add(next_city)
            current_city = next_city

        # Add path from last city to first
        self.tour.append(self.tour[0])

        # Apply two opt
        # self.tour = self._two_opt(self.tour)
        self.tour_length = self._calculate_tour_length(self.tour)

    def _choose_next_city(self, current_city: int, visited: Set[int]) -> int:
        # Compute probabilities for choosing the next city
        probabilities = self._calculate_probabilities(current_city, visited)

        # Choose the next city based on the probabilities
        return np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))

    def _calculate_probabilities(self, current_city: int, visited: Set[int]) -> Dict[int, float]:
        probabilities = {}
        pheromones = self.pheromones[current_city]
        total = 0

        for city, pheromone in enumerate(pheromones):
            if city not in visited:
                desirability = self._calculate_desirability(pheromone, current_city, city)
                total += desirability
                probabilities[city] = desirability

        if total == 0:
            for city in probabilities:
                probabilities[city] = 1 / len(probabilities)
        else:
            for city in probabilities:
                probabilities[city] /= total

        return probabilities

    def _calculate_desirability(self, pheromone: float, current_city: int, next_city: int) -> float:
        distance = self.distances[current_city, next_city]
        if distance == 0:
            return 0.0
        return (pheromone ** self.alpha) * (1.0 / distance) ** self.beta

    def _calculate_tour_length(self, tour) -> int:
        total_distance = 0
        for i in range(len(tour) - 1):
            total_distance += self.distances[tour[i], tour[i + 1]]
        return total_distance

    def two_opt(self, check_if_time_limit_exceeded):
        n = len(self.tour)
        improved = True

        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if check_if_time_limit_exceeded():
                        return
                    if j - i == 1:
                        continue
                    new_tour = self.tour[:]
                    new_tour[i:j] = reversed(self.tour[i:j])
                    new_distance = self._calculate_tour_length(new_tour)
                    if new_distance < self.tour_length:
                        self.tour = new_tour
                        self.tour_length = new_distance
                        improved = True
