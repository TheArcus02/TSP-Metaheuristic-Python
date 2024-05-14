from typing import List, Tuple, Set, Dict

import numpy as np

from utils import calculate_distance


class Ant:
    def __init__(self, cities: List[Tuple[int, int]], pheromones: np.ndarray, alpha: float, beta: float):
        self.cities = cities
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.tour = []
        self.tour_length = 0

    def construct_tour(self) -> None:
        num_cities = len(self.cities)
        current_city = np.random.randint(num_cities)  # Start from a random city
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

        # Calculate the length of the tour
        self.calculate_tour_length()

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

        for city in probabilities:
            probabilities[city] /= total

        return probabilities

    def _calculate_desirability(self, pheromone: float, current_city: int, next_city: int) -> float:
        distance = calculate_distance(self.cities[current_city], self.cities[next_city])
        return (pheromone ** self.alpha) * (1.0 / distance) ** self.beta

    def calculate_tour_length(self) -> None:
        total_distance = 0
        for i in range(len(self.tour) - 1):
            city1 = self.cities[self.tour[i]]
            city2 = self.cities[self.tour[i + 1]]
            total_distance += calculate_distance(city1, city2)

        self.tour_length = total_distance
