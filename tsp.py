import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class TSP:
    def __init__(self, cities: list[tuple[int, int]]):
        self._cities = cities
        self._tour = None
        self._distance = None
        self._lower_bound = None

    def __repr__(self):
        return (f'The tour is {list(map(lambda x: x + 1, self.tour))}\nThe tour distance is {self.distance}'
                f'\nThe lower bound is {self.lower_bound}')

    def run_greedy(self):
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
                                key=lambda x: self.calculate_distance(points[current_point], points[x]))
            tour.append(nearest_point)
            remaining_points.remove(nearest_point)
            total_distance += self.calculate_distance(points[current_point], points[nearest_point])
            current_point = nearest_point

        # Add return to the starting point
        tour.append(0)
        # Add distance to return to the starting point
        total_distance += self.calculate_distance(points[tour[-2]], points[0])

        self.distance = total_distance
        self.tour = tour

        return tour, total_distance

    def calculate_lower_bound(self):
        points = self.cities
        G = nx.Graph()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = self.calculate_distance(points[i], points[j])
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

        plt.title('TSP Greedy Solution')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

    @staticmethod
    def calculate_distance(point1, point2):
        """
        Calculate the distance between two points
        :param point1:
        :param point2:
        :return:
        """
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @property
    def cities(self):
        return self._cities

    @cities.setter
    def cities(self, value: list[tuple[int, int]]):
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
