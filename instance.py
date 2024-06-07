import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt


class Instance:
    def __init__(self):
        self._cities: List[Tuple[int, int]] = []
        self._filename: str | None = None

    def get_from_file(self, file_path: str) -> None:
        cities = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 3:
                    continue
                city_number = float(parts[0])
                x_coordinate = float(parts[1])
                y_coordinate = float(parts[2])
                cities.append((x_coordinate, y_coordinate))
        self.filename = os.path.splitext(os.path.basename(file_path))[0]
        self.cities = cities

    def generate_cities(self, num_cities: int, min_coord: int, max_coord: int) -> List[Tuple[int, int]]:
        cities = []
        existing_coordinates = set()  # Keep track of existing coordinates

        while len(cities) < num_cities:
            x_coordinate = random.randint(min_coord, max_coord)
            y_coordinate = random.randint(min_coord, max_coord)
            coordinates = (x_coordinate, y_coordinate)

            # Check if the coordinates already exist
            if coordinates not in existing_coordinates:
                cities.append(coordinates)
                existing_coordinates.add(coordinates)

        self.cities = cities
        return cities

    def create_file(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            file.write(f'{len(self.cities)}\n')
            for i, city in enumerate(self.cities):
                file.write(str(i + 1))
                file.write(f' {city[0]} {city[1]}\n')

    def get_length(self) -> int:
        return len(self._cities)

    def plot_cities(self) -> None:
        if not self.cities:
            print("No cities to plot.")
            return

        x_coords, y_coords = zip(*self.cities)
        plt.figure(figsize=(8, 6))
        plt.scatter(x_coords, y_coords, c='blue', marker='o')
        plt.title('Cities Plot')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

    @property
    def cities(self) -> List[Tuple[int, int]]:
        return self._cities

    @cities.setter
    def cities(self, cities: List[Tuple[int]]):
        self._cities = cities

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        self._filename = filename
