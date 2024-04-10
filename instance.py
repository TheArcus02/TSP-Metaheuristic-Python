import random


class Instance:
    def __init__(self):
        self._cities: list[tuple[int, int]] = []

    def get_from_file(self, file_path: str) -> None:
        cities = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 3:
                    continue
                city_number = int(parts[0])
                x_coordinate = int(parts[1])
                y_coordinate = int(parts[2])
                cities.append((x_coordinate, y_coordinate))
        self.cities = cities

    def generate_cities(self, num_cities: int, min_coord: int, max_coord: int) -> list[tuple[int, int]]:
        cities = []
        for _ in range(num_cities):
            x_coordinate = random.randint(min_coord, max_coord)
            y_coordinate = random.randint(min_coord, max_coord)
            cities.append((x_coordinate, y_coordinate))
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

    @property
    def cities(self) -> list[tuple[int, int]]:
        return self._cities

    @cities.setter
    def cities(self, cities: list[tuple[int]]):
        self._cities = cities
