import random


class Instance(object):
    def __init__(self):
        self.cities: list[tuple[float, float]] = []

    def get_from_file(self, file_path: str) -> None:
        cities = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 3:
                    continue
                city_number = int(parts[0])
                x_coordinate = float(parts[1])
                y_coordinate = float(parts[2])
                cities.append((x_coordinate, y_coordinate))
        self.cities = cities

    def generate_cities(self, num_cities: int, min_coord: int, max_coord: int) -> list[tuple[float, float]]:
        cities = []
        for _ in range(num_cities):
            x_coordinate = random.randint(min_coord, max_coord)
            y_coordinate = random.randint(min_coord, max_coord)
            cities.append((x_coordinate, y_coordinate))
        self.cities = cities
        return cities

    def create_file(self, file_path: str) -> None:
        with open(f'data/{file_path}', 'w') as file:
            file.write(f'{len(self.cities)}\n')
            for i, city in enumerate(self.cities):
                file.write(str(i + 1))
                file.write(f' {city[0]} {city[1]}\n')

    def get_length(self) -> int:
        return len(self.cities)

    def get_cities(self) -> list[tuple[float, float]]:
        return self.cities
