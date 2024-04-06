from utils import calculate_distance


def greedy_tsp(points):
    num_points = len(points)
    remaining_points = set(range(num_points))
    tour = []

    current_point = 0
    tour.append(current_point + 1)
    remaining_points.remove(current_point)

    total_distance = 0

    while remaining_points:
        nearest_point = min(remaining_points, key=lambda x: calculate_distance(points[current_point], points[x]))
        tour.append(nearest_point + 1)
        remaining_points.remove(nearest_point)
        total_distance += calculate_distance(points[current_point], points[nearest_point])
        current_point = nearest_point

    # Add return to the starting point
    tour.append(1)
    # Add distance to return to the starting point
    total_distance += calculate_distance(points[tour[-2] - 1], points[0])

    return tour, total_distance
