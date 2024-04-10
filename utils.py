import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_tsp_solution(points, tour):
    """
    Plots the tour of the tsp algorithm
    :param points:
    :param tour:
    :return:
    """

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


def calculate_lower_bound(points):
    """
    Calculate the lower bound of the solution
    :param points:
    :return:
    """

    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = calculate_distance(points[i], points[j])
            G.add_edge(i, j, weight=distance)

    MST = nx.minimum_spanning_tree(G)

    # Calculate lower bound as the sum of weights of edges in MST
    lower_bound = sum(MST[i][j]['weight'] for i, j in MST.edges())

    return lower_bound
