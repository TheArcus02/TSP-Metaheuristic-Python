from instance import Instance
from tsp import TSP

if __name__ == '__main__':
    file_path = 'data/berlin52.txt'

    instance = Instance()
    # instance.get_from_file(file_path)
    instance.generate_cities(5, 1, 100)
    print(f'cities: {instance.cities}')
    tsp = TSP(instance.cities)

    lower_bound = tsp.calculate_lower_bound()

    # Greedy
    tsp.run_greedy()
    print('Results for Greedy algorithm:')
    print(tsp)
    tsp.plot_solution()

    print('-----------------------------------------')
    # ACO
    tsp.run_aco(
        num_ants=5,
        num_iterations=5,
        alpha=1,
        beta=5,
        rho=0.3
    )
    print('Results for ACO algorithm:')
    print(tsp)
    tsp.plot_solution()
