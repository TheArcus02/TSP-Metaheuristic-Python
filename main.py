import time

import numpy as np

from instance import Instance
from tsp import TSP

if __name__ == '__main__':
    file_path = 'data/berlin52.txt'

    instance = Instance()
    instance.get_from_file(file_path)
    # instance.generate_cities(10, 1, 10)
    print(f'cities: {instance.cities}')
    tsp = TSP(instance.cities)

    lower_bound = tsp.calculate_lower_bound()

    # Prepere the ACO
    parameter_values = {
        'num_ants': np.arange(200, 300, 20),
        'num_iterations': [200, 400, 600],
        'alpha': np.arange(1, 4, 0.5),
        'beta': np.arange(2, 5, 0.5),
        'rho': np.arange(0.1, 0.4, 0.1)
    }

    # best_params, best_len, best_tour = tsp.tune_aco_parameters(parameter_values, 20, verbose=True,
    #                                                            aco_multiprocessing=True)

    # Greedy
    tsp.run_greedy()
    print('-----------------------------------------')
    print('Results for Greedy algorithm:')
    print(tsp)
    tsp.plot_solution()

    # ACO
    start_time = time.perf_counter()
    tsp.run_aco(
        num_ants=23,
        num_iterations=1000,
        alpha=1.03,
        beta=3.5,
        rho=0.12,
        use_multiprocessing=True,
        verbose=True
    )
    end_time = time.perf_counter()
    print('-----------------------------------------')
    print(f'Full runtime: {end_time - start_time}')
    print('Results for ACO algorithm:')
    print(tsp)
    tsp.plot_solution()
