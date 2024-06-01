import numpy as np

from instance import Instance
from tsp import TSP

if __name__ == '__main__':
    file_path = 'data/tsp1000.txt'

    instance = Instance()
    instance.get_from_file(file_path)
    # instance.generate_cities(10, 1, 10)
    print(f'cities: {instance.cities}')
    tsp = TSP(instance.cities)

    lower_bound = tsp.calculate_lower_bound()

    # Prepere the ACO
    parameter_values = {
        'num_ants': np.arange(10, 110, 10),
        'num_iterations': [100],
        'alpha': np.arange(1, 6, 0.5),
        'beta': np.arange(1, 6, 0.5),
        'rho': np.arange(0.1, 0.6, 0.1)
    }
    best_params, best_len, best_tour = tsp.tune_aco_parameters(parameter_values, 500, log=True)

    # Greedy
    tsp.run_greedy()
    print('-----------------------------------------')
    print('Results for Greedy algorithm:')
    print(tsp)
    tsp.plot_solution()

    # ACO
    tsp.run_aco(
        num_ants=best_params['num_ants'],
        num_iterations=best_params['num_iterations'],
        alpha=best_params['alpha'],
        beta=best_params['beta'],
        rho=best_params['rho'],
    )
    # tsp.run_aco(
    #     num_ants=1,
    #     num_iterations=1,
    #     alpha=2,
    #     beta=2,
    #     rho=0.1,
    # )
    print('-----------------------------------------')
    print('Results for ACO algorithm:')
    print(tsp)
    tsp.plot_solution()
