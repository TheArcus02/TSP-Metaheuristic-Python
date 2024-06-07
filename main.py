import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from instance import Instance
from tsp import TSP

optimal_results = {
    'a280': 2579,
    'ch130': 6110,
    'ch150': 6528,
    'd198': 15780,
    'd493': 35002,
    'kroA100': 21282,
    'kroA150': 26524,
    'kroB100': 22141,
    'kroB150': 26130,
    'lin318': 42029,
    'berlin52': 7544,
    'bier127': 118282
}


def compare_greedy_and_aco(num_measurements=15, start_num_cities=50, step_size=25):
    # Results storage
    aco_results = []
    greedy_results = []

    for i in range(num_measurements):
        num_cities = start_num_cities + (i * step_size)

        # Generate instance
        instance = Instance()
        instance.generate_cities(num_cities, 0, 1000)
        print(f'cities: {instance.cities}')

        # Greedy algorithm
        tsp = TSP(instance.cities)
        tsp.run_greedy()
        greedy_results.append(tsp.distance)

        # ACO algorithm
        start_time = time.perf_counter()
        tsp.run_aco(
            num_ants=24,
            num_iterations=1000,
            alpha=1.03,
            beta=3.5,
            rho=0.12,
            use_multiprocessing=True,
            verbose=False,
            max_time=60
        )
        end_time = time.perf_counter()
        aco_results.append(tsp.distance)
        print(
            f'Instance {i + 1}/{num_measurements} - ACO Result: {aco_results[-1]}, Greedy Result: {greedy_results[-1]},'
            f' Time: {end_time - start_time:.2f} seconds')

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_measurements), aco_results, color='blue', alpha=0.7, label='ACO')
    plt.bar(np.arange(num_measurements), greedy_results, color='orange', alpha=0.7, label='Greedy')
    plt.xlabel('Instance')
    plt.ylabel('Tour Length')
    plt.title('Comparison of ACO and Greedy Algorithms')
    plt.legend()
    plt.xticks(np.arange(num_measurements), [start_num_cities + i * step_size for i in range(num_measurements)])
    plt.grid(axis='y')
    plt.show()


def run_benchmark_instances():
    benchmark_dir = './data/benchmark'
    instance_files = [f for f in os.listdir(benchmark_dir) if os.path.isfile(os.path.join(benchmark_dir, f))]

    aco_results = []
    instance_names = []

    for file in instance_files:
        instance_name = file.split('.')[0]
        instance_names.append(instance_name)
        instance = Instance()
        instance.get_from_file(os.path.join(benchmark_dir, file))
        instance.plot_cities()

        # Initialize TSP and ACO
        tsp = TSP(instance.cities)

        print(f'Starting TSP for {instance_name}')
        # Run ACO algorithm
        best_tour, best_length = tsp.run_aco(
            num_ants=24,
            num_iterations=1000,
            alpha=1.03,
            beta=3.5,
            rho=0.12,
            use_multiprocessing=True,
            verbose=False,
            max_time=3 * 60,
            optimum=optimal_results[instance_name]
        )

        tsp.plot_solution()

        # Check if the optimum result is found
        if best_length <= optimal_results[instance_name]:
            print(f"Optimum found for {instance_name}: {best_length}")
        else:
            print(f"Best result for {instance_name}: {best_length}")

        aco_results.append(best_length)

    # Calculate relative error
    relative_errors = [(aco_results[i] - optimal_results[instance_name]) / optimal_results[instance_name] * 100
                       for i, instance_name in enumerate(instance_names)]

    # Create a bar plot
    optimal_values = [optimal_results[name] for name in instance_names]

    plt.figure(figsize=(14, 8))
    width = 0.35  # width of the bars
    indices = np.arange(len(instance_names))

    plt.subplot(2, 1, 1)
    plt.bar(indices - width / 2, aco_results, width, label='ACO', color='blue', alpha=0.7)
    plt.bar(indices + width / 2, optimal_values, width, label='Optimum', color='orange', alpha=0.7)
    plt.xlabel('Instance')
    plt.ylabel('Tour Length')
    plt.title('Comparison of ACO and Optimum Results')
    plt.xticks(indices, instance_names, rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    # Create a bar plot for relative errors
    plt.subplot(2, 1, 2)
    plt.bar(indices, relative_errors, width, label='Relative Error', color='red', alpha=0.7)
    plt.xlabel('Instance')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error of ACO Results')
    plt.xticks(indices, instance_names, rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    plt.show()


def run_single_instance_from_file(file_path: str, compare_to_greedy: bool = True):
    instance = Instance()

    instance.get_from_file(file_path)
    instance.plot_cities()
    tsp = TSP(instance.cities)

    if compare_to_greedy:
        tsp.run_greedy()
        print('-----------------------------------------')
        print('Results for Greedy algorithm:')
        print(tsp)
        tsp.plot_solution()

    # ACO
    start_time = time.perf_counter()
    tsp.run_aco(
        num_ants=24,
        num_iterations=20,
        alpha=1.03,
        beta=3.5,
        rho=0.12,
        use_multiprocessing=True,
        verbose=True,
        max_time=60,
        optimum=optimal_results[instance.filename]
    )
    end_time = time.perf_counter()
    print('-----------------------------------------')
    print(f'Full runtime: {end_time - start_time}')
    print('Results for ACO algorithm:')
    print(tsp)
    tsp.plot_solution()


def run_all_ranking_instances():
    benchmark_dir = './data/ranking'
    instance_files = [f for f in os.listdir(benchmark_dir) if os.path.isfile(os.path.join(benchmark_dir, f))]

    for file in instance_files:
        instance_name = file.split('.')[0]
        instance = Instance()
        instance.get_from_file(os.path.join(benchmark_dir, file))
        instance.plot_cities()

        tsp = TSP(instance.cities)

        print(f'Starting TSP for {instance_name}')

        best_tour, best_length = tsp.run_aco(
            num_ants=24,
            num_iterations=1000,
            alpha=1.03,
            beta=3.5,
            rho=0.12,
            use_multiprocessing=True,
            verbose=False,
            max_time=3 * 60 if instance_name != 'tsp1000' else 5 * 60,
            optimum=optimal_results[instance_name]
        )

        tsp.plot_solution()

        if int(best_length) <= optimal_results[instance_name]:
            print(f"Optimum found for {instance_name}: {best_length}")
            print(f'Optimal tour: {best_tour}')
        else:
            print(f"Best result for {instance_name}: {best_length}")
            print(f'The tour is: {best_tour}')


def run_single_instance(num_cities: int, compare_to_greedy: bool = True):
    instance = Instance()

    instance.generate_cities(num_cities, 0, 1000)
    instance.plot_cities()
    tsp = TSP(instance.cities)

    if compare_to_greedy:
        tsp.run_greedy()
        print('-----------------------------------------')
        print('Results for Greedy algorithm:')
        print(tsp)
        tsp.plot_solution()

    # ACO
    start_time = time.perf_counter()
    tsp.run_aco(
        num_ants=24,
        num_iterations=20,
        alpha=1.03,
        beta=3.5,
        rho=0.12,
        use_multiprocessing=True,
        verbose=True,
        max_time=60,
        optimum=7544
    )
    end_time = time.perf_counter()
    print('-----------------------------------------')
    print(f'Full runtime: {end_time - start_time}')
    print('Results for ACO algorithm:')
    print(tsp)
    tsp.plot_solution()


def tune_parameters_on_instance(file_path: str, parameters: Dict[str, List]):
    instance = Instance()

    instance.get_from_file(file_path)
    tsp = TSP(instance.cities)

    best_params, best_len, best_tour = tsp.tune_aco_parameters(parameters, 20, verbose=True,
                                                               aco_multiprocessing=True)


if __name__ == '__main__':
    # compare_greedy_and_aco()

    # run_benchmark_instances()

    # run_single_instance(num_cities=60)

    # run_single_instance_from_file(file_path='data/ranking/berlin52.txt',
    #                               compare_to_greedy=True)

    run_all_ranking_instances()

    # GridSearch Tuning
    # parameter_values = {
    #     'num_ants': np.arange(200, 300, 20),
    #     'num_iterations': [200, 400, 600],
    #     'alpha': np.arange(1, 4, 0.5),
    #     'beta': np.arange(2, 5, 0.5),
    #     'rho': np.arange(0.1, 0.4, 0.1)
    # }
    # tune_parameters_on_instance('data/berlin52.txt', parameter_values)
