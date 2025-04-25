import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Benchmark Functions

class Ackley:
    @staticmethod
    def evaluate(x, y):
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = 2  # Dimensions
        sum1 = x**2 + y**2
        sum2 = np.cos(c * x) + np.cos(c * y)
        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

class Bukin:
    @staticmethod
    def evaluate(x, y):
        return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)

class CrossInTray:
    @staticmethod
    def evaluate(x, y):
        return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi))) + 1)**0.1

class DropWave:
    @staticmethod
    def evaluate(x, y):
        return -(1 + np.cos(12 * np.sqrt(x**2 + y**2))) / (0.5 * (x**2 + y**2) + 2)

class EggHolder:
    @staticmethod
    def evaluate(x, y):
        return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

# Differential Evolution

class DifferentialEvolution:
    def __init__(self, objective_function, num_dimensions, population_size, mutation_factor, crossover_rate, max_generations):
        self.objective_function = objective_function
        self.num_dimensions = num_dimensions
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = np.random.uniform(low=-5, high=5, size=(population_size, num_dimensions))

    def evolve(self):
        for generation in range(self.max_generations):
            for i in range(self.population_size):
                target_vector = self.population[i]
                random_indices = np.random.choice(range(self.population_size), size=3, replace=False)
                a, b, c = self.population[random_indices]
                mutant_vector = a + self.mutation_factor * (b - c)

                # Crossover
                trial_vector = np.where(np.random.rand(self.num_dimensions) < self.crossover_rate, mutant_vector, target_vector)

                # Selection
                target_fitness = self.objective_function.evaluate(*target_vector)
                trial_fitness = self.objective_function.evaluate(*trial_vector)
                if trial_fitness < target_fitness:
                    self.population[i] = trial_vector

            # Report progress
            best_fitness = min(self.objective_function.evaluate(*individual) for individual in self.population)
            print(f"Generation {generation}: DE Best Fitness = {best_fitness}")

    def get_best_solution(self):
        return min(self.population, key=lambda ind: self.objective_function.evaluate(*ind))

# Genetic Algorithm

class GeneticAlgorithm:
    def __init__(self, objective_function, num_dimensions, population_size, mutation_rate, crossover_rate, max_generations):
        self.objective_function = objective_function
        self.num_dimensions = num_dimensions
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = np.random.uniform(low=-5, high=5, size=(population_size, num_dimensions))

    def evolve(self):
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness = np.array([self.objective_function.evaluate(*individual) for individual in self.population])

            # Select parents
            parents = self.selection(fitness)

            # Apply crossover and mutation
            offspring = self.crossover(parents)
            offspring = self.mutation(offspring)

            # Evaluate offspring fitness
            offspring_fitness = np.array([self.objective_function.evaluate(*individual) for individual in offspring])

            # Select survivors
            self.population = self.survivor_selection(self.population, fitness, offspring, offspring_fitness)

            # Report progress
            best_fitness = np.min(fitness)
            print(f"Generation {generation}: GA Best Fitness = {best_fitness}")

    def selection(self, fitness):
        # Roulette wheel selection
        probabilities = abs(fitness / np.sum(fitness))
        return self.population[np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)]

    def crossover(self, parents):
        # Single-point crossover
        offspring = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.num_dimensions)
                offspring[i] = np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
                offspring[i+1] = np.concatenate((parents[i+1][:crossover_point], parents[i][crossover_point:]))
            else:
                offspring[i] = parents[i]
                offspring[i+1] = parents[i+1]
        return offspring

    def mutation(self, offspring):
        # Gaussian mutation
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_index = np.random.randint(0, self.num_dimensions)
                offspring[i][mutation_index] += np.random.normal(0, 0.1)
        return offspring

    def survivor_selection(self, population, fitness, offspring, offspring_fitness):
        # Replace worst individuals with offspring
        combined_population = np.concatenate((population, offspring))
        combined_fitness = np.concatenate((fitness, offspring_fitness))
        sorted_indices = np.argsort(combined_fitness)[:self.population_size]
        return combined_population[sorted_indices]

    def get_best_solution(self):
        return min(self.population, key=lambda ind: self.objective_function.evaluate(*ind))

# Plotting the functions with DE and GA predictions

def plot_2d_function_and_predictions(benchmark_function, de, ga, x_range=(-5, 5), y_range=(-5, 5)):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = benchmark_function.evaluate(X, Y)

    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, levels=100, cmap='viridis')
    ax.set_title(f'{benchmark_function.__name__} Function')

    best_solution_de = de.get_best_solution()
    ax.plot(best_solution_de[0], best_solution_de[1], 'ro', label='Best DE Solution')

    best_solution_ga = ga.get_best_solution()
    ax.plot(best_solution_ga[0], best_solution_ga[1], 'go', label='Best GA Solution')

    ax.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_3d_function_and_predictions(benchmark_function, de, ga, x_range=(-5, 5), y_range=(-5, 5)):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = benchmark_function.evaluate(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    best_solution_de = de.get_best_solution()
    ax.scatter(best_solution_de[0], best_solution_de[1], benchmark_function.evaluate(*best_solution_de), color='r', label='Best DE Solution')

    best_solution_ga = ga.get_best_solution()
    ax.scatter(best_solution_ga[0], best_solution_ga[1], benchmark_function.evaluate(*best_solution_ga), color='g', label='Best GA Solution')

    ax.set_title(f'{benchmark_function.__name__} Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.legend()
    plt.show()

# Example usage
benchmark_functions = [Ackley, Bukin, CrossInTray, DropWave]
for benchmark_function in benchmark_functions:
    print(f"Optimizing {benchmark_function.__name__} Function with DE and GA")
    de = DifferentialEvolution(objective_function=benchmark_function, num_dimensions=2, population_size=50, mutation_factor=0.8, crossover_rate=0.9, max_generations=100)
    ga = GeneticAlgorithm(objective_function=benchmark_function, num_dimensions=2, population_size=50, mutation_rate=0.1, crossover_rate=0.7, max_generations=100)
    de.evolve()
    ga.evolve()
    plot_2d_function_and_predictions(benchmark_function, de, ga, x_range=(-10, 10), y_range=(-10, 10))
    plot_3d_function_and_predictions(benchmark_function, de, ga, x_range=(-10, 10), y_range=(-10, 10))
print(f"Optimizing Eggholder Function with DE and GA")
de = DifferentialEvolution(objective_function=EggHolder, num_dimensions=2, population_size=50, mutation_factor=0.8, crossover_rate=0.9, max_generations=100)
ga = GeneticAlgorithm(objective_function=EggHolder, num_dimensions=2, population_size=50, mutation_rate=0.1, crossover_rate=0.7, max_generations=100)
de.evolve()
ga.evolve()
plot_2d_function_and_predictions(EggHolder, de, ga, x_range=(-512, 512), y_range=(-512, 512))
plot_3d_function_and_predictions(EggHolder, de, ga, x_range=(-512, 512), y_range=(-512, 512))