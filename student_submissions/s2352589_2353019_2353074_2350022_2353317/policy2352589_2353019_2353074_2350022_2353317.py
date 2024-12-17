from policy import Policy
import numpy as np


class Policy2352589_2353019_2353074_2350022_2353317(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = FirstFitDecreasing()
        elif policy_id == 2:
            self.policy = GeneticAlgorithm()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed


class FirstFitDecreasing(Policy):
    def __init__(self):
        # Tính toán trước kích thước của các kho
        self.stock_dimensions = None


    def get_action(self, observation, info):
        """
        Lựa chọn hành động dựa trên Heuristic First-Fit Decreasing.


        Parameters:
            observation (dict): Chứa 'stocks' và 'products'.
            info (dict): Chứa thông tin bổ sung như 'filled_ratio' và 'trim_loss'.


        Returns:
            action (dict): Chứa 'stock_idx', 'size', và 'position'.
        """
        stocks = observation['stocks']
        products = observation['products']


        # Tính toán kích thước các kho trước
        if self.stock_dimensions is None:
            self.stock_dimensions = []
            for stock in stocks:
                stock_width = np.sum(np.any(stock != -2, axis=1))
                stock_height = np.sum(np.any(stock != -2, axis=0))
                self.stock_dimensions.append((stock_width, stock_height))


        # Tạo danh sách các sản phẩm với diện tích, loại bỏ các sản phẩm hết hàng
        product_list = []
        for idx, product in enumerate(products):
            size = product['size']
            quantity = product['quantity']
            if quantity > 0:
                area = size[0] * size[1]
                product_list.append((area, idx, size, quantity))


        # Sắp xếp các sản phẩm theo diện tích giảm dần
        product_list.sort(reverse=True, key=lambda x: x[0])


        # Duyệt qua các sản phẩm đã sắp xếp
        for _, prod_idx, size, quantity in product_list:
            # Xem xét cả 2 chiều của sản phẩm (gốc và xoay)
            orientations = [size, size[::-1]]
            for oriented_size in orientations:
                width, height = oriented_size


                # Duyệt qua các kho và tìm vị trí đặt sản phẩm
                for stock_idx, stock in enumerate(stocks):
                    stock_width, stock_height = self.stock_dimensions[stock_idx]


                    # Duyệt qua các vị trí có thể trong kho
                    for x in range(stock_width - width + 1):
                        for y in range(stock_height - height + 1):
                            if self._can_place(stock, x, y, width, height):
                                # Nếu tìm được vị trí hợp lệ, trả về hành động ngay
                                action = {
                                    "stock_idx": stock_idx,
                                    "size": np.array([width, height]),
                                    "position": np.array([x, y])
                                }
                                return action


        # Nếu không tìm được vị trí, trả về hành động mặc định
        return {
            "stock_idx": 0,
            "size": np.array([1, 1]),
            "position": np.array([0, 0])
        }


    def _can_place(self, stock, x, y, width, height):
        """
        Kiểm tra xem sản phẩm có thể đặt vào vị trí x, y trong kho hay không.


        Parameters:
            stock (np.ndarray): Trạng thái hiện tại của kho.
            x (int): Tọa độ X của vị trí đặt sản phẩm.
            y (int): Tọa độ Y của vị trí đặt sản phẩm.
            width (int): Chiều rộng của sản phẩm.
            height (int): Chiều cao của sản phẩm.


        Returns:
            bool: True nếu có thể đặt sản phẩm, False nếu không.
        """
        region = stock[x:x+width, y:y+height]
        # Kiểm tra xem tất cả các ô trong vùng có phải là ô trống (-1)
        return np.all(region == -1)


class GeneticAlgorithm(Policy):
    def __init__(self):
        """
        Initializes the Policy class. No parameters are provided during initialization.
        """
        pass


    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]


        for product in products:
            size = product["size"]
            quantity = product["quantity"]


            if quantity <= 0:
                continue  # Skip fulfilled products


            for stock_idx, stock in enumerate(stocks):
                stock_width, stock_height = stock.shape


                # Create a sliding window to find an empty fit
                for x in range(stock_width - size[0] + 1):
                    for y in range(stock_height - size[1] + 1):
                        if np.all(stock[x:x + size[0], y:y + size[1]] == -1):
                            # Early return for efficiency
                            return {
                                "stock_idx": stock_idx,
                                "size": size,
                                "position": (x, y),
                            }


        # Return an invalid action if no valid placement is found
        return {
            "stock_idx": 0,
            "size": (1, 1),
            "position": (0, 0),
        }


    def deterministic_sample(self, low, high, count):
        """Generates a sequence of numbers deterministically within a range."""
        numbers = []
        for i in range(count):
            numbers.append(low + (i % (high - low + 1)))
        return numbers


    def deterministic_shuffle(self, array):
        """Performs a deterministic shuffle based on element positions."""
        n = len(array)
        for i in range(n):
            swap_idx = (i * 2 + 1) % n  # Example deterministic swapping
            array[i], array[swap_idx] = array[swap_idx], array[i]
        return array


    def genetic_algorithm(self, stock_length, demands, population_size=100, generations=20, mutation_rate=0.1):
        """Core logic for the genetic algorithm."""
        def heuristic_initialize_population():
            population = []
            for i in range(population_size):
                individual = []
                for length, quantity in demands:
                    individual += [length] * quantity
                # Group similar lengths for better initial placement
                individual.sort(reverse=True)
                population.append(self.deterministic_shuffle(individual))
            return population


        def fitness(individual):
            total_waste = 0
            total_stock_used = 0
            start = 0


            while start < len(individual):
                stock_used = 0
                for length in individual[start:]:
                    if stock_used + length > stock_length:
                        break
                    stock_used += length
                total_waste += stock_length - stock_used
                total_stock_used += 1
                start += len([x for x in individual[start:] if x + stock_used <= stock_length])


            # Maximize utilization and minimize waste
            return 1 / (1 + total_waste + total_stock_used * 0.1)


        def select_parents(population, fitness_scores):
            """Selects two parents deterministically based on fitness."""
            sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
            return sorted_population[0][0], sorted_population[1][0]


        def crossover(parent1, parent2):
            # Ensure offspring meet demand constraints
            point1, point2 = self.deterministic_sample(0, len(parent1) - 1, 2)
            point1, point2 = sorted((point1, point2))
            offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]


            # Validate and adjust offspring
            def validate_offspring(offspring):
                demand_count = {length: quantity for length, quantity in demands}
                valid_offspring = []
                for length in offspring:
                    if length in demand_count and demand_count[length] > 0:
                        valid_offspring.append(length)
                        demand_count[length] -= 1
                return valid_offspring


            return validate_offspring(offspring1), validate_offspring(offspring2)


        def mutate(individual):
            for i in range(len(individual)):
                if (i + 1) / len(individual) < mutation_rate:  # Deterministic mutation condition
                    # Swap within problematic regions
                    problematic_indices = [
                        idx for idx, length in enumerate(individual)
                        if length > stock_length // 2  # Example criterion
                    ]
                    if problematic_indices:
                        swap_idx = problematic_indices[(i * 3) % len(problematic_indices)]  # Deterministic swap
                        individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
            return individual


        # Initialize the population
        population = heuristic_initialize_population()

        # Evolve the population for a number of generations
        for _ in range(generations):
            fitness_scores = [fitness(ind) for ind in population]
            next_generation = []


            while len(next_generation) < population_size:
                parent1, parent2 = select_parents(population, fitness_scores)
                offspring1, offspring2 = crossover(parent1, parent2)
                next_generation.append(mutate(offspring1))
                next_generation.append(mutate(offspring2))


            population = next_generation[:population_size]


        # Return the best solution found
        best_individual = max(population, key=fitness)
        return best_individual