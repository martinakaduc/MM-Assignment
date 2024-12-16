from policy import Policy


import random
import numpy as np
from policy import Policy


class GAPolicy(Policy):
    def __init__(self):
        # Parameters for the genetic algorithm
        self.pop_size = 120
        self.max_generations = 20
        self.mutation_chance = 0.01
        self.elite_size = 5

    def generate_random_action(self, observation):
        available_items = [item for item in observation['products'] if item['quantity'] > 0]
        chosen_item = random.choice(available_items)
        item_size = chosen_item['size']
        stock_idx, x_pos, y_pos, rotated = self.find_stock_position(item_size, observation)
        rotated_size = item_size if not rotated else (item_size[1], item_size[0])

        return {'stock_idx': stock_idx, 'size': rotated_size, 'position': (x_pos, y_pos), 'rotated': rotated}

    def find_stock_position(self, item_size, observation):
        while True:
            stock_idx = random.randint(0, len(observation['stocks']) - 1)
            stock = observation['stocks'][stock_idx]
            stock_width, stock_height = self._get_stock_size_(stock)

            for rotated in [False, True]:
                current_size = item_size if not rotated else (item_size[1], item_size[0])

                if stock_width < current_size[0] or stock_height < current_size[1]:
                    continue

                for i in range(stock_width - current_size[0] + 1):
                    for j in range(stock_height - current_size[1] + 1):
                        if self._can_place_(stock, (i, j), current_size):
                            return stock_idx, i, j, rotated

        raise ValueError("No suitable stock position found.")

    def compute_fitness(self, action, observation, info):
        stock = observation["stocks"][action["stock_idx"]]
        product_size = action["size"]
        x_pos, y_pos = action["position"]

        stock_width, stock_height = self._get_stock_size_(stock)
        target_x, target_y = stock_width // 2, stock_height // 2

        dist_to_center = np.hypot(x_pos - target_x, y_pos - target_y)

        utilization = np.prod(product_size) / (stock_width * stock_height)

        occupied_area = np.sum(stock >= 0)

        fitness_score = utilization + dist_to_center + occupied_area ** 2

        if not self._can_place_(stock, (x_pos, y_pos), product_size):
            fitness_score -= 1000

        return fitness_score

    def select_top_population(self, population, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        return [population[i] for i in sorted_indices[:len(sorted_indices) // 2]]

    def perform_crossover(self, selected_parents):
        offspring = []
        while len(offspring) < self.pop_size - self.elite_size:
            parent1, parent2 = random.sample(selected_parents, 2)
            child = {
                'stock_idx': parent1['stock_idx'],
                'size': parent2['size'],
                'position': parent1['position'],
                'rotated': parent1['rotated']
            }
            offspring.append(child)
        return offspring

    def apply_mutation(self, offspring):
        for action in offspring:
            if random.random() < self.mutation_chance:
                action['size'] = self.mutate_size(action['size'])
                action['rotated'] = not action['rotated'] if random.random() < 0.5 else action['rotated']
        return offspring

    def genetic_solver(self, observation, info):
        population = [self.generate_random_action(observation) for _ in range(self.pop_size)]

        for _ in range(self.max_generations):
            fitness = [self.compute_fitness(action, observation, info) for action in population]

            sorted_indices = np.argsort(fitness)[::-1]
            sorted_population = [population[i] for i in sorted_indices]

            elites = sorted_population[:self.elite_size]

            selected_parents = self.select_top_population(
                sorted_population[self.elite_size:], fitness[self.elite_size:])
            offspring = self.perform_crossover(selected_parents)
            offspring = self.apply_mutation(offspring)

            population = elites + offspring[:self.pop_size - self.elite_size]

        best_action = population[np.argmax(fitness)]
        return best_action

    def mutate_size(self, size):
        new_size = list(size)
        new_size[0] += random.randint(-1, 1)
        new_size[1] += random.randint(-1, 1)
        return tuple(new_size)

    def get_action(self, observation, info):
        return self.genetic_solver(observation, info)



class BestFitDecreasingPolicy(Policy):
    def __init__(self):
        self.cut_stocks = dict()

    def check_reset(self, info):
        return info['trim_loss'] == 1

    def reset(self):
        self.cut_stocks = dict()

    def compute_wastage(self, stock, position, prod_size):
        x, y = position
        prod_w, prod_h = prod_size

        stock_after_placed = np.copy(stock)
        stock_after_placed[x: x + prod_w, y: y + prod_h] = 0

        return (stock_after_placed == -1).sum() / (stock_after_placed != -2).sum()

    def can_place_with_check(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        if (pos_x + prod_w > stock.shape[0]) or (pos_y + prod_h > stock.shape[1]):
            return False

        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)

    def compute_wastage_all_stocks(self, stocks, stock_indices, product_size):
        dict_wastage = dict()
        prod_size = tuple(product_size)

        # Loop through all stocks and compute the wastage for each possible position on each stock
        for stock_idx, stock in zip(stock_indices, stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            can_place = False
            max_w = max(stock_w - prod_w + 1, stock_w - prod_h + 1)
            max_h = max(stock_h - prod_h + 1, stock_h - prod_w + 1)
            for x in range(max_w):
                for y in range(max_h):
                    if self.can_place_with_check(stock, (x, y), prod_size):
                        can_place = True
                        dict_wastage[(x, y, stock_idx, prod_size)] = self.compute_wastage(stock, (x, y), prod_size)
                        break
                    elif self.can_place_with_check(stock, (x, y), prod_size[::-1]):
                        can_place = True
                        dict_wastage[(x, y, stock_idx, prod_size[::-1])
                                     ] = self.compute_wastage(stock, (x, y), prod_size[::-1])
                        break
                if can_place:
                    break
        return dict_wastage

    def get_action(self, observation, info):
        if self.check_reset(info):
            self.reset()

        # Get the product with the largest area
        largest_prod = max(
            (product for product in observation['products'] if product['quantity'] > 0),
            key=lambda x: x['size'][0] * x['size'][1],
        )
        prod_size = largest_prod['size']
        # Only consider cut stocks if there are any, otherwise consider all stocks
        if self.cut_stocks:
            stock_indices = sorted(self.cut_stocks, key=self.cut_stocks.get, reverse=False)
            stocks = [observation['stocks'][i] for i in stock_indices]
        else:
            stocks = observation['stocks']
            stock_indices = list(range(len(stocks)))
        dict_wastage = self.compute_wastage_all_stocks(stocks, stock_indices, prod_size)

        # If there is no possible position, find a new stock to place the product
        if not dict_wastage:
            stocks = observation['stocks']
            stock_indices = list(range(len(stocks)))
            dict_wastage = self.compute_wastage_all_stocks(stocks, stock_indices, prod_size)

        # Get the position with the minimum wastage
        min_wastage = min(dict_wastage, key=dict_wastage.get)
        pos_x, pos_y, stock_idx, _prod_size = min_wastage
        prod_size[0], prod_size[1] = _prod_size

        # Update the cut stocks with leftover space
        self.cut_stocks[stock_idx] = dict_wastage[min_wastage]
        result = {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        return result



class Policy2252853_2252022_2252818_2252154_2252215(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = BestFitDecreasingPolicy()
        elif policy_id == 2:
            self.policy = GAPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

