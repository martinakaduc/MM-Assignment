from policy import Policy
import numpy as np
import random
from scipy.optimize import linprog

class Policy2310097_2312963_2310577_2213195_2310052(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.Policy = GA()
        elif policy_id == 2:
            self.Policy = SARSA()

    def get_action(self, observation, info):
        # Student code here
        return self.Policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed
class GA(Policy):
    def __init__(self):
        """
        Initialize the policy for the cutting stock problem
        """
        # Default genetic algorithm hyperparameters
        self.population_size = 100
        self.max_generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # Default policy-specific configuration
        self.policy_id = 1
        if self.policy_id == 1:
            self.population_size = 150
            self.mutation_rate = 0.15
        elif self.policy_id == 2:
            self.population_size = 200
            self.mutation_rate = 0.05

    def get_action(self, observation, info):
        """
        Determine the best action for cutting stock placement

        Args:
            observation (dict): Current environment state
            info (dict): Additional information

        Returns:
            dict: Placement action with stock_idx, size, and position
        """
        list_prods = observation["products"]
        if not list_prods:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Find the first available product
        current_product = None
        for prod in list_prods:
            if prod["quantity"] > 0:
                current_product = prod
                break

        if not current_product:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Use genetic algorithm to find best placement
        return self._genetic_placement(
            observation["stocks"],
            current_product
        )

    def _genetic_placement(self, stocks, product):
        prod_size = product.get("size", [0, 0])

        def calculate_fitness(placement):
            stock_idx, pos_x, pos_y, rotated = placement

            if stock_idx == -1 or stock_idx >= len(stocks):
                return -np.inf

            stock = stocks[stock_idx]
            test_size = prod_size[::-1] if rotated else prod_size

            if not self._is_valid_placement(stock, (pos_x, pos_y), test_size):
                return -np.inf

            stock_w, stock_h = self._get_stock_size_(stock)
            edge_proximity = min(pos_x, stock_w - (pos_x + test_size[0])) + \
                             min(pos_y, stock_h - (pos_y + test_size[1]))

            return edge_proximity

        population = self._initialize_population(stocks, prod_size)

        for _ in range(self.max_generations):
            fitness_scores = [calculate_fitness(placement) for placement in population]
            selected = self._tournament_selection(population, fitness_scores)
            offspring = self._crossover(selected)
            offspring = self._mutate(offspring, stocks, prod_size)
            population = offspring

        best_placement_idx = np.argmax([calculate_fitness(p) for p in population])
        best_placement = population[best_placement_idx]

        return {
            "stock_idx": best_placement[0],
            "size": prod_size[::-1] if best_placement[3] else prod_size,
            "position": (best_placement[1], best_placement[2])
        }

    def _initialize_population(self, stocks, prod_size):
        population = []
        for _ in range(self.population_size):
            stock_idx = random.randint(0, len(stocks) - 1)
            stock = stocks[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            prod_w, prod_h = prod_size
            rotated = random.choice([True, False])

            if rotated:
                pos_x = random.randint(0, max(1, stock_w - prod_h))
                pos_y = random.randint(0, max(1, stock_h - prod_w))
            else:
                pos_x = random.randint(0, max(1, stock_w - prod_w))
                pos_y = random.randint(0, max(1, stock_h - prod_h))

            population.append((stock_idx, pos_x, pos_y, rotated))

        return population

    def _tournament_selection(self, population, fitness_scores, tournament_size=5):
        selected = []
        for _ in range(len(population)):
            actual_tournament_size = min(tournament_size, len(population))
            tournament_indices = random.sample(range(len(population)), actual_tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])

        return selected

    def _crossover(self, population):
        offspring = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]

            if random.random() < self.crossover_rate:
                child1 = (parent2[0], parent1[1], parent2[2], parent1[3])
                child2 = (parent1[0], parent2[1], parent1[2], parent2[3])
            else:
                child1, child2 = parent1, parent2

            offspring.extend([child1, child2])

        return offspring

    def _mutate(self, population, stocks, prod_size):
        mutated_population = []
        for placement in population:
            if random.random() < self.mutation_rate:
                stock_idx = random.randint(0, len(stocks) - 1)
                stock = stocks[stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)

                prod_w, prod_h = prod_size
                rotated = random.choice([True, False])

                if rotated:
                    pos_x = random.randint(0, max(1, stock_w - prod_h))
                    pos_y = random.randint(0, max(1, stock_h - prod_w))
                else:
                    pos_x = random.randint(0, max(1, stock_w - prod_w))
                    pos_y = random.randint(0, max(1, stock_h - prod_h))

                mutated_placement = (stock_idx, pos_x, pos_y, rotated)
                mutated_population.append(mutated_placement)
            else:
                mutated_population.append(placement)

        return mutated_population

    def _is_valid_placement(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        stock_w, stock_h = self._get_stock_size_(stock)
        if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
            return False

        return self._can_place_(stock, position, prod_size)



class SARSA(Policy):
    def __init__(self, num_actions=10, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__()
        self.q_table = {}  # Dictionary to store Q-values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.num_actions = num_actions  # Number of possible actions

    def _get_state_key(self, observation, info):
        return str(observation)

    def get_action(self, observation, info):
        state = self._get_state_key(observation, info)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        if random.uniform(0, 1) < self.epsilon:
            action = self._random_action(observation)
        else:
            action = self._best_action(observation, state)

        return action

    def _random_action(self, observation):
        stock_indices = list(range(len(observation["stocks"])))
        product_indices = list(range(len(observation["products"])))

        random.shuffle(stock_indices)
        random.shuffle(product_indices)

        for stock_idx in stock_indices:
            stock = observation["stocks"][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            for product_idx in product_indices:
                product = observation["products"][product_idx]

                if product["quantity"] > 0:
                    prod_size = product["size"]

                    for _ in range(10):  # Limit retries to reduce lag
                        pos_x = random.randint(0, stock_w - 1)
                        pos_y = random.randint(0, stock_h - 1)

                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (pos_x, pos_y),
                            }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _best_action(self, observation, state):
        action_space = []  # To map all possible actions to indices
        for stock_idx, stock in enumerate(observation["stocks"]):
            for product_idx, product in enumerate(observation["products"]):
                if product["quantity"] > 0:
                    prod_size = product["size"]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    for x in range(stock_w - prod_size[0] + 1):
                        for y in range(stock_h - prod_size[1] + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                action_space.append({
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y),
                                })
                                if len(action_space) >= self.num_actions:
                                    break
                    if len(action_space) >= self.num_actions:
                        break
            if len(action_space) >= self.num_actions:
                break

        if action_space:
            action_idx = np.argmax(self.q_table[state][:len(action_space)])
            return action_space[action_idx]

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def update_q_table(self, state, action, next_state, next_action, reward):
        """Update Q-table using SARSA update rule."""
        action_idx = self._action_to_index(action)
        next_action_idx = self._action_to_index(next_action)

        if action_idx >= self.num_actions:
            action_idx = self.num_actions - 1  # Clamp to max index

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        current_q = self.q_table[state][action_idx]
        next_q = self.q_table[next_state][next_action_idx]

        # SARSA update rule
        self.q_table[state][action_idx] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def step(self, observation, action, reward, next_observation, info):
        """Perform one step of the SARSA learning algorithm."""
        state = self._get_state_key(observation, info)
        next_state = self._get_state_key(next_observation, info)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        next_action = self.get_action(next_observation, info)

        # Update Q-table using SARSA rule
        self.update_q_table(state, action, next_state, next_action, reward)

        return next_action

    def _action_to_index(self, action):
        """
        Map action to a valid index within the range of the Q-table.
        Ensures the index is within the valid range [0, num_actions - 1].
        """
        stock_idx = action.get("stock_idx", -1)
        if stock_idx < 0 or stock_idx >= self.num_actions:
            # Handle invalid stock index gracefully
            return 0  # Default to a valid index
        return stock_idx % self.num_actions

    def _compute_reward(self, observation, action, info):
        """Compute reward based on the current observation, action, and trim loss."""
        if action["stock_idx"] == -1:
            # Negative reward for invalid actions
            return -10

        product_size = action["size"]
        position = action["position"]

        # Calculate the trim loss if available in `info`
        trim_loss = info.get("trim_loss", 0)  # Assume trim_loss is part of info
        max_trim_loss_penalty = 50  # Define the maximum penalty for trim loss
        trim_loss_penalty = max_trim_loss_penalty * (trim_loss / max(trim_loss, 1))

        # Positive reward for successfully placing a product
        for stock in observation["stocks"]:
            if self._can_place_(stock, position, product_size):
                # High reward for valid placements, reduced by trim loss
                return max(100 - trim_loss_penalty, 0)

        # Small penalty for valid but suboptimal actions, increased by trim loss
        return -1 - trim_loss_penalty