from policy import Policy
from abc import abstractmethod
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import  Dense
from tensorflow.keras.optimizers import Adam

class Genetic(Policy):
    def __init__(self, population_size=10, generations=10, mutation_rate=0.1, crossover_rate=0.9):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_individual = None
        self.elite_population = []

    def _place_product_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] = 1

    def initialize_population(self, observation):
        population = []
        list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        
        while len(population) < self.population_size:
            individual = []
            stock_states = [np.copy(stock) for stock in observation["stocks"]]
            
            for prod in list_prods:
                for _ in range(prod["quantity"]):
                    prod_size = prod["size"]
                    placed = False
                    for stock_idx, stock in enumerate(stock_states):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        possible_positions = np.argwhere(
                            np.lib.stride_tricks.sliding_window_view(stock == -1, prod_size).all(axis=(2, 3))
                        )

                        if len(possible_positions) > 0:
                            pos_x, pos_y = possible_positions[np.random.randint(len(possible_positions))]
                            self._place_product_(stock, (pos_x, pos_y), prod_size)
                            individual.append((stock_idx, prod_size, (pos_x, pos_y)))
                            placed = True
                            break
                    if not placed:
                        break
            if individual:
                population.append(individual)
        return population

    def fitness(self, individual, observation):
        filled_area = 0
        empty_area = 0
        used_stocks = set()
        stock_states = [np.copy(stock) for stock in observation["stocks"]]

        for action in individual:
            stock_idx, prod_size, position = action
            stock = stock_states[stock_idx]
            prod_w, prod_h = prod_size

            if self._can_place_(stock, position, prod_size):
                self._place_product_(stock, position, prod_size)
                filled_area += prod_w * prod_h
                used_stocks.add(stock_idx)

        for stock_idx in used_stocks:
            empty_area += np.sum(stock_states[stock_idx] == -1)

        num_used_stocks = len(used_stocks)

        penalty_for_unused_stocks = (num_used_stocks - 1) * 500
        penalty_for_empty_area = empty_area * 0.5
        bonus_for_filled_area = filled_area * 2

        return bonus_for_filled_area - penalty_for_unused_stocks - penalty_for_empty_area

    def select_parents(self, population, observation):
        fitness_values = [self.fitness(ind, observation) for ind in population]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            probabilities = [1 / len(fitness_values)] * len(fitness_values)
        else:
            probabilities = [f / total_fitness for f in fitness_values]
        
        indices = np.arange(len(population))
        selected_indices = np.random.choice(indices, size=2, p=probabilities, replace=False)
        return [population[selected_indices[0]], population[selected_indices[1]]]

    def crossover(self, parent1, parent2, observation):
        crossover_point = np.random.randint(1, len(parent1))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return self.validate_individual(child, observation)

    def mutate(self, individual, observation):
        if np.random.random() < self.mutation_rate:
            index_to_mutate = np.random.randint(0, len(individual))
            stock_idx = np.random.randint(0, len(observation["stocks"]))
            stock = observation["stocks"][stock_idx]
            prod_size = individual[index_to_mutate][1]
            pos_x = np.random.randint(0, max(1, stock.shape[0] - prod_size[0]))
            pos_y = np.random.randint(0, max(1, stock.shape[1] - prod_size[1]))
            individual[index_to_mutate] = (stock_idx, prod_size, (pos_x, pos_y))
        return individual

    def validate_individual(self, individual, observation):
        valid_individual = []
        stock_states = [np.copy(stock) for stock in observation["stocks"]]
        for action in individual:
            stock_idx, prod_size, position = action
            stock = stock_states[stock_idx]
            if self._can_place_(stock, position, prod_size):
                self._place_product_(stock, position, prod_size)
                valid_individual.append(action)
        return valid_individual

    def run(self, observation):
        population = self.initialize_population(observation)

        for _ in range(self.generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population, observation)
                child = self.crossover(parent1, parent2, observation)
                child = self.mutate(child, observation)
                new_population.append(child)

            max_elites = max(1, self.population_size // 5)
            self.elite_population = sorted(population, key=lambda ind: self.fitness(ind, observation), reverse=True)[:max_elites]
            population = new_population + self.elite_population
            population = sorted(population, key=lambda ind: self.fitness(ind, observation), reverse=True)[:self.population_size]

        self.best_individual = population[0]
        return self.best_individual

    def get_action(self, observation, info):
        if (str(info) == "{'filled_ratio': 0.0, 'trim_loss': 1}"):
            self.elite_population = []  # Làm mới elite nếu bắt đầu mới
            self.best_individual = self.run(observation)

        if self.best_individual:
            action = self.best_individual.pop(0)
            return {"stock_idx": action[0], "size": action[1], "position": action[2]}
        else:
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

class Actor(tf.keras.Model):
    def __init__(self, name = "_actor_", dir = "student_submissions\s2310027_2311741_2311784_2311578\_check_point_"):
        super(Actor, self).__init__()
        self.checkpoint_dir = dir
        self.checkpoint_file = f"{self.checkpoint_dir}/{name}"

        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
    @tf.function
    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        return value

class Critic(tf.keras.Model):
    def __init__(self, name = "_critic_", dir = "student_submissions\s2310027_2311741_2311784_2311578\_check_point_"):
        super(Critic, self).__init__()
        self.checkpoint_dir = dir
        self.checkpoint_file = f"{self.checkpoint_dir}/{name}"

        self.fc1 = Dense(1024, activation='relu')
        self.fc2 = Dense(512, activation='relu')
        self.v = Dense(1) 

    @tf.function
    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        value = self.v(value)
        mean_value = tf.reduce_mean(value, axis=(1, 2))
        return mean_value

class ActorCritic(Policy):
    def __init__(self, alpha = 0.0005, alpha_ = 0.0005, gamma = 0.99):

        self.actor = Actor()
        self.critic = Critic()

        self.state = None
        self.alpha = alpha
        self.alpha_ = alpha_
        self.gamma = gamma

        self.softmax = None

        self.actor.compile(optimizer = Adam(learning_rate = self.alpha))
        self.critic.compile(optimizer = Adam(learning_rate = self.alpha_))

        self.load_models()

    def get_action(self, observation, info):

        stocks = observation["stocks"]
        products = observation["products"]
        
        n_actions = len(products)

        if str(info) == "{'filled_ratio': 0.0, 'trim_loss': 1}":
            self.softmax = Dense(n_actions, activation='softmax')

        state = tf.convert_to_tensor([stocks])

        action_matrix = self.actor(state[0])
        action_probabilities = self.softmax(action_matrix)
        flat = tf.reshape(action_probabilities, [-1])
        flat = tf.math.log([flat])
        shape = tf.shape(action_probabilities) # get shape
        position = None

        # While đến khi nào chọn ddược best_place != NoSe, tiếp tục Sampling
        while position is None:

            stock_idx, width, product_idx = self.sampling(flat, shape)

            product = products[product_idx]
            prod_size = product["size"]

            stock = stocks[stock_idx]

            if product["quantity"] <= 0 or np.all(stock[width, :] != -1):
                continue

            size, position = self._find_placement_(stock, prod_size, width)
        
        reward = info["trim_loss"]

        if str(info) != "{'filled_ratio': 0.0, 'trim_loss': 1}":
            # self.update(self.state, state, reward)
            with tf.GradientTape(persistent=True) as tape:
                state_value = tf.squeeze(self.critic(self.state))
                state_value_ = tf.squeeze(self.critic(state))
                td_error = reward + self.gamma * state_value_ - state_value
                
                # Actor loss calculation (probability of action taken)
                value = self.actor(state[0])
                prob = self.softmax(value)

                actor_loss = -tf.math.log(prob + 1e-32) * td_error  # No gradient for td_error
                # Critic loss calculation (TD error squared)
                critic_loss = tf.reduce_mean(tf.square(td_error))

            # Compute gradients
            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            # Update weights
            self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        self.state = state
        return {"stock_idx": stock_idx, "size": size, "position": position}
    
    def sampling(self, flat, shape):
        
        sample_idx = tf.random.categorical(flat, 1)  
        sample_idx = tf.squeeze(sample_idx) 
        sample_idx = sample_idx.numpy()

        shape = shape.numpy()

        stock_idx = sample_idx // (shape[1] * shape[2])
        width = (sample_idx % (shape[1] * shape[2])) // shape[2]
        prod_idx = sample_idx % shape[2]

        return stock_idx, width, prod_idx

    def _find_placement_(self, stock, prod_size, width):
        _, stock_h = self._get_stock_size_(stock)

        prod_w, prod_h = prod_size 

        position = None

        size = None
        height = None
        for y in range(stock_h - 1, -1, -1):
            if stock[width, y] == -1:
                height = y
                break
    
        if height is not None:
            if height - prod_h >= -1 and self._can_place_(stock, (width, height - prod_h + 1), (prod_w, prod_h)):
                position = (width, height - prod_h + 1)
                size = (prod_w, prod_h)
            if position is None or prod_w > prod_h:
                if height - prod_w >= -1 and self._can_place_(stock, (width, height - prod_w + 1), (prod_h, prod_w)):
                    position = (width, height - prod_w + 1)
                    size = (prod_h, prod_w)

        return  size, position
    
    def save_models(self):
        # print("...saving models...")
        try:
            self.actor.save_weights(self.actor.checkpoint_file)
            self.critic.save_weights(self.critic.checkpoint_file)
        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self):
        # print("...loading models...")
        try:
            self.actor.load_weights(self.actor.checkpoint_file)
            self.critic.load_weights(self.critic.checkpoint_file)
        except Exception as e:
            print(f"Error loading models: {e}")
        
class Policy2310027_2311741_2311784_2311578(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = ActorCritic()
            self.id = policy_id
        elif policy_id == 2:
            self.policy = Genetic()
            self.id = policy_id

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
