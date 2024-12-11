from policy import Policy
import numpy as np
import random
import math

class SimulatedAnnealing1(Policy):
    def __init__(self, initial_temperature=1000, cooling_rate=0.95, iterations=100):
        super().__init__()
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations = iterations

    def get_action(self, observation, info):
        ffd_policy = FFD()
        initial_action = ffd_policy.get_action(observation, info)

        if initial_action is None:
            return self.get_default_action()

        optimized_action = self.run_simulated_annealing(initial_action, observation)

        return optimized_action

    def run_simulated_annealing(self, initial_action, observation):
        current_action = initial_action.copy()
        current_cost = self.calculate_cost(current_action, observation)
        best_action = current_action.copy()
        best_cost = current_cost
        temperature = self.initial_temperature

        for _ in range(self.iterations):
            neighbor_action = self.generate_neighbor(current_action, observation)

            if neighbor_action is None:
                continue  

            neighbor_cost = self.calculate_cost(neighbor_action, observation)
            delta_cost = neighbor_cost - current_cost

            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_action = neighbor_action.copy()
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_action = neighbor_action.copy()
                    best_cost = current_cost

            temperature *= self.cooling_rate

            if temperature < 1e-3:
                break

        return best_action

    def generate_neighbor(self, action, observation):
        neighbor = action.copy()

        move_or_rotate = random.choice(['move', 'rotate'])

        if move_or_rotate == 'move':
            new_x = max(0, action['position'][0] + random.randint(-1, 1))
            new_y = max(0, action['position'][1] + random.randint(-1, 1))
            neighbor['position'] = (new_x, new_y)
        elif move_or_rotate == 'rotate':
            neighbor['size'] = (action['size'][1], action['size'][0])

        if self.is_valid_action(neighbor, observation):
            return neighbor
        else:
            return None  # Không hợp lệ, bỏ qua

    def is_valid_action(self, action, observation):
        stock = observation["stocks"][action["stock_idx"]]
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = action["position"]
        prod_w, prod_h = action["size"]

        # Kiểm tra giới hạn của stock
        if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
            return False
        return self._can_place_(stock, (pos_x, pos_y), action["size"])

    def calculate_cost(self, action, observation):
        stock = observation["stocks"][action["stock_idx"]].copy() 
        pos_x, pos_y = action["position"]
        prod_w, prod_h = action["size"]

        if pos_x + prod_w > stock.shape[1] or pos_y + prod_h > stock.shape[0]:
            return float('inf') 

        if not self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
            return float('inf')  

        stock[pos_y: pos_y + prod_h, pos_x: pos_x + prod_w] = 1  
        remaining_empty_area = np.sum(stock == -1)

        return remaining_empty_area

    def get_default_action(self):
        return {"stock_idx": 0, "size": [1, 1], "position": (0, 0)}

class FFD(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        # Sắp xếp sản phẩm theo diện tích giảm dần
        list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                placed = False
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    stock_idx = i
                                    placed = True
                                    break
                            if placed:
                                break

                    if not placed and stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):  
                                    prod_size = prod_size[::-1]  
                                    pos_x, pos_y = x, y
                                    stock_idx = i
                                    placed = True
                                    break
                            if placed:
                                break

                    if placed:
                        break
                if placed:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}