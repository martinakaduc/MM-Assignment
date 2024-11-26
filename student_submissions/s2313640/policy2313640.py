import numpy as np
from policy import Policy


class Policy2313640(Policy):
    def __init__(self, initial_temp=100, cooling_rate=0.95, max_iter=1000):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter

    def cost_function(self, stocks, products):
        waste = sum(np.sum(stock == -1) for stock in stocks)  
        unmet_demand = sum(prod["quantity"] for prod in products) 
        return waste + unmet_demand * 100  

    def generate_neighbor(self, current_solution, observation):
        new_solution = current_solution.copy()
        random_action = np.random.choice(new_solution)
    
        stock_idx = random_action["stock_idx"]
        stock = observation["stocks"][stock_idx]

        valid = False
        for _ in range(100):  
            new_position = (
                np.random.randint(0, stock.shape[0]),
                np.random.randint(0, stock.shape[1]),
            )
            prod_w, prod_h = random_action["size"]
            if (new_position[0] + prod_w <= stock.shape[0] and 
                new_position[1] + prod_h <= stock.shape[1] and 
                np.all(stock[new_position[0]:new_position[0]+prod_w, 
                        new_position[1]:new_position[1]+prod_h] == -1)):
                valid = True
                random_action["position"] = new_position
                break
        if not valid:
            print("Failed to generate valid neighbor")
        return new_solution
    
    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]

        current_solution = [
            {"stock_idx": 0, "size": (prod["size"][0], prod["size"][1]), "position": (0, 0)}
            for prod in products if prod["quantity"] > 0
        ]
        if not current_solution:
            print("No products to cut!")
            return None
        
        current_cost = self.cost_function(stocks, products)
        best_solution = current_solution
        best_cost = current_cost

        T = self.initial_temp  

        for _ in range(self.max_iter):
            neighbor_solution = self.generate_neighbor(current_solution, observation)
            neighbor_cost = self.cost_function(stocks, products)

            delta_cost = neighbor_cost - current_cost

            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
                current_solution = neighbor_solution
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            T *= self.cooling_rate
            if T < 1e-3: 
                break

        return best_solution[0] if best_solution else None

