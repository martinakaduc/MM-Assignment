from policy import Policy
from scipy.optimize import linprog

import numpy as np

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            self.columns = []
            pass
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.Knapsack_Alg(observation, info)
        elif self.policy_id == 2:
            return self.Column_Generation(observation, info)


    def Column_Generation(self, observation, info):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        # Student code here
        self.initial_columns(observation)
        while True:
            dual_values = self.solve_MP(observation)
            new_column = self.generate_new_col(observation, dual_values)
            if new_column is None:
                break
            self.columns.append(new_column)

            for i, stock in enumerate(list_stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                for prod in list_prods:
                    prod_size = prod["size"]
                    if prod["quantity"] > 0:
                        for orientation in range(2):
                            if orientation == 1:
                                prod_size = prod_size[::-1]
                            prod_w, prod_h = prod_size

                            for x in range(stock_w - stock_h + 1):
                                for y in range(prod_w - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        stock_idx = i
                                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def initial_columns(self, observation):
        list_prods = observation["products"]
        for idx, prod in enumerate(list_prods):
            if prod["quantity"] > 0:
                column = np.zeros(len(list_prods) * 2)
                prod_idx = idx * 2 
                column[prod_idx] = prod["quantity"]
                column[prod_idx + 1] = prod["quantity"]
                self.columns.append(column)


    def solve_MP(self, observation):
        list_prods = observation["products"]
        c = np.ones(len(self.columns))
        A_eq = np.zeros((len(list_prods)*2, len(self.columns)))
        b_eq = np.zeros(len(list_prods)*2)

        for i, prod in enumerate(list_prods):
            b_eq[i * 2] = prod["quantity"]
            b_eq[i * 2 + 1] = prod["quantity"]
            for j, column in enumerate(self.columns):
                A_eq[i * 2, j] = column[i * 2]
                A_eq[i * 2 + 1, j] = column[i * 2 + 1]

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds = (0, None), method='highs')
        return result.con
    
    def generate_new_col(self, observation, dual_values):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]
        for stock_idx, stock in enumerate(list_stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            column = np.zeros(len(list_prods)*2)
            for prod_idx, prod in enumerate(list_prods):
                prod_size = prod["size"]
                prod_quantity = prod["quantity"]
                if prod_quantity > 0:
                    for orientation in range(2):
                        if orientation == 1:
                            prod_size = prod_size[::-1]
                        prod_w, prod_h = prod_size

                        if stock_w >= prod_w and stock_h >= prod_h:
                            column[prod_idx * 2 + orientation] += 1
                            if self._can_place_(stock, (0, 0), prod_size):
                                stock[0:prod_w,0:prod_h] == prod_idx
                                reduced_cost = np.dot(column, dual_values) - 1
                                if reduced_cost < 0:
                                    return column
                        
        return None

    #Knapsack
    def Knapsack_Alg(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for i, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            max_area = stock_w * stock_h
            weights = []
            values = []
            indices = []

            for j, prod in enumerate(list_prods):
                if prod["quantity"] > 0:
                    prod_w, prod_h = prod["size"]
                    area = prod_w * prod_h
                    if area <= max_area:
                        weights.append(area)
                        values.append(prod["quantity"])
                        indices.append(j)

            if not weights:
                continue

            result = linprog(
                c=[-v for v in values],
                A_ub=[weights],
                b_ub=[max_area],
                bounds=[(0, 1) for _ in values],
                method="highs"
            )

            if result.success:
                selected_idx = indices[int(result.x.argmax())]
                prod = list_prods[selected_idx]
                prod_size = prod["size"]
                
                for x in range(stock_w - prod_size[0] + 1):
                    for y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            stock_idx = i
                            pos_x, pos_y = x, y
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                            prod_size = prod_size[::-1]  
                            stock_idx = i
                            pos_x, pos_y = x, y
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    # Student code here
    # You can add more functions if needed
