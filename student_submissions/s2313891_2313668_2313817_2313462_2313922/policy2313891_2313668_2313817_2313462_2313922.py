from policy import Policy
import numpy as np
from scipy.optimize import linprog


class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        super().__init__()
        self.actions = []
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here
        self.algorithym = None
        if policy_id == 1:
            self.algorithym = 1
            pass
        elif policy_id == 2:
            self.algorithym = 2
            pass

    def get_action(self, observation, info):
        if self.algorithym == 1:
            if not self.actions:
                self.ILP(observation)
            return self.actions.pop(0)
        
        elif self.algorithym == 2:
            return self.Best_fit(observation,info)

    # Student code here
    def Best_fit(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]
        products = sorted(products, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
        
        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                placed = False
                best_fit_stock = None
                best_fit_position = None
                best_fit_size = None
                best_fit_left_space = float('inf')

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Kiểm tra lần 1: sản phẩm theo chiều dài rộng gốc
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for i in range(stock_w - prod_w + 1):
                            for j in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (i, j), prod_size):
                                    # Tính không gian còn lại sau khi đặt sản phẩm
                                    left_space = (stock_w * stock_h) - (prod_w * prod_h)
                                    if left_space < best_fit_left_space:
                                        best_fit_left_space = left_space
                                        best_fit_stock = stock_idx
                                        best_fit_position = (i, j)
                                        best_fit_size = prod_size
                                    placed = True
                    
                    # Kiểm tra lần 2: sản phẩm xoay 90 độ
                    if not placed and stock_w >= prod_h and stock_h >= prod_w:
                        for i in range(stock_w - prod_h + 1):
                            for j in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (i, j), prod_size[::-1]):
                                    left_space = (stock_w * stock_h) - (prod_h * prod_w)
                                    if left_space < best_fit_left_space:
                                        best_fit_left_space = left_space
                                        best_fit_stock = stock_idx
                                        best_fit_position = (i, j)
                                        best_fit_size = prod_size[::-1]
                                    placed = True

                    if placed:
                        break

                if best_fit_stock is not None:
                    return {"stock_idx": best_fit_stock, "size": best_fit_size, "position": best_fit_position}

        return None

    
    def ILP(self, observation):
        list_prods = np.copy(observation["products"])  # List of products with size and profits
        stocks = np.copy(observation["stocks"])        # List of available stock sheets

        list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        
        stock_dims = [self._get_stock_size_(stock) for stock in stocks]
        
        # Number of products and stocks
        n_products = len(list_prods)
        n_stocks = len(stocks)

        # Build objective: minimize unused stock area (trim loss)
        c = []  # Objective function coefficients (minimize trim loss)
        for j in range(n_stocks):
            stock_w, stock_h = stock_dims[j]
            c += [0] * n_products  # Variables for each product in this stock
            # c += [stock_w * stock_h] * n_products

        # Constraint 1: Each product must be placed exactly once
        A_ub = []  # Inequality constraints matrix
        b_ub = []  # Inequality constraints bounds
        for i, prod in enumerate(list_prods):
            row = [0] * (n_products * n_stocks)
            for j in range(n_stocks):
                row[i * n_stocks + j] = 1
            A_ub.append(row)
            b_ub.append(1)  # Each product must be placed exactly once

        # Constraint 2: Stock capacity constraints
        for j, stock in enumerate(stocks):
            stock_w, stock_h = stock_dims[j]
            row = [0] * (n_products * n_stocks)
            for i, prod in enumerate(list_prods):
                prod_w, prod_h = prod["size"]
                if self._can_place_(stocks[j], (0, 0), (prod_w, prod_h)) or self._can_place_(stocks[j], (0, 0), (prod_h, prod_w)):
                    row[i * n_stocks + j] = max(prod_w * prod_h, prod_h * prod_w)
            A_ub.append(row)
            b_ub.append(stock_w * stock_h)

        # Solve ILP using linprog
        result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), method='simplex')

        idx_stock = -1
        prev_allo = 0
        allocation = result.x
        for i in range(n_products):
            for j in range(n_stocks):
                idx = i * n_stocks + j
                if allocation[idx] > 0.5:  # Assigned to stock j
                    if idx_stock == -1 or prev_allo < allocation[idx]:
                        idx_stock = j
                        prev_allo = allocation[idx]

        stock = stocks[idx_stock]
        stock_w, stock_h = stock_dims[idx_stock]
        for i in range(n_products):
            prod_size = list_prods[i]["size"]
            prod_q = list_prods[i]["quantity"]
            # Find position to place the product
            while prod_q > 0:
                placed = False
                for x in range(stock_w):
                    for y in range(stock_h):      
                        if self._can_place_(stock, (x, y), prod_size):
                            self.actions.append({
                                "stock_idx": idx_stock,
                                "size": prod_size,
                                "position": (x, y)
                            })
                            placed = True
                            stock[x:x + prod_size[0], y:y + prod_size[1]] = 1
                            prod_q -= 1
                            break
                        elif self._can_place_(stock, (x, y), prod_size[::-1]):
                            self.actions.append({
                                "stock_idx": idx_stock,
                                "size": prod_size[::-1],
                                "position": (x, y)
                            })
                            placed = True
                            stock[x:x + prod_size[1], y:y + prod_size[0]] = 1
                            prod_q -= 1
                            break
                    if placed:
                        break
                if not placed:
                    break
        return self.actions
        
    def get_action_1(self, observation,info):
        if not self.actions:
            self.ILP(observation)
            
        return self.actions.pop(0)
    # You can add more functions if needed
