from policy import Policy
import numpy as np
import scipy
from scipy.signal import correlate
from scipy.optimize import milp
from scipy.optimize import LinearConstraint, Bounds

class Policy2310013_2310038_2310016_2310641(Policy):
    # Initialize
    def __init__(self, policy_id = 1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.checksum = -1
            self.strategy = 1
        elif policy_id == 2:
            self.strategy = 2
            self.solutions = [] # solutions: list of decisions
            self.n_solutions = 0 # number of solutions
            self.current_solution = 0 # current index

    def get_action(self,observation,info):
        if (self.strategy == 1):
            return self.BSBP(observation,info)
        elif (self.strategu == 2):
            return self.MILP(observation,info)
    ###############################################################
    # HELPER FUNCTION 1
    ###############################################################    
            
    def check_sum(self, stocks):
        return np.sum(stocks == -1)

    def find_corner_spots(self, stock, prod_size):
        r_rows, r_cols = prod_size
        rows = np.argmin(stock[:,0])
        collumns = np.argmin(stock[0])
        if stock[rows,0] != -2: rows = stock[:,0].size # get stock's length
        if stock[0,collumns] != -2: collumns = stock[0].size # get stock's width
        
        stocker = np.pad(1 - (stock[:rows,:collumns] == -1), 1, mode="constant", constant_values=1)
        
        rect_kernel = np.ones((r_rows, r_cols), dtype=int) # 
        up_kernel = np.ones((1, r_cols), dtype=int)
        left_kernel = np.ones((r_rows, 1), dtype=int)

        # cross correlation and indexing
        result1 = correlate(stocker, rect_kernel)[r_rows:-1,r_cols:-1] == 0 # all placements on the stock that fits the product
        result2 = correlate(stocker, up_kernel)[0:-2,r_cols:-1] >= 1 # all placements that block the product 1 cell upward
        result3 = correlate(stocker, left_kernel)[r_rows:-1,0:-2] >= 1 # all placements that block the product 1 cell leftward
        
        result = result1 & result2 & result3 # combining all the criterions, we get all valid upper-left corner placements
        corners = np.argwhere(result)
        return corners

    def sort_products(self, list_prods):
        areas = np.array([prod["size"][0] * prod["size"][1] for prod in list_prods])
        return np.argsort(areas)[::-1]

    def sort_stocks_1(self, stocks):
        area = np.sum(stocks != -2, axis=(1,2))
        return np.argsort(area)[::-1]
    
    def sort_stocks_2(self, stocks):
        area = np.sum(stocks != -2, axis=(1,2))
        return np.argsort(area)[::1]
    
    def make_decision(self, stocks, list_prods):
        for prod_idx in self.sorted_prod_indices:
            prod = list_prods[prod_idx]
            if prod["quantity"] > 0:
                pw, ph = np.array(prod["size"])
                # rotate the product, prioritize vertical placements (for uniform placements)
                if pw > ph:
                    pw, ph = ph, pw
                # prioritize fitting products in bigger stocks if the amount of products to place is large
                # else do the otherwise to save area, avoid unnecessary choosing bigger stocks
                if self.small_strategy:
                    stocks_indices_list = self.sorted_stocks_indices_ascending
                else:
                    stocks_indices_list = self.sorted_stocks_indices_descending
                for s_idx in stocks_indices_list:
                    stock = stocks[s_idx]
                    # find upperleft corner
                    ul_corners = self.find_corner_spots(stock, [pw, ph])
                    rotated = False
                    # if there is no way to place this product then rotate it
                    if ul_corners.size == 0:
                        ul_corners = self.find_corner_spots(stock, [ph, pw])
                        rotated = True
                    if ul_corners.size > 0:
                        x, y = ul_corners[0]
                        self.checksum -= ph * pw
                        return {
                            "stock_idx": s_idx,
                            "size": (ph, pw) if rotated else (pw, ph),
                            "position": (x, y),
                        }
        # cant find any placement, might have encountered hashing error -> resetting hash
        self.checksum = -1
        return {
            "stock_idx": 0,
            "size": np.array([0, 0]),
            "position": (0, 0),
        }
    
    ###############################################################
    # POLICY 1
    ###############################################################

    def BSBP(self, observation, info):
            stocks = np.stack(observation["stocks"])
            checksummer = self.check_sum(stocks) # observing if the environment has changed to correctly manage cache
            list_prods = observation["products"]
            if checksummer != self.checksum:
                area = sum(prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in list_prods)
                avg_area = checksummer / stocks.shape[0]
                if area < avg_area:
                    self.small_strategy = True
                    self.sorted_stocks_indices_ascending = self.sort_stocks_2(stocks)
                else:
                    self.small_strategy = False
                    self.sorted_stocks_indices_descending = self.sort_stocks_1(stocks)
                # sort products indices in ascending, prioritize placing bigger products first,
                # and save smaller products to fill in the gaps after that
                self.sorted_prod_indices = self.sort_products(list_prods)
                self.checksum = checksummer
            return self.make_decision(stocks, list_prods)

    ###############################################################
    # HELPER FUNCTION 2
    ###############################################################  

    def get_stock_size(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h
    
    def create_variable_mapping(self):
        X_mapping = []
        
        # get all stocks and products sizes
        stock_dims = np.array([(s["length"], s["width"]) for s in self.stocks])
        product_dims = np.array([(p["length"], p["width"]) for p in self.products])
        
        # create meshgrid for all combinations
        S, P = np.meshgrid(np.arange(len(self.stocks)), np.arange(len(self.products)))
        S = S.flatten()
        P = P.flatten()
        
        for s_idx, p_idx in zip(S, P):
            stock_len, stock_width = stock_dims[s_idx]
            prod_len, prod_width = product_dims[p_idx]
            
            # checking if the current product fits inside the current stock
            prod_dims = np.array([[prod_len, prod_width], 
                                [prod_width, prod_len]])
            fits = (prod_dims <= np.array([stock_len, stock_width])).all(axis=1)
            
            for r, fit in enumerate(fits):
                # if it does, calculate the meshgrid for all position combinations and put it
                # in the searching space
                if fit:
                    if r == 0:
                        p_length, p_width = prod_len, prod_width
                    else:
                        p_length, p_width = prod_width, prod_len
                        
                    max_i = stock_len - p_length + 1
                    max_j = stock_width - p_width + 1
                    I, J = np.meshgrid(np.arange(max_i), np.arange(max_j))
                    positions = np.column_stack((I.flatten(), J.flatten()))
                    
                    X_mapping.extend([
                        (s_idx, p_idx, i, j, r) 
                        for i, j in positions
                    ])
        
        return X_mapping

    def create_constraint_matrices(self, X_mapping, n_x_vars, n_y_vars):
        n_vars = n_x_vars + n_y_vars
        constraints = []
        
        ### Constraint 1. Demand fulfillment
        product_ids = np.array([x[1] for x in X_mapping])
        for p_id in range(len(self.products)):
            A_row = np.zeros(n_vars)
            A_row[:n_x_vars] = (product_ids == p_id).astype(float)
            constraints.append(LinearConstraint(A_row, self.products[p_id]["demand"], self.products[p_id]["demand"]))
        
        ### Constraint 2. Stock usage
        stock_ids = np.array([x[0] for x in X_mapping])
        for s_id in range(len(self.stocks)):
            mask = stock_ids == s_id
            if np.any(mask):
                for idx in np.where(mask)[0]:
                    A_row = np.zeros(n_vars)
                    A_row[idx] = 1
                    A_row[n_x_vars + s_id] = -1
                    constraints.append(LinearConstraint(A_row, -1, 0))
        
        ### Constraint 3. No overlapping
        X_data = np.array(X_mapping, dtype=[
            ('s_id', int), ('p_id', int), ('i', int), ('j', int), ('r', int)
        ])
        
        for s in self.stocks:
            s_id = s["id"]
            L_s = s["length"]
            W_s = s["width"]
            
            # create points meshgridd
            U, V = np.meshgrid(np.arange(L_s), np.arange(W_s))
            points = np.column_stack((U.flatten(), V.flatten()))

            mask = X_data['s_id'] == s_id
            masker = np.where(mask)[0]
            
            for u, v in points:
                A_row = np.zeros(n_vars)
                
                for idx in masker:
                    istock, iproduct, i, j, rotated = X_mapping[idx]
                    p = self.products[iproduct]

                    # flip dimensions if rotated
                    if rotated == 0:
                        p_length, p_width = p["length"], p["width"]
                    else:
                        p_length, p_width = p["width"], p["length"]
                    
                    # if the product covers (u, v)
                    if (i <= u < i + p_length and 
                        j <= v < j + p_width):
                        A_row[idx] = 1
                        
                constraints.append(LinearConstraint(A_row, 0, 1))
        
        return constraints

    def solve_2d_cutting_stock(self):
        # create variable mapping
        X_mapping = self.create_variable_mapping()
        n_x_vars = len(X_mapping)
        n_y_vars = len(self.stocks)
        n_vars = n_x_vars + n_y_vars

        # objective function coefficients
        c = np.zeros(n_vars)
        c[n_x_vars:] = self.stock_data['length'] * self.stock_data['width']

        constraints = self.create_constraint_matrices(X_mapping, n_x_vars, n_y_vars)

        bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
        integrality = np.ones(n_vars, dtype=bool)

        # solve and process results
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if result.success:
            x = result.x

            # return placements
            placements = np.where(x[:n_x_vars] > 0.5)[0]
            for idx in placements:
                s_id, p_id, i, j, r = X_mapping[idx]
                self.solutions.append({
                    "stock_id": s_id,
                    "product_id": p_id,
                    "position": (i, j),
                    "rotated": r,
                })
            self.n_solutions = len(self.solutions)

    ###############################################################
    # POLICY 2
    ###############################################################      

    def MILP(self, observation, info):
        if self.n_solutions == self.current_solution:
            self.current_solution = 0
            self.solutions = []
            self.stocks = []
            self.products = []

            for id, stock in enumerate(observation["stocks"]):
                s_w, s_h = self.get_stock_size(stock)
                self.stocks.append({"id": id, "length":s_h, "width": s_w})

            for id, product in enumerate(observation["products"]):
                p_w, p_h = product["size"]
                demand = product["quantity"]
                self.products.append({"id": id, "length": p_h, "width": p_w, "demand": demand})

            self.stock_data = np.array([(s["id"], s["length"], s["width"]) for s in self.stocks], 
                                dtype=[('id', int), ('length', int), ('width', int)])
            self.product_data = np.array([(p["id"], p["length"], p["width"], p["demand"]) for p in self.products],
                                dtype=[('id', int), ('length', int), ('width', int), ('demand', int)])

            self.solve_2d_cutting_stock()

        if self.n_solutions:
            solution = self.solutions[self.current_solution]
            stock_idx = solution["stock_id"]
            product_id = solution["product_id"]
            j, i = solution["position"]
            rotated = solution["rotated"]
            self.current_solution += 1
            x, y = observation["products"][product_id]["size"]
            if rotated:
                x, y = y, x
            return {"stock_idx": stock_idx, "size": (x, y), "position": (i,j)}