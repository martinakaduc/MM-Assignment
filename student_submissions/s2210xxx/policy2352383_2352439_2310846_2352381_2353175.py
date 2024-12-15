from policy import Policy
from scipy.optimize import linprog
import numpy as np

class Policy2352383_2352439_2310846_2352381_2353175(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Floor-Ceiling Algorithm
        self.bin_width = None
        self.bin_height = None

        self.items = []  # Items to be packed
        self.last_prod = None
        self.checkLastProd = 0
        
        self.shelves = []  # Containing shelves
        self.floors = {}
        self.last_shelves = None

        # Column Generation Policy
        self.col_gen_policy = ColumnGenerationPolicy()

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.get_action_fcrg(observation, info)
        if self.policy_id == 2:
            return self.col_gen_policy.get_action_colGen(observation, info)

    def _initialize_items(self, observation):
        # Sort items by their shortest edge for better packing.
        self.items = [
            (prod["size"], prod["quantity"]) for prod in observation["products"]
        ]
        self.items.sort(key=lambda x: min(x[0]), reverse=True)

    def _is_feasible(self, shelf, item):
        # Check if the item can fit on the shelf.
        remaining_floor_space = self.bin_width - sum(w for w, _ in shelf["items"])

        return (
            (remaining_floor_space >= item[0]) 
        )

    def _pack_item(self, shelf, item):
        # Pack an item into the best position 
        item = tuple(sorted(item, reverse=True))  # Align longest edge horizontally

        remaining_floor_space = self.bin_width - sum(w for w, _ in shelf["items"])

        if remaining_floor_space >= item[0]:
            shelf["items"].append(item)

    def _create_shelf(self, item):
        # Initialize a new shelf and place the item.
        shelf = {"items": []}
        self._pack_item(shelf, item)
        self.shelves.append(shelf)

    def _create_new_floor_(self, stock_idx, floor_height):
        if stock_idx not in self.floors:
            self.floors[stock_idx] = []
        self.floors[stock_idx].append(floor_height)

    def _pack_fcrg(self):
        # Implement the FCRG heuristic for packing.
        
        self.shelves = []  # Reset shelves for the current stock
        for item, quantity in self.items:
            for _ in range(quantity):
                placed = False
                first = True
                for shelf in self.shelves:
                    # Skip if the first item in the shelf matches the current item
                    if len(shelf["items"]) > 0:
                        if shelf["items"][0][0] == item[0] and shelf["items"][0][1] == item[1]:
                            continue
                    
                    # Check feasibility and pack the item if possible
                    if self._is_feasible(shelf, item):
                        self._pack_item(shelf, item)
                        placed = True
                        break
                
                # Create a new shelf if the item could not be placed

                if not placed: 
                    if first:
                        self._create_shelf(item)
                        first = False
                    else:
                        return


    def get_action_fcrg(self, observation, info):
        # Determine item placement in a stock.
        self._initialize_items(observation)

        list_stocks = list(enumerate(observation["stocks"]))

        # Sort stocks by available space (largest to smallest)
        list_stocks.sort(key=lambda list: np.sum(list[1] != -2), reverse=True)
        

        for i, stock in list_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            
            if self.last_shelves != None:
                if i == self.last_shelves["idx"]:
                    self.shelves = self.last_shelves["shelves"]
                else:
                    self.bin_width, self.bin_height = self._get_stock_size_(stock)
                    self._pack_fcrg()
            else:
                self.bin_width, self.bin_height = self._get_stock_size_(stock)
                self._pack_fcrg()

            # print("Shelves for stock", i, ":", self.shelves)

            for idx_shelf, shelf in enumerate(self.shelves):
                for item in shelf["items"]:
                    # Skip if the item is the last one in self.shelves
                    if self.last_prod == item and idx_shelf != len(self.shelves) - 1:
                        continue
                            
                    floors = self.floors.setdefault(i, [])
                    current_floor = 0

                    while True:
                        if current_floor < len(floors):
                            floor_y = sum(floors[:current_floor])
                            floor_height = floors[current_floor]
                        else:
                            # Create a new floor
                            if stock_w >= item[0] and stock_h >= item[1]:
                                floor_height = item[1]
                            else:
                                break  # No more space for new floors

                            floor_y = sum(floors) if floors else 0
                            if floor_y + floor_height <= stock_h:
                                self._create_new_floor_(i, floor_height)
                            else:
                                break  # No more vertical space for new floors

                        # Check if product can be placed on the ceiling
                        ceiling_y = floor_y + floor_height - item[1]
                        if ceiling_y >= 0 and ceiling_y < stock_h - item[1] + 1:
                            for x in range(stock_w - item[0] + 1):
                                if self._can_place_(stock, (x, ceiling_y), item):
                                    self.last_prod = item
                                    shelf["items"].remove(item)  # Remove item once placed
                                    self.last_shelves = {
                                        "idx": i,
                                        "shelves": self.shelves,
                                    }
                                    # print("stock_idx:", i, " size:", item, " position:", (x, ceiling_y))
                                    return {
                                        "stock_idx": i,
                                        "size": item,
                                        "position": (x, ceiling_y),
                                    }
                        if floor_y >= 0 and floor_y < stock_h - item[1] + 1:
                            for x in range(stock_w - item[0] + 1):
                                if self._can_place_(stock, (x, floor_y), item):
                                    self.last_prod = item
                                    shelf["items"].remove(item)  # Remove item once placed
                                    self.last_shelves = {
                                        "idx": i,
                                        "shelves": self.shelves,
                                    }
                                    # print("stock_idx:", i, " size:", item, " position:", (x, floor_y))
                                    return {
                                        "stock_idx": i,
                                        "size": item,
                                        "position": (x, floor_y),
                                    }
                        current_floor += 1

        return None
    
class ColumnGenerationPolicy(Policy):
    def __init__(self):
        self.initialized = False
        self.patterns = []          # Stores the patterns (columns)
        self.dual_variables = None  # Dual variables from the RMP
        self.num_prods = None       # Number of demands

    def _initialize_patterns(self, observation):
        # Initialize feasible patterns using a heuristic approach.
        stocks = observation["stocks"]
        products = observation["products"]
        
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            for product in products:
                if product["quantity"] > 0:
                    prod_w, prod_h = product["size"]
                    if stock_w >= prod_w and stock_h >= prod_h:
                        if self._can_place_(stock, (0, 0), (prod_w, prod_h)):
                            # Add pattern (stock_idx, size, position)
                            self.patterns.append({"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (0, 0)})

    def _solve_rmp(self, observation, demands):
        # Solve the Restricted Master Problem (RMP).
        
        c = []  # Cost of each pattern (area of stock used)
        A = []  # Demand constraints
        b = demands  # Product demands

        for pattern in self.patterns:
            stock_idx = pattern["stock_idx"]
            size = pattern["size"]
            stock = observation["stocks"][stock_idx]
            
            c.append(np.sum(stock != -2))  # Area cost
            row = [0] * len(b)

            for i, product in enumerate(observation["products"]):
                if product["size"][0] == size[0] and product["size"][1] == size[1]:
                    row[i] += 1  # This pattern can fulfill demand for product i

            A.append(row)

        # Solve the RMP using linear programming
        res = linprog(c, A_eq=np.array(A).T, b_eq=b, bounds=(0, None), method="highs")
        
        if not res.success:
            raise ValueError("RMP did not converge.")

        slack = np.array(A).T @ res.x - b
        self.dual_variables = -slack  # Use negative slack as dual variables
        return res.x

    def _generate_new_pattern(self, observation):
        # Generate a new pattern using the Sub Problem (SP).
        best_profit = -float('inf')
        best_pattern = None

        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            for product_idx, product in enumerate(observation["products"]):
                if product["quantity"] <= 0:
                    continue
                size = product["size"]
                profit = self.dual_variables[product_idx]  # Dual value
                
                if stock_w >= size[0] and stock_h >= size[1]:
                    for x in range(stock_w - size[0] + 1):
                            for y in range(stock_h - size[1] + 1):
                                if self._can_place_(stock, (x, y), size):
                                    position = (x, y)
                                    if self._can_place_(stock, position, size):
                                        # Calculate profit (dual value - cost)
                                        cost = np.prod(size)
                                        net_profit = profit - cost

                                        if net_profit - best_profit > 1e-6 :
                                            best_profit = net_profit
                                            best_pattern = {"stock_idx": stock_idx, "size": size, "position": position}

        if best_pattern:
            self.patterns.append(best_pattern)
        return best_pattern

    def get_action_colGen(self, observation, info):
        if not self.initialized:
            # Initialize with heuristic patterns
            self._initialize_patterns(observation)
            self.initialized = True

        # Solve RMP
        demands = [product["quantity"] for product in observation["products"]]
        solution = self._solve_rmp(observation, demands)

        # Check if we need a new pattern
        new_pattern = self._generate_new_pattern(observation)
        # print(new_pattern)
        if not new_pattern:
            # No new patterns; use the current solution
            chosen_pattern_idx = np.argmax(solution)
            return self.patterns[chosen_pattern_idx]
        else:
            # Add the new pattern to the RMP
            return new_pattern
