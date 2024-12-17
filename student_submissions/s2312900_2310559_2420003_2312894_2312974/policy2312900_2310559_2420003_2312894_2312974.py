from policy import Policy
import numpy as np

class Policy2312900_2310559_2420003_2312894_2312974(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        
        #Data initalization for Simulated Annealing
        self.initial_temperature = 100
        self.cooling_rate = 0.99
        self.max_iterations = 100
        self.restarts = 5


    def get_action(self, observation, info):
        # Id 1 for First Fit Decreasing implementation
        if self.policy_id == 1:
            #print("Products information before sorting: ", observation["products"])
            
            # Sort the products by size in descending order.
            # sorted() function is a built-in function for sorting. For more information see the documentation.
            products = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            #print("Products information after sorting: ", products)
            
            #Take out the information about available stocks
            stocks = observation["stocks"]
            
            # Iterate through the products
            for product in products:
                
                # If the quantity of current product is greater than 0,
                # continue to insert it into the stocks
                if product["quantity"] > 0:
                    
                    # Get the size of the current product
                    prod_size = product["size"]
                    
                    # Iterate through all available stocks.
                    # Iterate using enumerate() to get both the index and the stock,
                    # See the document for information.
                    for stock_idx, stock in enumerate(stocks):
                        
                        # Get the size of the current stock
                        stock_w, stock_h = self._get_stock_size_(stock)
                        
                        # Get the size of the current product into two variables
                        prod_w, prod_h = prod_size

                        # Initial condition: The size of the product must be less than or equal to the size of the stock.
                        if stock_w >= prod_w and stock_h >= prod_h:
                            
                            # Iterate through all possible indexes in the x coordinate where we can put 
                            # the top-left corner of the product.
                            for x in range(stock_w - prod_w + 1):
                                
                                # Inner loop to iterate through all possible indexes in the y coordinate where we can put
                                # the bootom-right corner of the product.
                                for y in range(stock_h - prod_h + 1):
                                    
                                    # Check if the product can be placed at the current position
                                    # If yes, return the information for the environment to place the product.
                                    if self._can_place_(stock, (x, y), prod_size):
                                        
                                        # Necessary datas for the environment to place the product,
                                        # including: the index of the stock that we insert the new product,
                                        # the size of the product, and the position of the top-left corner of the product.
                                        return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}

                        # else, rotate the product and check if it can fit into the stock
                        if stock_w >= prod_h and stock_h >= prod_w:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    
                                    # The slicing notation [::-1] is used to reverse the order of the elements in the list,
                                    # which means that the width and height of the product size are swapped.
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (x, y)}

            # If no valid position is found, return a dummy action
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
        
        # Id 2 for Genetic Algorithm Implementation
        elif self.policy_id == 2:
            """
            Get the best action for the current observation using simulated annealing.
            """
            products = observation["products"]
            stocks = observation["stocks"]
            best_solution = self.simulated_annealing(products, stocks)
            if best_solution:
                best_action = {
                    "stock_idx": best_solution[0][0],
                    "size": (best_solution[0][3], best_solution[0][4]),
                    "position": (best_solution[0][1], best_solution[0][2]),
                }
            else:
                best_action = {"stock_idx": 0, "size": [0, 0], "position": (0, 0), "rotated": False}
            return best_action

    ##################################
    #Helping functions for First Fit Decreasing
    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, size):
        # Check if the product can be placed at the given position
        
        # Get the x and y coordinates of the top-left corner of the product
        pos_x, pos_y = position
        
        # Get the width and height of the product
        width, height = size
        
        # As the "stocks" key of the observation is a list of 2D numpy arrays,
        # we check using the dimension of the stock and the position of the product
        if pos_x + width <= stock.shape[0] and pos_y + height <= stock.shape[1]:
            if np.all(stock[pos_x:pos_x + width, pos_y:pos_y + height] == -1):
                return True
        return False

    def calculate_available_space(self, stock):
        # Count the number of -1 elements in the stock
        available_space = np.sum(stock == -1)
        return available_space
    
    ##################################
    #Helping functions for Simulated Annealing
    def simulated_annealing(self, products, stocks):
        """
        Main Simulated Annealing algorithm.
        """
        current_solution = self.initialize_solution(products, stocks)
        current_fitness = self.evaluate_solution(current_solution, stocks)
        best_solution = current_solution
        best_fitness = current_fitness

        temperature = self.initial_temperature

        for iteration in range(self.max_iterations):
            neighbor = self.generate_neighbor(current_solution, products, stocks)
            neighbor_fitness = self.evaluate_solution(neighbor, stocks)

            # Accept neighbor with probability
            delta = neighbor_fitness - current_fitness
            if delta > 0 or random.uniform(0, 1) < np.exp(delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness:
                    best_solution = current_solution
                    best_fitness = current_fitness

            # Cool down
            temperature *= self.cooling_rate

        return best_solution
    
    def generate_neighbor(self, solution, products, stocks):
        """
        Generate a neighbor solution by intelligently modifying one cut.
        """
        neighbor = solution.copy()
        idx = random.randint(0, len(neighbor) - 1)
        product = products[idx]
        width, height = product['size']
        stock_idx = random.randint(0, len(stocks) - 1)
        stock_w, stock_h = self._get_stock_size_(stocks[stock_idx])

        # Randomly decide whether to rotate the product
        if random.random() < 0.5:
            width, height = height, width
            rotated = True
        else:
            rotated = False

        # Localized adjustments: small shifts in x and y coordinates
        shift_x = random.randint(-1, 1)
        shift_y = random.randint(-1, 1)
        x = max(0, min(stock_w - width, neighbor[idx][1] + shift_x))
        y = max(0, min(stock_h - height, neighbor[idx][2] + shift_y))

        # Edge and corner placement
        if random.random() < 0.3:
            if random.random() < 0.5:
                x = 0 if random.random() < 0.5 else stock_w - width
            else:
                y = 0 if random.random() < 0.5 else stock_h - height

        # Swap positions with another product
        if random.random() < 0.2:
            swap_idx = random.randint(0, len(neighbor) - 1)
            neighbor[idx], neighbor[swap_idx] = neighbor[swap_idx], neighbor[idx]
        else:
            neighbor[idx] = (stock_idx, x, y, width, height, rotated)

        return neighbor
    
    def evaluate_solution(self, solution, stocks):
        """
        Evaluate a solution based on waste and feasibility.
        """
        used_stocks = np.copy(stocks)
        waste = 0
        for stock_idx, x, y, width, height, rotated in solution:
            if np.all(used_stocks[stock_idx][y:y+height, x:x+width] == -1):
                used_stocks[stock_idx][y:y+height, x:x+width] = 1  # Mark as used
            else:
                waste += width * height  # Penalize overlap
        return -waste  # Higher fitness for lower waste
    
    def initialize_solution(self, products, stocks):
        """
        Generate an initial random solution.
        Each solution is a list of cuts representing (stock_idx, x, y, width, height, rotated).
        """
        solution = []
        for product in products:
            if product["quantity"] == 0:
                continue
            width, height = product['size']
            placed = False
            while not placed:
                stock_idx = random.randint(0, len(stocks) - 1)
                stock_w, stock_h = self._get_stock_size_(stocks[stock_idx])
                if stock_w >= width and stock_h >= height:
                    x = random.randint(0, stock_w - width)
                    y = random.randint(0, stock_h - height)
                    if self._can_place_(stocks[stock_idx], (x, y), (width, height)):
                        solution.append((stock_idx, x, y, width, height, False))
                        placed = True
                elif stock_w >= height and stock_h >= width:
                    x = random.randint(0, stock_w - height)
                    y = random.randint(0, stock_h - width)
                    if self._can_place_(stocks[stock_idx], (x, y), (height, width)):
                        solution.append((stock_idx, x, y, height, width, True))
                        placed = True
        return solution
