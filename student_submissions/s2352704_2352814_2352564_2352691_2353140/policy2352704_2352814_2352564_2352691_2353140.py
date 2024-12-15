from policy import Policy
import numpy as np

class Policy2352704_2352814_2352564_2352691_2353140(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        # Universal usage
        self.policy_id = policy_id
        self.init = False
        self.product_counter = 0
        self.list_products = []
        self.sorted_stocks = []
        if policy_id == 1:
            self.fit_all = None
            self.last_product = None
            self.total_area_of_products = 0
            self.cut_stock_counter = 0
        elif policy_id == 2:
            self.stock_at = 0
            self.patterns = {}
            self.used_patterns = {}
        
    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            """
            - Largest Fit 
            - Dynamic sorting
            """
            if not self.init:
                self.init = True
                self.fit_all = False
                for product in observation["products"]:
                    self.total_area_of_products += self._get_product_area_(product["size"]) * product["quantity"]
                    self.product_counter += product["quantity"]
                self.list_products = sorted(
                    enumerate(observation["products"]), 
                    key = lambda product: -self._get_product_area_(product[1]["size"])
                )
                self.sorted_stocks = sorted(
                    [{
                        "idx": stock_index, 
                        "info": stock, 
                        "cut": False, 
                        "remaining_area": np.sum(stock != -2)
                     }
                        for stock_index, stock in enumerate(observation["stocks"])
                    ], 
                    key = lambda stock: -stock["remaining_area"]
                )
            elif not self.fit_all:
                """
                - Sorting stocks based on no 'cut' and negative 'remaining_area' for largest fit
                - Example: 
                    With n = 5, maximum area is 10000 (100 x 100), products with areas of 4 (2 x 2) and 71 (1 x 71)
                    Result format will be something like this:
                    2 _ 0 2500 (50 x 50) -> 3 _ 1  138 -> 1 _ 1 1704
                    4 _ 0 2499 (49 x 51) -> 2 _ 0 2500 -> 3 _ 1  138
                    1 _ 0 1775 (25 x 71) -> 4 _ 0 2499 -> 2 _ 0 2500
                    0 _ 0  961 (31 x 31) -> 1 _ 0 1775 -> 4 _ 0 2499
                    3 _ 0  142 ( 2 x 71) -> 0 _ 0  961 -> 0 _ 0  961
                """
                self.sorted_stocks.sort(
                    key = lambda stock: (
                        not stock["cut"], 
                        -stock["remaining_area"]
                    )
                )
                """
                    [start : end = 0 : step = -1]
                - Running from the number of cut stocks or from the end if there isn't any cut stock
                - Prioritizing the specific smaller stock to minimize the waste and to minimize the stocks' usage
                because we speculated that stock can fit all the products inside (based on area)
                if not then we sort non-cut stocks ascendingly to get all the remaining products in case we also run out used stocks
                """
                stock_index = next((
                        stock["idx"] for stock in self.sorted_stocks[self.cut_stock_counter - 1 :: -1]
                        if self.total_area_of_products < stock["remaining_area"]
                    ),
                        next((
                            stock["idx"] for stock in self.sorted_stocks[::-1]
                            if self.total_area_of_products < stock["remaining_area"]
                        ), 
                        -1
                    )
                )
                if stock_index != -1:
                    self.fit_all = True
                    # Ascending for non-cut stocks after prioritizing the stock
                    self.sorted_stocks.sort(
                        key = lambda stock: (
                            stock["idx"] != stock_index, 
                            not stock["cut"], 
                            -stock["remaining_area"] if (stock["cut"]) else stock["remaining_area"]
                        )
                    )
            # Picking a product that has quantity > 0
            for product_index, product in self.list_products:
                if product["quantity"] > 0:
                    # Laying down the product (product_height < product_width)
                    product_size = product["size"]
                    product_width = product_height = 0
                    product_width, product_height = product_size[::-1] if (product_size[0] < product_size[1]) else product_size
                    area = product_width * product_height
                    # Looping through all sorted stocks
                    for stock in self.sorted_stocks:
                        stock_info = stock["info"]
                        stock_width, stock_height = self._get_stock_size_(stock_info)
                        # Checking if we can fit the producth
                        if (
                            stock["remaining_area"] < area
                            or (stock_width < product_width or stock_height < product_height) and (stock_width < product_height or stock_height < product_width)
                        ):
                            continue
                        for x in range(stock_width - product_height + 1):
                            for y in range(stock_height - product_height + 1):
                                product_state = [product_height, product_width] if (stock_width < x + product_width) else [product_width, product_height]
                                if stock_height < y + product_state[1]:
                                    break
                                position = (x, y)
                                if self._can_place_(stock_info, position, product_state):
                                    self.last_product = self.list_products[product_index][1]
                                    self.total_area_of_products -= area
                                    self.product_counter -= 1
                                    self.cut_stock_counter += 1
                                    stock["cut"] = True
                                    stock["remaining_area"] -= area
                                    # Reset if all products are placed
                                    if self.product_counter == 0:
                                        self.__init__()
                                    return {"stock_idx": stock["idx"], "size": product_state, "position": position}
            # Reset if all stocks are used but not all products are empty
            self.__init__()
        elif self.policy_id == 2:
            """
            - Column Generation
            """
            if not self.init:
                self.init = True
                self.stock_at = 0
                self.patterns.clear()
                self.used_patterns.clear()
                self.list_products = sorted(
                    observation["products"],
                    key = lambda product: -self._get_product_area_(product["size"])
                )
                self.sorted_stocks = sorted(
                    enumerate(observation["stocks"]),
                    key=lambda stock: -self._get_stock_area_(stock[1])
                )
                self.product_counter = np.sum(product["quantity"] for product in observation["products"])
            # Starting from the current stock
            for stock_index, stock in self.sorted_stocks[self.stock_at:]:
                stock_size = self._get_stock_size_(stock)
                if not self.patterns:
                    self.patterns = self._get_patterns_(stock_size)
                # Placing the products using existing patterns
                for pattern_index, pattern in enumerate(self.patterns):
                    if pattern_index not in self.used_patterns:
                        for product_index, product_usage_quantity in enumerate(pattern):
                            if product_usage_quantity > 0 and self.list_products[product_index]["quantity"] > 0:
                                product_size = self.list_products[product_index]["size"]
                                # Finding a possible placing position
                                suitable_position = None
                                for x in range(stock_size[0] - product_size[0] + 1):
                                    for y in range(stock_size[1] - product_size[1] + 1):
                                        position = (x, y)
                                        if self._can_place_(stock, position, product_size):
                                            suitable_position = position
                                            break
                                    if suitable_position is not None:
                                        break
                                if suitable_position is not None:
                                    self.product_counter -= 1
                                    if self.product_counter == 0:
                                        self.init = False
                                    return {"stock_idx": stock_index, "size": product_size, "position": (x, y)}
                self.stock_at += 1
            # Reset if all stocks are used but not all products are empty
            self.init = False
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    # Student code here
    # You can add more functions if needed
    # Column generation helpers
    def _get_patterns_(self, stock_size):
        """
        - Generating cut patterns
        """
        products = list(enumerate(self.list_products))
        patterns = list()
        # Picking a product that has quantity > 0
        for product_index, product in products:
            if product["quantity"] > 0:
                stock_width, stock_height = stock_size
                product_width, product_height = product["size"]
                if stock_width < product_width or stock_height < product_height:
                    continue
                # Estimating the quantity of the product we are going to use
                product_usage_quantity = min(
                    product["quantity"], 
                    (stock_width // product_width) * (stock_height // product_height)
                )
                if product_usage_quantity > 0:
                    pattern = np.zeros(len(products))
                    pattern[product_index] = product_usage_quantity
                    patterns.append(pattern)
        return patterns
    def _get_stock_area_(self, stock):
        stock_width, stock_height = self._get_stock_size_(stock)
        return stock_width * stock_height
    def _get_product_area_(self, product_size):
        product_width, product_height = product_size
        return product_width * product_height
