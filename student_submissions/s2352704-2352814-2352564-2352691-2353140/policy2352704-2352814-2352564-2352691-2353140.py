from policy import Policy
import numpy as np

class Policy2352814(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        if policy_id == 1:
            self.init = False

            self.list_prods = []
            self.fit_all = None
            self.last_prod = None
            self.prod_counter = 0
            self.total_area_of_products = 0

            self.sorted_stocks = []
            self.cut_stock_counter = 0
        elif policy_id == 2:
            pass
        
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
                self.list_prods = list(enumerate(observation["products"]))
                for _, prod in self.list_prods:
                    self.total_area_of_products += prod["size"][0] * prod["size"][1] * prod["quantity"]
                    self.prod_counter += prod["quantity"]
                self.sorted_stocks = sorted(
                    [{
                        "idx": stock_idx, 
                        "info": stock, 
                        "cut": False, 
                        "remaining_area": np.sum(np.any(stock != -2, axis=0)) * np.sum(np.any(stock != -2, axis=1))
                     }
                        for stock_idx, stock in enumerate(observation["stocks"])
                    ], 
                    key = lambda stock: -stock["remaining_area"]
                )
            elif not self.fit_all:
                """
                - Sorting stocks based on no 'cut' and negative 'remaining_area' for largest fit
                - Example: 
                With n = 5, maximum area is 10000 (100 x 100), 
                products with areas of 4 (2 x 2) and 71 (1 x 71)
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
            - Running from the number of cut stocks or from the end (start) if there isn't any cutted stocks
            to index 0 (end)
            - Prioritizing the specific smaller stock to minimize the waste and to minimize the stocks' usage
            because we speculated that stock can fit all the products inside (based on area)
            if not then we sort non-cut stocks ascendingly to get all the remaining products in case we also run out used stocks
            """
            if not self.fit_all:
                stock_idx = next((stock["idx"] 
                                    for stock in self.sorted_stocks[self.cut_stock_counter - 1 :: -1]
                                    if self.total_area_of_products < stock["remaining_area"]
                                ),
                                next((stock["idx"] 
                                        for stock in self.sorted_stocks[::-1]
                                        if self.total_area_of_products < stock["remaining_area"]
                                    ), 
                                    -1
                                )
                            )
                if stock_idx != -1:
                    self.fit_all = True
                    # Ascending for non-cut stocks after prioritizing the stock
                    self.sorted_stocks.sort(
                        key = lambda stock: (
                            stock["idx"] != stock_idx, 
                            not stock["cut"], 
                            -stock["remaining_area"] if (stock["cut"]) else stock["remaining_area"]
                        )
                    )
            # Picking a product that has quantity > 0
            for prod_idx, prod in self.list_prods:
                if prod["quantity"] > 0:
                    # Laying down the product (prod_h < prod_w)
                    prod_size = prod["size"]
                    prod_w = prod_h = 0
                    prod_w, prod_h = prod_size[::-1] if (prod_size[0] < prod_size[1]) else prod_size
                    area = prod_w * prod_h
                    # Looping through all sorted stocks
                    for stock in self.sorted_stocks:
                        stock_info = stock["info"]
                        stock_w, stock_h = self._get_stock_size_(stock_info)
                        # Checking if we can fit the producth
                        if (
                            stock["remaining_area"] < area
                            or (stock_w < prod_w or stock_h < prod_h) and (stock_w < prod_h or stock_h < prod_w)
                        ):
                            continue
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_h + 1):
                                if y + prod_h > stock_info.shape[1]:
                                    break
                                prod_state = [prod_w, prod_h] if (x + prod_w <= stock_info.shape[0]) else [prod_h, prod_w] if (prod_w > prod_h) else None
                                if prod_state == None:
                                    break
                                if self._can_place_(stock_info, (x, y), prod_state):
                                    self.last_prod = self.list_prods[prod_idx][1]
                                    self.total_area_of_products -= area
                                    self.prod_counter -= 1
                                    self.cut_stock_counter += 1
                                    stock["cut"] = True
                                    stock["remaining_area"] -= area
                                    # Reset if all products are placed
                                    if self.prod_counter == 0:
                                        self.__init__()
                                    return {"stock_idx": stock["idx"], "size": prod_state, "position": (x, y)}
        elif self.policy_id == 2:
            pass
        # Reset if all stocks are used but not all products are empty
        self.__init__()
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    # Student code here
    # You can add more functions if needed

