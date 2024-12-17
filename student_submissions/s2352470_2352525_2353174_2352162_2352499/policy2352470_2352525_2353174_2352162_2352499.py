from policy import Policy

class Policy2352470_2352525_2353174_2352162_2352499(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        # First Fit Decreasing - Policy 1
        if policy_id == 1:
            self.policy = FirstFitDecreasing()
        elif policy_id == 2:
            self.policy = BestFitDecreasing()
        
    def get_action(self, observation, info):
        return self.policy.get_action(observation=observation, info=info)
class FirstFitDecreasing(Policy):
    def __init__(self):
        self._sorted_products = None
        self._sorted_stocks = None

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]
        
        # Initiate or Reinitiate the sorted products and stacks
        if (info["filled_ratio"] == 0):
            self._sheets_sorting(stocks)
            self._products_sorting(products)
        
        # Return the result of first fit decreasing
        return self._first_fit_decreasing()
    
    def _products_sorting(self, products):
        self._sorted_products = sorted(
            [product for product in products if product["quantity"] > 0],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
    
    
    def _sheets_sorting(self, stocks):
        stocks_full = list()
        for i, stock in enumerate(stocks):
            w, h = self._get_stock_size_(stock)
            stocks_full.append(((w * h), i, stock))
            
        self._sorted_stocks = sorted(
            stocks_full,
            key=lambda stock: stock[0],
            reverse=True
        )
    
    def _first_fit_decreasing(self):
        for _, stock_idx, stock in self._sorted_stocks:
            stock_width, stock_height = self._get_stock_size_(stock)
            for product in self._sorted_products:
                if product["quantity"] <= 0:
                    continue
            
                prod_size = product["size"]
                prod_width, prod_height = prod_size
                or_prod_width, or_prod_height = prod_height, prod_width
                or_prod_size = (or_prod_width, or_prod_height)
                
                
                if stock_width < prod_width or stock_height < prod_height:
                    continue
                
                for x in range(stock_width - prod_width + 1):
                    for y in range(stock_height - prod_height + 1):
                        if self._can_place_(stock, (x, y), prod_size):  # Fixed method name
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (x, y)
                            }
                
                if stock_width < or_prod_width or stock_height < or_prod_height:
                    continue
            
                for x in range(stock_width - or_prod_width + 1):
                    for y in range(stock_height - or_prod_height + 1):
                        if self._can_place_(stock, (x, y), or_prod_size):  # Fixed method name
                            return {
                                "stock_idx": stock_idx,
                                "size": or_prod_size,
                                "position": (x, y)
                            }
        return None

class BestFitDecreasing(Policy):
    def __init__(self):
        self._sorted_products = None
        self._sorted_stocks = None
        self.threshold = 1e-6

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        # Initiate or Reinitiate the sorted products and stacks
        if info["filled_ratio"] == 0:
            self._sheets_sorting(stocks)
            self._products_sorting(products)

        # Return the result of best fit decreasing
        return self._best_fit_decreasing()

    def _products_sorting(self, products):
        self._sorted_products = sorted(
            [product for product in products if product["quantity"] > 0],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )

    def _sheets_sorting(self, stocks):
        stocks_full = list()
        for i, stock in enumerate(stocks):
            w, h = self._get_stock_size_(stock)
            stocks_full.append(((w * h), i, stock))
            
        self._sorted_stocks = sorted(
            stocks_full,
            key=lambda stock: stock[0],
            reverse=False
        )

    def _best_fit_decreasing(self):        
        best_fit = {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (None, None),
            "trim": float('inf'),  # Large value for initial comparison
            "cost": (float('inf'), float('inf'), float('inf'))
        }

        # Iterate over sorted products
        for product in self._sorted_products:
            if product["quantity"] <= 0:
                continue

            prod_size = product["size"]
            prod_width, prod_height = prod_size
            or_prod_width, or_prod_height = prod_height, prod_width

            # Iterate over sorted stocks
            for _, stock_idx, stock in self._sorted_stocks:
                stock_width, stock_height = self._get_stock_size_(stock)

                # Try both orientations
                for _, (w, h) in enumerate([(prod_width, prod_height), (or_prod_width, or_prod_height)]):
                    if stock_width < w or stock_height < h:
                        continue

                    # Check all potential positions
                    for x in range(stock_width - w + 1):
                        for y in range(stock_height - h + 1):
                            if self._can_place_(stock, (x, y), (w, h)):
                                remaining_area = (stock_width * stock_height) - (w * h)
                                cost = (remaining_area, y, x)

                                # Update best fit conditions
                                former_condition: bool = remaining_area < best_fit["trim"]
                                latter_condition: bool = remaining_area == best_fit["trim"] and cost < best_fit["cost"]
                                if former_condition or latter_condition:
                                    best_fit = {
                                        "stock_idx": stock_idx,
                                        "size": (w, h),
                                        "position": (x, y),
                                        "trim": remaining_area,
                                        "cost": cost
                                    }

                                # Early termination if optimal placement is found
                                if remaining_area <= self.threshold:
                                    product["quantity"] -= 1  # Decrement product quantity
                                    return {
                                        "stock_idx": best_fit["stock_idx"],
                                        "size": best_fit["size"],
                                        "position": best_fit["position"]
                                    }

            # Decrement product quantity after placement
            if best_fit["stock_idx"] != -1:
                product["quantity"] -= 1
                return {
                    "stock_idx": best_fit["stock_idx"],
                    "size": best_fit["size"],
                    "position": best_fit["position"]
                }

        # Default return if no placement is found
        return {
            "stock_idx": best_fit["stock_idx"],
            "size": best_fit["size"],
            "position": best_fit["position"]
        }
