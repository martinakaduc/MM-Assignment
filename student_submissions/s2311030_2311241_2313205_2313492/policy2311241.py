import numpy as np
from policy import Policy
from gym_cutting_stock.envs.cutting_stock import CuttingStockEnv  # Assuming you're using Gym

class sub_env(CuttingStockEnv):
    def __init__(self, render_mode=None, **kwargs):
        """
        Initialize the ISHP environment.

        Args:
            render_mode: The rendering mode (e.g., "human").
            kwargs: Additional parameters passed to CuttingStockEnv.
        """
        super().__init__(render_mode=render_mode, **kwargs)

    def set_env(self, observation):
        self._stocks = [stock.copy() for stock in observation["stocks"]]
        self._products = [product.copy() for product in observation["products"]]

class Policy2311241(Policy):
    """
    Implements the Iterative Sequential Heuristic Procedure (ISHP) algorithm using only the 3SGP strategy.
    Tracks patterns, frequencies, and adjusts correction values dynamically.
    """

    def __init__(self) -> object:
        super().__init__()
        self.P = []  # Set of patterns
        self.F = {}  # Frequency of patterns
        self.correction_values = {}  # Correction values for products
        self.env = sub_env(
            # render_mode='human'
        )
        self.action_queue = []
        self.out_env_reset = True
        self.terminated = False
        self.observation = None
        self.info = None

    def get_action(self, observation, info):
        if self.out_env_reset:
            self.observation, self.info = self.env.reset()
            self.env.set_env(observation)
            self.out_env_reset = False
        if not self.terminated:
            actions = self.get_actions(self.observation, self.info)
            self.action_queue.extend(actions)
            for action in actions:
                self.observation, reward, self.terminated, truncated, self.info = self.env.step(action)
        action = self.action_queue.pop(0)
        if not self.action_queue:
            self.terminated = False
            self.out_env_reset = True
        return action

    def get_actions(self, observation, info):
        """
        Return the next action(s) based on the ISHP algorithm (using only 3SGP).

        Args:
            observation: Current state of stocks and products.
            info: Additional environment information.

        Returns:
            A list of actions (pattern) to apply to the environment.
        """
        stocks = observation["stocks"]
        products = observation["products"]

        # Step 1: Generate patterns using only 3SGP
        patterns = self.generate_patterns(stocks, products)

        if not patterns:
            return []  # No valid patterns available

        # Step 2: Evaluate patterns
        best_pattern = self.select_best_pattern(patterns)

        # Step 3: Adjust correction values
        self.adjust_correction_values(best_pattern, products)

        # Step 4: Update frequency of the chosen pattern
        pattern_key = self._pattern_to_key(best_pattern)
        self.F[pattern_key] = self.F.get(pattern_key, 0) + 1

        # Step 5: Convert the best pattern to a list of actions
        actions = [
            {"stock_idx": best_pattern["stock_idx"], "size": p["size"], "position": p["position"]}
            for p in best_pattern["placements"]
        ]

        return actions

    def generate_patterns(self, stocks, products):
        """
        Generate feasible patterns using only the 3SGP strategy.

        Args:
            stocks: List of stocks.
            products: List of products.

        Returns:
            List of feasible patterns.
        """
        patterns = []
        for stock_idx, stock in enumerate(stocks):
            stock_size = self._get_stock_size_(stock)

            # Create copies of stock and products for each pattern generation
            # stock_copy_for_3sgp = stock.copy()
            # products_copy_for_3sgp = [product.copy() for product in products]
            # pattern_3sgp = self.build_3sgp(stock_idx, stock_size, products_copy_for_3sgp, stock_copy_for_3sgp)

            stock_copy_for_3sgp_r = stock.copy()
            products_copy_for_3sgp_r = [product.copy() for product in products]
            pattern_3sgp_r = self.build_3sgp_r(stock_idx, stock_size, products_copy_for_3sgp_r, stock_copy_for_3sgp_r)

            # if pattern_3sgp:
            #     patterns.append(pattern_3sgp)
            if pattern_3sgp_r:
                patterns.append(pattern_3sgp_r)

        return patterns

    def select_best_pattern(self, patterns):
        """
        Select the best pattern based on utilization and frequency.

        Args:
            patterns: List of generated patterns.

        Returns:
            The best pattern.
        """
        def score(pattern):
            # Combine utilization and inverse frequency for scoring
            pattern_key = self._pattern_to_key(pattern)
            frequency = self.F.get(pattern_key, 0)
            return pattern["utilization"] - 0.2 * frequency  # Penalize high-frequency patterns

        return max(patterns, key=score)

    def adjust_correction_values(self, pattern, products):
        """
        Adjust correction values based on the selected pattern.

        Args:
            pattern: The selected pattern.
            products: List of products.
        """
        for placement in pattern["placements"]:
            product_idx = placement["product_idx"]
            product = products[product_idx]
            area = product["size"][0] * product["size"][1]
            utilization = pattern["utilization"]
            if product_idx not in self.correction_values:
                self.correction_values[product_idx] = 1.0

            # Update correction value
            self.correction_values[product_idx] = (
                    0.8 * self.correction_values[product_idx] + 0.2 * (area / utilization)
            )

    def build_3sgp_r(self, stock_idx, stock_size, products, stock):
        """
        Build patterns using the 3SHP logic with descending area sorting and correction value adjustment,
        including the ability to rotate pieces.

        Args:
            stock_idx: Index of the stock.
            stock_size: Size of the stock (width, height).
            products: List of products with their sizes and quantities.
            stock: The stock matrix.

        Returns:
            A 3SHP pattern with placements and utilization.
        """
        placements = []
        stock_width, stock_height = stock_size
        total_utilization = 0

        # Sort products by descending area (largest products first) and correction value
        sorted_products = sorted(
            [(product_idx, product) for product_idx, product in enumerate(products) if product["quantity"] > 0],
            key=lambda x: (x[1]["size"][0] * x[1]["size"][1], self.correction_values.get(x[0], 1.0)),
            reverse=True,
        )

        def recursive_partition(remaining_width, remaining_height, start_x, start_y):
            """
            Recursive helper function to partition and place products, with rotation.

            Args:
                remaining_width: Remaining width of the current partition.
                remaining_height: Remaining height of the current partition.
                start_x: Starting x-coordinate of the partition.
                start_y: Starting y-coordinate of the partition.

            Returns:
                Utilization for this partition.
            """
            nonlocal total_utilization

            if remaining_width <= 0 or remaining_height <= 0:
                return

            available_products = [
                (product_idx, product) for product_idx, product in sorted_products
                if product["quantity"] > 0 and (
                        (product["size"][0] <= remaining_width and product["size"][1] <= remaining_height) or
                        (product["size"][1] <= remaining_width and product["size"][0] <= remaining_height)
                )
            ]

            if not available_products:
                return

            # Try to place the largest fitting product
            product_idx, product = available_products[0]
            prod_width, prod_height = product["size"]

            if self._can_place_(stock, (start_x, start_y), (prod_width, prod_height)):
                # Place product in original orientation
                placements.append(
                    {"product_idx": product_idx, "size": (prod_width, prod_height), "position": (start_x, start_y)})
                total_utilization += prod_width * prod_height / (stock_width * stock_height)
                product["quantity"] -= 1

                # Recur to partition the remaining space
                recursive_partition(remaining_width, remaining_height - prod_height, start_x, start_y + prod_height)
                recursive_partition(remaining_width - prod_width, prod_height, start_x + prod_width, start_y)

                recursive_partition(remaining_width - prod_width, remaining_height - prod_height, start_x + prod_width, start_y + prod_height)

            elif self._can_place_(stock, (start_x, start_y), (prod_height, prod_width)):
                # Place product in rotated orientation
                placements.append(
                    {"product_idx": product_idx, "size": (prod_height, prod_width), "position": (start_x, start_y)})
                total_utilization += prod_width * prod_height / (stock_width * stock_height)
                product["quantity"] -= 1

                # Recur to partition the remaining space
                recursive_partition(remaining_width - prod_height, prod_width, start_x + prod_height, start_y)
                recursive_partition(remaining_width - prod_width, remaining_height - prod_height, start_x + prod_width, start_y + prod_height)
                recursive_partition(remaining_width, remaining_height - prod_width, start_x, start_y + prod_width)


        recursive_partition(stock_width, stock_height, 0, 0)

        return {
            "stock_idx": stock_idx,
            "placements": placements,
            "utilization": total_utilization,
        } if placements else None


    def build_3sgp(self, stock_idx, stock_size, products, stock):
        """
        Build patterns using the 3SHP logic with descending area sorting and correction value adjustment.

        Args:
            stock_idx: Index of the stock.
            stock_size: Size of the stock (width, height).
            products: List of products with their sizes and quantities.
            stock: The stock matrix.

        Returns:
            A 3SHP pattern with placements and utilization.
        """
        placements = []
        stock_width, stock_height = stock_size
        total_utilization = 0

        # Sort products by descending area (largest products first) and correction value
        # sorted_products = sorted(
        #     [(product_idx, product) for product_idx, product in enumerate(products) if product["quantity"] > 0],
        #     key=lambda x: (x[1]["size"][0] * x[1]["size"][1], self.correction_values.get(x[0], 1.0)),
        #     reverse=True,
        # )
        sorted_products = sorted(
            [(product_idx, product) for product_idx, product in enumerate(products) if product["quantity"] > 0],
            key=lambda x: (x[1]["size"][0] * x[1]["size"][1]),
            reverse=True,
        )

        def recursive_partition(remaining_width, remaining_height, start_x, start_y):
            """
            Recursive helper function to partition and place products.

            Args:
                remaining_width: Remaining width of the current partition.
                remaining_height: Remaining height of the current partition.
                start_x: Starting x-coordinate of the partition.
                start_y: Starting y-coordinate of the partition.

            Returns:
                Utilization for this partition.
            """
            nonlocal total_utilization

            if remaining_width <= 0 or remaining_height <= 0:
                return

            available_products = [
                (product_idx, product) for product_idx, product in sorted_products
                if product["quantity"] > 0 and ((product["size"][0] <= remaining_width and product["size"][
                    1] <= remaining_height) or (product["size"][1] <= remaining_width and product["size"][0] <= remaining_height))
            ]

            if not available_products:
                return

                # Try to place the largest fitting product
            product_idx, product = available_products[0]
            prod_width, prod_height = product["size"]

            # Check if the product can be placed in the current region using the _can_place method
            if self._can_place_(stock, (start_x, start_y), product["size"]):
                placements.append({"product_idx": product_idx, "size": product["size"], "position": (start_x, start_y)})
                total_utilization += prod_width * prod_height / (stock_width * stock_height)
                product["quantity"] -= 1

                # Recur to partition the remaining space after placing the product
                recursive_partition(remaining_width - prod_width, prod_height, start_x + prod_width, start_y)
                recursive_partition(remaining_width, remaining_height - prod_height, start_x, start_y + prod_height)
                recursive_partition(remaining_width - prod_width, remaining_height - prod_height, start_x + prod_width, start_y + prod_height)

        recursive_partition(stock_width, stock_height, 0, 0)

        return {
            "stock_idx": stock_idx,
            "placements": placements,
            "utilization": total_utilization,
        } if placements else None

    def _pattern_to_key(self, pattern):
        """
        Create a unique key for a pattern to track frequency.

        Args:
            pattern: A pattern.

        Returns:
            A tuple representing the unique key of the pattern.
        """
        placements_key = tuple(
            (p["product_idx"], tuple(p["size"]), tuple(p["position"])) for p in pattern["placements"])
        return (pattern["stock_idx"], placements_key)

    def _get_stock_size_(self, stock):
        """
        Get the dimensions of a stock (width and height).

        Args:
            stock: A 2D array representing the stock matrix.

        Returns:
            A tuple (width, height) of the stock.
        """
        stock_width = np.sum(np.any(stock != -2, axis=1))  # Width of the stock (non -2 cells)
        stock_height = np.sum(np.any(stock != -2, axis=0))  # Height of the stock (non -2 cells)
        return stock_width, stock_height