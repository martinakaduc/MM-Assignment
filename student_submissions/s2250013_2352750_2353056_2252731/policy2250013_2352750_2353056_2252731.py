import numpy as np

from policy import Policy


class Policy1(Policy):
    """
    A class that defines a policy for selecting and placing products into stocks
    based on their sizes and availability.

    Inherits:
        Policy: A base class for policies.
    """

    def __init__(self):
        """
        Initialize the Policy1 instance.

        Inherits:
            Policy: Calls the initializer of the base Policy class.
        """
        super().__init__()

    def get_action(self, observation, info):
        """
        Determines the best placement of a product into a stock.

        Args:
            observation (dict): A dictionary containing:
                - 'stocks' (tuple[np.ndarray]): A tuple of 2D arrays representing stocks.
                  Each stock is represented as a grid where -1 indicates an empty slot.
                - 'products' (tuple[dict]): A tuple of dictionaries representing products,
                  each with keys:
                  - 'size' (np.ndarray): A 2-element array [width, height] indicating the product size.
                  - 'quantity' (int): The number of items of this product available.
            info (dict): Additional information (not used in this implementation).

        Returns:
            dict or None: A dictionary representing the placement action, with keys:
                - 'stock_idx' (int): Index of the selected stock.
                - 'size' (np.ndarray): The size of the product being placed.
                - 'position' (tuple[int, int]): The (x, y) coordinates for the placement in the stock.

            Returns None if no suitable placement is found.
        """
        stocks = observation['stocks']
        products = observation['products']

        # Sort products by area (width * height) in descending order
        sorted_products = sorted(
            products,
            key=lambda _p: _p['size'][0] * _p['size'][1],
            reverse=True
        )

        for product in sorted_products:
            product_size = product['size']

            if product['quantity'] > 0:
                for stock_idx, stock in enumerate(stocks):
                    stock_matrix = np.array(stock)

                    best_fit = None

                    # Iterate over possible placement positions in the stock
                    for x in range(stock_matrix.shape[0] - product_size[0] + 1):
                        for y in range(stock_matrix.shape[1] - product_size[1] + 1):
                            # Check if the product fits in the candidate area
                            candidate_area = stock_matrix[
                                             x:x + product_size[0],
                                             y:y + product_size[1],
                                             ]

                            if np.all(candidate_area == -1):  # Check if all slots are empty
                                waste = stock_matrix.size - product_size[0] * product_size[1]
                                if best_fit is None or waste < best_fit['waste']:
                                    best_fit = {
                                        'x': x,
                                        'y': y,
                                        'waste': waste,
                                    }

                    if best_fit:
                        # Return the placement action
                        return {
                            'stock_idx': stock_idx,
                            'size': product_size,
                            'position': (best_fit['x'], best_fit['y']),
                        }

        # Return None if no suitable placement is found
        return None


class Policy2(Policy):
    """
    A class implementing a stock placement policy with advanced arrangement
    generation and position identification logic.

    Attributes:
        pattern_collection (dict): Collection of generated patterns for product arrangements.
        utilized_patterns (set): Tracks the patterns already used.
        active_stock (int): Index of the currently active stock being processed.
        occupied_positions (dict): Tracks occupied positions in each stock.
        initiate_game (bool): Flag to reinitialize the state at the start of a game.
    """

    def __init__(self):
        """
        Initializes the Policy2 instance with empty tracking attributes.
        """
        super().__init__()
        self.pattern_collection = {}
        self.utilized_patterns = set()
        self.active_stock = 0
        self.occupied_positions = {}
        self.initiate_game = True

    def get_action(self, observation, info):
        """
        Determines the best placement of a product into a stock based on
        the current state of stocks and product availability.

        Args:
            observation (dict): Contains:
                - "stocks" (list[np.ndarray]): List of 2D arrays representing the stocks.
                - "products" (list[dict]): List of dictionaries with product details:
                    - "size" (list[int]): Dimensions [width, height] of the product.
                    - "quantity" (int): Quantity available for this product.
            info (dict): Additional information (not used in this implementation).

        Returns:
            dict: Action indicating the stock, product size, and position for placement:
                - "stock_idx" (int): Index of the stock to place the product in.
                - "size" (np.ndarray): Dimensions of the product to place.
                - "position" (np.ndarray): Coordinates (x, y) for placement.
            Returns a default action with -1 stock index if placement is not possible.
        """
        if self.initiate_game:
            self.pattern_collection.clear()
            self.utilized_patterns.clear()
            self.active_stock = 0
            self.occupied_positions.clear()
            self.initiate_game = False

        available_stocks = observation["stocks"]
        available_items = observation["products"]
        remaining_items = np.sum([item["quantity"] for item in available_items])

        if remaining_items == 1:
            self.initiate_game = True

        if not any(item["quantity"] > 0 for item in available_items):
            return {
                "stock_idx": -1,
                "size": np.array([0, 0]),
                "position": np.array([0, 0])
            }

        available_items = sorted(
            available_items,
            key=lambda x: -x['size'][0] * x['size'][1]
        )

        sorted_stocks = sorted(
            enumerate(available_stocks),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True
        )

        for idx, _ in sorted_stocks[self.active_stock:]:
            if idx not in self.occupied_positions:
                self.occupied_positions[idx] = set()

            stock_dimensions = self._get_stock_size_(available_stocks[idx])

            if not self.pattern_collection:
                self.pattern_collection = self.generate_arrangements(available_items, stock_dimensions)

            for pattern_idx, pattern in enumerate(self.pattern_collection):
                if pattern_idx not in self.utilized_patterns:
                    for item_idx, count in enumerate(pattern):
                        if count > 0 and available_items[item_idx]["quantity"] > 0:
                            locations = self.identify_positions(
                                available_stocks[idx],
                                available_items[item_idx]["size"]
                            )

                            for loc in locations:
                                loc_key = (*loc, *available_items[item_idx]["size"])
                                if loc_key not in self.occupied_positions[idx]:
                                    self.occupied_positions[idx].add(loc_key)
                                    return {
                                        "stock_idx": idx,
                                        "size": np.array(available_items[item_idx]["size"]),
                                        "position": np.array(loc)
                                    }

            self.active_stock += 1
            if self.active_stock >= len(available_stocks):
                self.active_stock = 0
                self.pattern_collection = self.generate_arrangements(available_items, stock_dimensions)
                self.utilized_patterns.clear()

        return {
            "stock_idx": -1,
            "size": np.array([0, 0]),
            "position": np.array([0, 0])
        }

    def identify_positions(self, container, item_dimensions):
        """
        Identifies all valid positions in a stock where an item can be placed.

        Args:
            container (np.ndarray): A 2D array representing the stock.
            item_dimensions (list[int]): Dimensions [width, height] of the item.

        Returns:
            list[tuple[int, int]]: List of (x, y) coordinates for valid positions.
        """
        container_w, container_h = self._get_stock_size_(container)
        item_w, item_h = item_dimensions
        positions = []

        for x in range(container_w - item_w + 1):
            for y in range(container_h - item_h + 1):
                if self._can_place_(container, (x, y), (item_w, item_h)):
                    positions.append((x, y))

        return positions

    @staticmethod
    def generate_arrangements(items, container_dimensions):
        """
        Generates possible arrangements of items within the given container dimensions.

        Args:
            items (list[dict]): List of product dictionaries with "size" and "quantity".
            container_dimensions (tuple[int, int]): Dimensions [width, height] of the container.

        Returns:
            list[list[int]]: List of arrangements, where each arrangement is a list
            indicating the count of each item to place.
        """
        items = sorted(enumerate(items), key=lambda x: -x[1]['size'][0] * x[1]['size'][1])
        arrangements = []

        for i, item in items:
            if item['quantity'] > 0:
                container_w, container_h = container_dimensions
                item_w, item_h = item['size']

                if item_w > container_w or item_h > container_h:
                    continue

                arrangement = [0] * len(items)
                remaining_w, remaining_h = container_w, container_h
                max_in_row = remaining_w // item_w
                max_in_column = remaining_h // item_h
                count = min(item['quantity'], max_in_row * max_in_column)

                if count > 0:
                    arrangement[i] = count

                    for j, other_item in items:
                        if j != i and other_item['quantity'] > 0:
                            w2, h2 = other_item['size']
                            if w2 <= remaining_w and h2 <= remaining_h:
                                max_other = min(
                                    other_item['quantity'],
                                    (remaining_w // w2) * (remaining_h // h2)
                                )
                                if max_other > 0:
                                    arrangement[j] = max_other
                                    remaining_w -= w2 * max_other
                                    remaining_h -= h2 * max_other

                    arrangements.append(arrangement)

        return arrangements


class Policy2250013_2352750_2353056_2252731(Policy):
    """
    A composite policy class that delegates actions to either Policy1 or Policy2
    based on a specified policy ID during initialization.

    Attributes:
        policy (Policy): An instance of either Policy1 or Policy2, selected
                         based on the provided policy ID.
    """

    def __init__(self, policy_id=1):
        """
        Initializes the composite policy with the specified policy ID.

        Args:
            policy_id (int): The ID of the policy to use (1 for Policy1, 2 for Policy2).

        Raises:
            AssertionError: If the policy_id is not 1 or 2.
        """
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy = Policy1() if policy_id == 1 else Policy2()

    def get_action(self, observation, info):
        """
        Delegates the action decision to the selected policy.

        Args:
            observation (dict): Input observation, forwarded to the selected policy's `get_action` method.
            info (dict): Additional information, forwarded to the selected policy's `get_action` method.

        Returns:
            dict: The action returned by the selected policy's `get_action` method.
        """
        return self.policy.get_action(observation, info)


if __name__ == '__main__':
    import gymnasium as gym

    env = gym.make(
        "gym_cutting_stock/CuttingStock-v0",
        render_mode="human",  # Comment this line to disable rendering
    )
    NUM_EPISODES = 100

    observation, info = env.reset(seed=42)
    print(info)

    policy2210xxx = Policy2250013_2352750_2353056_2252731(policy_id=2)
    for _ in range(200):
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
