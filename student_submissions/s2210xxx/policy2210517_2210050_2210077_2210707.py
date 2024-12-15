from policy import Policy
from policy import GreedyPolicy, RandomPolicy

class Policy2210517_2210050_2210077_2210707(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = ColGenPolicy()
        elif policy_id == 2:
            self.policy = RandomPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed

class ColGenPolicy(Policy):
    def __init__(self):
        super().__init__()
        # Initialize class variables
        self.products = None  # List of products
        self.stock_size = None  # Dimensions of the stock (width, height)
        self.patterns = None  # Patterns for column generation

    def initialize(self, observation):
        """
        Initialize the policy with the observation data.
        """
        self.products = observation["products"]
        stock_example = observation["stocks"][0]
        self.stock_size = self._get_stock_size_(stock_example)
        self.patterns = []  # Initialize an empty list of patterns

    def get_action(self, observation, info):
        """
        Generate the best placement for products using column generation.
        """
        # Ensure initialization is done at the start
        if self.products is None or self.stock_size is None:
            self.initialize(observation)

        selected_stock_idx, selected_x, selected_y = -1, None, None
        selected_product_size = None
        optimal_placement = None

        # Iterate over all products to determine placement
        for product in self.products:
            if product["quantity"] > 0:  # Only consider products with remaining quantity
                product_size = product["size"]
                min_placement_score = float("inf")  # Track the best score for placement
                current_best_placement = None  # Best placement configuration for this product

                # Iterate over all available stocks
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_width, stock_height = self._get_stock_size_(stock)
                    product_width, product_height = product_size

                    # Check placement in regular orientation
                    pos_x, pos_y = self._find_position_(stock, product_size)
                    if pos_x is not None and pos_y is not None:
                        remaining_width = stock_width - pos_x - product_width
                        remaining_height = stock_height - pos_y - product_height
                        wasted_area = (stock_width * stock_height) - (product_width * product_height)
                        usable_area_ratio = max(remaining_width, 0) * max(remaining_height, 0) / (stock_width * stock_height)
                        placement_score = wasted_area + (1 - usable_area_ratio)  # Scoring based on waste and usability

                        # Update best score and placement for this product
                        if placement_score < min_placement_score:
                            min_placement_score = placement_score
                            current_best_placement = (stock_idx, pos_x, pos_y, product_size)

                    # Check placement in rotated orientation
                    pos_x, pos_y = self._find_position_(stock, product_size[::-1])
                    if pos_x is not None and pos_y is not None:
                        remaining_width = stock_width - pos_x - product_height
                        remaining_height = stock_height - pos_y - product_width
                        wasted_area = (stock_width * stock_height) - (product_height * product_width)
                        usable_area_ratio = max(remaining_width, 0) * max(remaining_height, 0) / (stock_width * stock_height)
                        placement_score = wasted_area + (1 - usable_area_ratio)  # Scoring based on waste and usability

                        # Update best score and placement for rotated product
                        if placement_score < min_placement_score:
                            min_placement_score = placement_score
                            current_best_placement = (stock_idx, pos_x, pos_y, product_size[::-1])

                # If a valid placement is found, update the optimal placement
                if current_best_placement:
                    optimal_placement = current_best_placement
                    break  # Stop searching once the first valid placement is found

        # If a valid placement is found, update the selected placement details
        if optimal_placement:
            selected_stock_idx, selected_x, selected_y, selected_product_size = optimal_placement

        # Return the placement action
        return {
            "stock_idx": selected_stock_idx,
            "size": selected_product_size,
            "position": (selected_x, selected_y),
        }

    def _find_position_(self, stock, product_size):
        """
        Find a position in the stock where the product can be placed.
        """
        stock_width, stock_height = self._get_stock_size_(stock)
        product_width, product_height = product_size

        # Iterate over possible positions in the stock
        for x in range(stock_width - product_width + 1):
            for y in range(stock_height - product_height + 1):
                # Check if the product can be placed at this position
                if self._can_place_(stock, (x, y), product_size):
                    return x, y  # Return valid position

        # Return None if no valid position is found
        return None, None
