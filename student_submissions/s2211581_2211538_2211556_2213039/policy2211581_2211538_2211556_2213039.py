from policy import Policy

class Policy2211581_2211538_2211556_2213039(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
           self.policy = DynamicPriorityPolicyKnapsack()
        elif policy_id == 2:
           self.policy = DynamicPriorityPolicyV2()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

class DynamicPriorityPolicyV2(Policy):
    def __init__(self, efficiency_weight=0.5, area_weight=0.5):
        self.efficiency_weight = efficiency_weight
        self.area_weight = area_weight
        print("Dynamic Priority Policy V2")

    def get_action(self, observation, info):
        list_prods = observation["products"]
        prioritized_prods = []

        # Calculate priority dynamically with weighting factors
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                placement_options = 0
                total_free_area = 0

                # Calculate placement options and stock efficiency
                for stock in observation["stocks"]:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    if stock_w >= prod_w and stock_h >= prod_h:
                        placement_options += (stock_w - prod_w + 1) * (stock_h - prod_h + 1)
                        total_free_area += (stock_w * stock_h)

                # Calculate priority score with adaptive weights
                priority_score = (
                    self.efficiency_weight * (placement_options / (total_free_area + 1)) +
                    self.area_weight * (prod_w * prod_h)
                )

                prioritized_prods.append({
                    "product": prod,
                    "priority_score": priority_score
                })

        # Sort products by priority score descending
        prioritized_prods.sort(
            key=lambda x: -x["priority_score"]
        )

        # Attempt to place the highest priority product
        for item in prioritized_prods:
            prod = item["product"]
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y)
                                }

        # If no valid placement is found, return a default action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

class DynamicPriorityPolicyKnapsack(Policy):
    def __init__(self, efficiency_weight=0.5, area_weight=0.5):
        self.efficiency_weight = efficiency_weight
        self.area_weight = area_weight
        print("Dynamic Priority Policy Knapsack")

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # Filter products with available quantities
        available_prods = [
            prod for prod in list_prods if prod["quantity"] > 0
        ]

        # Calculate value-to-area ratio for each product (Knapsack-like heuristic)
        for prod in available_prods:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            area = prod_w * prod_h
            placement_options = self._count_placement_options(prod_size, stocks)

            # Add value-to-area ratio (priority) to product metadata
            prod["priority_score"] = (
                self.efficiency_weight * (placement_options or 1) +
                self.area_weight / area
            )

        # Sort products by priority score (higher is better)
        available_prods.sort(key=lambda x: -x["priority_score"])

        # Try placing products in stocks using a greedy placem  ent algorithm
        for prod in available_prods:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y)
                                }

        # If no valid placement is found, return a default action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _count_placement_options(self, prod_size, stocks):
        """Counts how many placement options exist for a product size in all stocks."""
        prod_w, prod_h = prod_size
        placement_options = 0

        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w >= prod_w and stock_h >= prod_h:
                placement_options += (stock_w - prod_w + 1) * (stock_h - prod_h + 1)

        return placement_options


