from policy import Policy

class Policy2352590(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        # Dynamically select the appropriate algorithm based on policy_id
        if self.policy_id == 1:
            return self.ffd_policy(observation)
        elif self.policy_id == 2:
            return self.column_generation_policy(observation)

    """
    Brief Description of First-Fit-Decreasing: 
    First Fit Decreasing (FFD) Algorithm
    The First Fit Decreasing Algorithm places products into the first stock that can accommodate them, starting with the largest products (sorted in decreasing order by size).

    Steps:

    Sort products by size in descending order (considering either width, height, or area).
    For each product, iterate through stocks to find the first one where it fits.
    Place the product in that stock.
    If rotation is allowed, try both orientations.
    Advantages: Simple and quick to implement with decent results.

    Disadvantages: Can leave more trim loss compared to Best Fit as it doesn’t optimize for space utilization.

    Average trim-loss (300) : 0.2076
    """
    def ffd_policy(self, observation):
    # Sort products by area in descending order
        list_prods = sorted(
            observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )
        stock_idx, pos_x, pos_y = -1, None, None
        prod_size = None

        # Iterate over sorted products
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                best_fit = None
                min_leftover_area = float("inf")

                # Iterate over stocks to find the best fit
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Check normal orientation
                    if stock_w >= prod_w and stock_h >= prod_h:
                        leftover_area = (stock_w * stock_h) - (prod_w * prod_h)
                        if leftover_area < min_leftover_area:
                            pos_x, pos_y = self._find_position_(stock, prod_size)
                            if pos_x is not None and pos_y is not None:
                                best_fit = (i, pos_x, pos_y, prod_size)
                                min_leftover_area = leftover_area

                    # Check rotated orientation
                    if stock_w >= prod_h and stock_h >= prod_w:
                        leftover_area = (stock_w * stock_h) - (prod_h * prod_w)
                        if leftover_area < min_leftover_area:
                            pos_x, pos_y = self._find_position_(stock, prod_size[::-1])
                            if pos_x is not None and pos_y is not None:
                                best_fit = (i, pos_x, pos_y, prod_size[::-1])
                                min_leftover_area = leftover_area

                # Place product in the best fit stock
                if best_fit:
                    stock_idx, pos_x, pos_y, prod_size = best_fit
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    """
    The Column Generation Algorithm is an optimization technique used primarily in solving large-scale linear programming problems, such as cutting stock or bin-packing
    Solves a master problem to optimize the current set of columns.
    Generates new columns by solving a subproblem that identifies potential improvements.
    Adds the new columns to the master problem and repeats until no further improvements can be made.
    This approach is efficient because it avoids enumerating all possibilities, focusing only on relevant patterns that improve the solution. It is widely used in logistics, scheduling, and manufacturing for minimizing waste or cost.
    """
    def column_generation_policy(self, observation):
        list_prods = observation["products"]
        stock_idx, pos_x, pos_y = -1, None, None
        prod_size = None
        best_placement = None

        # Iterate over products to generate columns (placements)
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_score = float("inf")  # Track the best placement score
                current_placement = None

                # Iterate over stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Generate columns for regular orientation
                    pos_x, pos_y = self._find_position_(stock, prod_size)
                    if pos_x is not None and pos_y is not None:
                        leftover_w = stock_w - pos_x - prod_w
                        leftover_h = stock_h - pos_y - prod_h
                        waste = (stock_w * stock_h) - (prod_w * prod_h)
                        usable_area_ratio = max(leftover_w, 0) * max(leftover_h, 0) / (stock_w * stock_h)
                        placement_score = waste + (1 - usable_area_ratio)  # Combine waste and usability
                        if placement_score < best_score:
                            best_score = placement_score
                            current_placement = (i, pos_x, pos_y, prod_size)

                    # Generate columns for rotated orientation
                    pos_x, pos_y = self._find_position_(stock, prod_size[::-1])
                    if pos_x is not None and pos_y is not None:
                        leftover_w = stock_w - pos_x - prod_h
                        leftover_h = stock_h - pos_y - prod_w
                        waste = (stock_w * stock_h) - (prod_h * prod_w)
                        usable_area_ratio = max(leftover_w, 0) * max(leftover_h, 0) / (stock_w * stock_h)
                        placement_score = waste + (1 - usable_area_ratio)  # Combine waste and usability
                        if placement_score < best_score:
                            best_score = placement_score
                            current_placement = (i, pos_x, pos_y, prod_size[::-1])

                # Update the best placement if found
                if current_placement:
                    best_placement = current_placement
                    break

        # Place the product
        if best_placement:
            stock_idx, pos_x, pos_y, prod_size = best_placement

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}



  
    def _find_position_(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None, None
