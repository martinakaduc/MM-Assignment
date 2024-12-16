import numpy as np
from policy import Policy


class Policy2352117_2352743_2353077_2352698_2352152(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = BLD()
        elif policy_id == 2:
            self.policy = FFDInspired()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)


class BLD(Policy):
    def __init__(self):
        """
        Initializes the BLD policy.

        Attributes:
            counter (int): Tracks the number of actions taken.
            sorted_stocks_index (np.array): Array of stock indices sorted by area in descending order.
            sorted_products (list): List of products sorted by size in descending order.
        """
        self.counter = 0
        self.sorted_stocks_index = np.array([])
        self.sorted_products = []
        self.flag = True

    def get_action(self, observation, info):
        """
        Determines the action to take based on the current observation.

        Args:
            observation (dict): Current state of the environment, containing stocks and products.
            info (dict): Additional information provided by the environment.

        Returns:
            dict: Action to be taken, containing stock index, product size, and position.
        """
        # Sort products and stocks in decending order
        if self.counter == 0:
            self.sort_stock_product(
                observation['stocks'], observation['products']
            )
        self.counter += 1

        # When there's 1 product left, reset the counter
        if self.sorted_products[-1]['quantity'] == 1:
            self.counter = 0

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quantity > 0
        for prod in self.sorted_products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks in descending order
                for i in self.sorted_stocks_index:
                    stock_w, stock_h = self._get_stock_size_(
                        observation['stocks'][i]
                    )
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if not self.quick_check(
                                observation['stocks'][i], (x, y)
                            ):
                                continue
                            if self._can_place_(
                                observation['stocks'][i], (x, y), prod_size
                            ):
                                pos_x, pos_y = x, y
                                break

                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {
            "stock_idx": stock_idx,
            "size": prod_size,
            "position": (pos_x, pos_y),
        }

    def sort_stock_product(self, stock_array, prod_array):
        stock_areas = np.array([])
        for i in stock_array:
            stock_w, stock_h = self._get_stock_size_(i)
            stock_areas = np.append(stock_areas, stock_w * stock_h)

        self.sorted_stocks_index = np.argsort(stock_areas)[::-1]
        self.sorted_products = sorted(
            list(prod_array),
            key=lambda x: x['size'][0] * x['size'][1],
            reverse=True
        )

    def quick_check(self, stock, position):
        pos_x, pos_y = position
        return stock[pos_x, pos_y] == -1


class FFDInspired(Policy):
    def __init__(self):
        """
        Initializes the FFDInspired instance with default values.

        Attributes:
            actions (list): A list to store action sequences.
            action (int): The current index of the action to be taken.
            solved (bool): A flag to indicate if the problem has been solved.
        """
        self.actions = []
        self.action = 0
        self.solved = False

    def get_action(self, observation, info):
        """
        Retrieves the next action based on the current observation.

        If the problem has not been solved, it will first attempt to solve it
        by cutting the stocks based on the products. Once the solution is found,
        it cycles through the actions in a round-robin manner.

        Args:
            observation (dict): The current observation containing stock and product details.
            info (dict): Additional information for the action decision (not used here).

        Returns:
            dict: The action to be taken.
        """
        if not self.solved:
            stocks = []
            for id, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                stocks.append({
                    "height": stock_h,
                    "width": stock_w,
                    "count": 1,
                    "id": id
                })

            products = []
            for id, product in enumerate(observation["products"]):
                prod_w = product["size"][0]
                prod_h = product["size"][1]
                count = product["quantity"]
                products.append({
                    "height": prod_h,
                    "width": prod_w,
                    "count": count,
                    "id": id
                })

            self.cut_stocks(stocks, products)
            self.solved = True

        act = self.action
        acts = self.actions[act]
        self.action += 1

        # Reset class attributes when last action is returned
        if self.action == len(self.actions):
            self.actions = []
            self.action = 0
            self.solved = False

        return acts

    def cut_stocks(self, stocks, products):
        """
        Cuts stocks into products, updating the state of products and stocks.

        Args:
            stocks (list): A list of available stocks to be cut.
            products (list): A list of products to be placed in the stocks.
        """
        def not_empty(objects):
            for obj in objects:
                if obj["count"] > 0:
                    return True
            return False

        def cut_stock(stocks, products):
            """
            Cuts a stock into the products, trying to maximize the filled area.

            Args:
                stocks (list): A list of available stocks.
                products (list): A list of products to be placed in the stocks.

            Returns:
                tuple: The updated list of products, the best stock, and the cutting patterns.
            """
            products_state = None
            best_stock = None
            best_stock_patterns = None
            max_fill = 0  # Used to compare filled ratios between differrent stock choices

            # Loop through all stocks
            for stock in stocks:
                if stock["count"] <= 0:
                    continue

                products_dummy = [product.copy() for product in products]  # Preserve original products state
                filled_area = 0
                total_area = stock["height"] * stock["width"]
                patterns = []

                def span_stock(products, pos, scope, vertical, append):
                    """
                    Tries to place products in the given stock, either vertically or horizontally.

                    Args:
                        products (list): The products to be placed.
                        pos (tuple): The current position in the stock.
                        scope (tuple): The target space in the stock.
                        vertical (bool): Whether to place products vertically.
                        append (bool): Whether to append to the pattern list.

                    Returns:
                        tuple: The updated position and scope after placing products.
                    """
                    sub_pos = pos
                    sub_scope = scope
                    if scope[0] > 0 and scope[1] > 0:
                        nonlocal filled_area
                        used_count = 0

                        def rotate(product):
                            temp = product["height"]
                            product["height"] = product["width"]
                            product["width"] = temp
                            return product

                        def can_place(product, scope):
                            return product["height"] <= scope[0] and product["width"] <= scope[1]

                        def span(scope, product, vertical):
                            """
                            Calculates the span of a product placed within the stock.

                            Args:
                                product (dict): The product to place.
                                scope (tuple): The target space.
                                vertical (bool): Whether to place vertically.

                            Returns:
                                int: The span of the products.
                            """
                            if can_place(product, scope):
                                product_area = product["height"] * product["width"]
                                if vertical:
                                    product_span = product_area * min(scope[0] // product["height"], product["count"])
                                    return product_span
                                else:
                                    product_span = product_area * min(scope[1] // product["width"], product["count"])
                                    return product_span
                            return 0

                        # Compare spans between orientations
                        for product in products:
                            if span(scope, product, vertical) < span(scope, rotate(product.copy()), vertical):
                                rotate(product)

                        # Compare spans between products
                        best_product = max(products, key = lambda product: span(scope, product, vertical))

                        best_product_span = span(scope, best_product, vertical)
                        if not append:
                            return best_product_span

                        # Update next position and next scope
                        used_count = best_product_span // (best_product["height"] * best_product["width"])
                        if vertical:
                            sub_pos = (pos[0] + used_count * best_product["height"], pos[1])
                            sub_scope = (scope[0] - used_count * best_product["height"], best_product["width"])
                        else:
                            sub_pos = (pos[0], pos[1] + used_count * best_product["width"])
                            sub_scope = (best_product["height"], scope[1] - used_count * best_product["width"])

                        # Append pattern if the cut is non-trivial
                        if used_count:
                            patterns.append((best_product.copy(), used_count, pos, vertical))
                            best_product["count"] -= used_count
                            filled_area += best_product_span
                    return sub_pos, sub_scope

                def RFFD(products, pos=(0, 0), scope=(stock["height"], stock["width"])):
                    """
                    Recursive function to place products in the stock.

                    Args:
                        products (list): The products to be placed.
                        pos (tuple): The current position in the stock.
                        scope (tuple): The target space in the stock.
                    """
                    pos_original = pos
                    products_dummy = [product.copy() for product in products]  # Preserve products state for comparison
                    if span_stock(products_dummy, pos, scope, True, False) > span_stock(products_dummy, pos, scope, False, False):
                        sub_pos, sub_scope = span_stock(products, pos, scope, True, True)

                        # Update stock dimensions and call recursively if the cut is non-trivial
                        if sub_pos > pos:
                            pos = list(pos)
                            scope = list(scope)
                            pos[1] += sub_scope[1]
                            scope[1] -= sub_scope[1]
                            pos = tuple(pos)
                            scope = tuple(scope)
                            RFFD(products, sub_pos, sub_scope)
                    else:
                        sub_pos, sub_scope = span_stock(products, pos, scope, False, True)

                        # Update stock dimensions and call recursively if the cut is non-trivial
                        if sub_pos > pos:
                            pos = list(pos)
                            scope = list(scope)
                            pos[0] += sub_scope[0]
                            scope[0] -= sub_scope[0]
                            pos = tuple(pos)
                            scope = tuple(scope)
                            RFFD(products, sub_pos, sub_scope)

                    # Call recursively if the cut is non-trivial
                    if pos > pos_original:
                        RFFD(products, pos, scope)

                RFFD(products_dummy)

                fill = filled_area / total_area

                # Adopt best stock scenario
                if max_fill < fill:
                    products_state = [product.copy() for product in products_dummy]
                    best_stock = stock
                    best_stock_patterns = patterns
                    max_fill = fill

            best_stock["count"] -= 1
            return products_state, best_stock, best_stock_patterns

        def build_actions(stock, patterns):
            """
            Builds actions from the cutting patterns and adds them to the action list.

            Args:
                stock (dict): The stock from which the action is derived.
                patterns (list): The cutting patterns for the stock.
            """
            def build_action(pattern):
                product, used_count, pos, vertical = pattern
                (x, y) = pos  # Allocated in reversed order
                for _ in range(used_count):
                    self.actions.append({
                        "stock_idx": stock["id"],
                        "size": [product["width"], product["height"]],
                        "position": (y, x)
                    })
                    if vertical:
                        x += product["height"]
                    else:
                        y += product["width"]

            for pattern in patterns:
                build_action(pattern)

        while not_empty(products) and not_empty(stocks):
            products, stock, patterns = cut_stock(stocks, products)
            build_actions(stock, patterns)
