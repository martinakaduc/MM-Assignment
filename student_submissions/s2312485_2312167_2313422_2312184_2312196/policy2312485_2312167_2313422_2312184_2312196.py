from policy import Policy


class Policy2312485_2312167_2313422_2312184_2312196(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = LargestFirst()
        elif policy_id == 2:
            self.policy = BestFit()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed
class LargestFirst(Policy):
        def __init__(self):
            pass

        def _find_placement(self, stock, prod_size):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        return x, y
            return None, None
        
        def get_action(self, observation, info):
            list_prods = sorted(
                observation["products"],
                key=lambda p: p["size"][0] * p["size"][1],
                reverse=True
            )
            stock_sizes = [
                (self._get_stock_size_(stock), idx) for idx, stock in enumerate(observation["stocks"])
            ]
            stock_sizes.sort(key=lambda x: (x[0][0] * x[0][1], x[0][0]), reverse=True)
            for prod in list_prods:
                if prod["quantity"] > 0:
                    for (stock_size, i) in stock_sizes:
                        stock = observation["stocks"][i]
                        rotations = [prod["size"], prod["size"][::-1]]
                        for size in rotations:
                            pos_x, pos_y = self._find_placement(stock, size)
                            if pos_x is not None and pos_y is not None:
                                return {
                                    "stock_idx": i,
                                    "size": size,
                                    "position": (pos_x, pos_y)
                                }
class BestFit(Policy):
    def __init__(self):
        pass

    def _find_placement(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        best_x, best_y = None, None
        min_remaining_area = float('inf')

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    remaining_width = stock_w - (x + prod_w)
                    remaining_height = stock_h - (y + prod_h)
                    remaining_area = remaining_width * stock_h + remaining_height * stock_w - remaining_width * remaining_height
                    if remaining_area < min_remaining_area:
                        best_x, best_y = x, y
                        min_remaining_area = remaining_area

        return best_x, best_y

    def get_action(self, observation, info):
        list_prods = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
        stock_sizes = [
            (self._get_stock_size_(stock), idx) for idx, stock in enumerate(observation["stocks"])
        ]
        stock_sizes.sort(key=lambda x: (x[0][0] * x[0][1], x[0][0]), reverse=True)

        for prod in list_prods:
            if prod["quantity"] > 0:
                for (stock_size, i) in stock_sizes:
                    stock = observation["stocks"][i]
                    rotations = [prod["size"], prod["size"][::-1]]
                    for size in rotations:
                        pos_x, pos_y = self._find_placement(stock, size)
                        if pos_x is not None and pos_y is not None:
                            return {
                                "stock_idx": i,
                                "size": size,
                                "position": (pos_x, pos_y)
                            }
