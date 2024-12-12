import numpy as np

from policy import Policy


class Policy1(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info):
        # observation['stocks']: tuple[np.ndarray]
        # observation['products']: tuple[dict[np.ndarray, int]]
        stocks = observation['stocks']
        products = observation['products']

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

                    for x in range(stock_matrix.shape[0] - product_size[0] + 1):
                        for y in range(stock_matrix.shape[1] - product_size[1] + 1):
                            candidate_area = stock_matrix[
                                             x:x + product_size[0],
                                             y:y + product_size[1],
                                             ]

                            if np.all(candidate_area == -1):
                                waste = stock_matrix.size - product_size[0] * product_size[1]
                                if best_fit is None or waste < best_fit['waste']:
                                    best_fit = {
                                        'x': x,
                                        'y': y,
                                        'waste': waste,
                                    }

                    if best_fit:
                        return {
                            'stock_idx': stock_idx,
                            'size': product_size,
                            'position': (best_fit['x'], best_fit['y']),
                        }

        return None


class Policy2(Policy):
    def __init__(self):
        super().__init__()
        self.pattern_collection = {}
        self.utilized_patterns = set()
        self.active_stock = 0
        self.occupied_positions = {}
        self.initiate_game = True

    def get_action(self, observation, info):
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


class Policy2250013(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy = Policy1() if policy_id == 1 else Policy2()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
