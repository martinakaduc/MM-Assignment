from policy import Policy
import numpy as np
class Policy2313280_2311382_2310143_2313140(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = Column_Generations()
        elif policy_id == 2:
            self.policy = First_Fit()
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
class Column_Generations(Policy):
    def __init__(self):
       pass
    def _generate_columns(self, observation):
        columns = []
        sorted_products = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            
            for prod in sorted_products:  
                prod_w, prod_h = prod["size"]
                if prod["quantity"] > 0:
                    pos_x, pos_y = self._find_valid_position(stock, prod, rotated=False)
                    if pos_x is None and pos_y is None: 
                        pos_x, pos_y = self._find_valid_position(stock, prod, rotated=True)
                        if pos_x is not None and pos_y is not None:
                            prod["size"] = (prod_h, prod_w)  
                            columns.append((prod, stock, pos_x, pos_y))
                    elif pos_x is not None and pos_y is not None:
                        columns.append((prod, stock, pos_x, pos_y))
        return columns

    def _find_valid_position(self, stock, prod, rotated=False):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod["size"]
        if rotated:
            prod_w, prod_h = prod_h, prod_w
        
        if prod_w * prod_h > stock_h * stock_w:
            return None, None

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    return x, y

        return None, None

    def get_action(self, observation, info):
        self.cut_patterns = self._generate_columns(observation)

        best_pattern = None
        for pattern in self.cut_patterns:
            prod, stock, pos_x, pos_y = pattern
            if best_pattern is None:
                best_pattern = pattern
            else:
                pass

        if best_pattern is not None:
            prod, stock, pos_x, pos_y = best_pattern
            stock_idx = -1
            for idx, s in enumerate(observation["stocks"]):
                if np.array_equal(self._get_stock_size_(s), self._get_stock_size_(stock)):
                    stock_idx = idx
                    break
            return {"stock_idx": stock_idx, "size": prod["size"], "position": (pos_x, pos_y)}
class First_Fit(Policy):
    def __init__(self):
        self.reset()

    def reset(self):
        self.storage_units = []  
        self.current_product_index = 0  
        self.placed_products = [] 

    def get_action(self, observation, info):
        if not self.placed_products:
            self.storage_units = [
                self.Storage(self._get_stock_size_(stock)[0], self._get_stock_size_(stock)[1], idx + 1)
                for idx, stock in enumerate(observation["stocks"])
            ]

            products_to_place = [
                self.Item(prod["size"][0], prod["size"][1])
                for prod in observation["products"]
                for _ in range(prod["quantity"])
            ]

            # Sắp xếp sản phẩm theo diện tích giảm dần
            for item in sorted(products_to_place, key=lambda p: p.width * p.height, reverse=True):
                placed = False
                for storage in self.storage_units:
                    if storage._can_place_(item):
                        placed = True
                        break

                if not placed:
                    new_storage = self.Storage(self.storage_units[0].width, self.storage_units[0].height, len(self.storage_units) + 1)
                    new_storage._can_place_(item)
                    self.storage_units.append(new_storage)

            # Lưu thông tin vị trí sản phẩm
            self.placed_products = [
                (storage.id - 1, x, y, w, h)
                for storage in self.storage_units
                for (x, y, w, h) in storage.placed_items
            ]

        if self.current_product_index >= len(self.placed_products):
            self.reset()
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        stock_idx, pos_x, pos_y, w, h = self.placed_products[self.current_product_index]
        self.current_product_index += 1
        return {"stock_idx": stock_idx, "size": [w, h], "position": (pos_x, pos_y)}

    class Storage:
        def __init__(self, width, height, storage_id):
            self.width = width
            self.height = height
            self.id = storage_id
            self.available_spaces = [(0, 0, width, height)]  # Không gian trống
            self.placed_items = []  # Sản phẩm đã đặt

        def _can_place_(self, item):
            self.available_spaces.sort(key=lambda s: s[2] * s[3], reverse=True)

            for i, (x, y, w, h) in enumerate(self.available_spaces):
                for item_width, item_height in [(item.width, item.height), (item.height, item.width)]:
                    if item_width <= w and item_height <= h:
                        self.available_spaces.pop(i)

                        new_spaces = []
                        if item_width < w:
                            new_spaces.append((x + item_width, y, w - item_width, item_height))
                        if item_height < h:
                            new_spaces.append((x, y + item_height, w, h - item_height))

                        self.available_spaces.extend(new_spaces)
                        self.available_spaces = self._combine_adjacent_spaces(self.available_spaces)
                        self.placed_items.append((x, y, item_width, item_height))
                        return True
            return False

        def _combine_adjacent_spaces(self, spaces):
            spaces.sort()
            merged = []
            for space in spaces:
                if merged and self._can_merge(merged[-1], space):
                    merged[-1] = self._merge_spaces(merged[-1], space)
                else:
                    merged.append(space)
            return merged

        def _can_merge(self, space1, space2):
            x1, y1, w1, h1 = space1
            x2, y2, w2, h2 = space2
            return (x1 == x2 and y1 + h1 == y2) or (y1 == y2 and x1 + w1 == x2)

        def _merge_spaces(self, space1, space2):
            x1, y1, w1, h1 = space1
            x2, y2, w2, h2 = space2
            return (x1, y1, w1 + w2, h1) if y1 == y2 else (x1, y1, w1, h1 + h2)

    class Item:
        def __init__(self, width, height):
            self.width = width
            self.height = height
