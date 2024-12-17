from policy import Policy
import numpy as np
from abc import abstractmethod


class Policy2352358_2352109_2353290_2352052_2352514(Policy):
    def __init__(self, policy_id: int = 1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.strategy = LevelBasedStrategy()
        elif policy_id == 2:
            self.strategy = BestFitStrategy()

    def get_action(self, observation, info):
        return self.strategy.get_action(observation, info)
    
    
class CuttingStockStrategy:
    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)

    @abstractmethod
    def get_action(self, observation, info):
        pass


class LevelBasedStrategy(CuttingStockStrategy):
    def __init__(self):
        super().__init__()
        self.stock_manager = StockManager()

    def _get_sorted_stocks_indices(self, stocks) -> tuple[int, ...]:
        # Sort the stocks in the ascending order of the stock's height
        # Return the list of stock indices
        indices = sorted(range(len(stocks)),
                         key=lambda i: self._get_stock_size_(stocks[i])[1], reverse=True)
        return tuple(indices)

    def _get_sorted_products_indices(self, products) -> tuple[int, ...]:
        # Sort the products in the descending order of the product's height
        # Return the list of product indices
        indices = sorted(range(len(products)),
                         key=lambda i: products[i]["size"][1], reverse=True)
        return tuple(indices)

    def can_reset(self, products) -> bool:
        return sum(prod["quantity"] for prod in products) == 1

    def reset(self):
        self.stock_manager = StockManager()

    def get_action(self, observation, info):
        # Sorting the stocks
        stocks = observation["stocks"]
        sorted_stocks_indices = self._get_sorted_stocks_indices(stocks)

        # Sorting the products
        products = observation["products"]
        sorted_products_indices = self._get_sorted_products_indices(products)

        # Main Algorithm
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        i = 0
        while i < len(sorted_stocks_indices):
            idx = sorted_stocks_indices[i]
            stock = stocks[idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            if idx not in self.stock_manager.stock_indices:
                my_stock: Stock = Stock(
                    index=idx, width=stock_w, height=stock_h, levels=())
                self.stock_manager.add_stock(my_stock)
            else:
                # Get the most recent stock
                my_stock: Stock = self.stock_manager.get_most_recent_stock()
                if idx != my_stock.index:
                    i += 1
                    continue

            # Choose a product that can be placed in the stock
            pos_x, pos_y = None, None
            j = 0
            while j < len(sorted_products_indices):
                prod_idx = sorted_products_indices[j]
                prod = products[prod_idx]
                prod_size = prod["size"]
                prod_w, prod_h = int(prod_size[0]), int(prod_size[1])

                if prod["quantity"] == 0:
                    j += 1
                    continue

                # Check if the product can be placed in the stock
                if prod_w <= stock_w and prod_h <= stock_h:
                    # Check if it is a new level
                    if my_stock.is_new_level:
                        # Check if the product can be placed in the new level
                        # If yes, initialize the new level
                        if my_stock.can_place(prod_h):
                            my_stock.add_new_level(prod_h)
                            current_level = my_stock.get_current_level()
                            my_stock.is_new_level = False
                            pos_x, pos_y = 0, current_level.pos
                            current_level.place_product(prod_w)
                            break
                        # Try stock rotation
                        elif my_stock.can_place(prod_w):
                            my_stock.add_new_level(prod_w)
                            current_level = my_stock.get_current_level()
                            my_stock.is_new_level = False
                            pos_x, pos_y = 0, current_level.pos
                            current_level.place_product(prod_h)
                            prod_size = prod_size[::-1]
                            break
                        else:  # If the product cannot be placed in the new level, move to the next product
                            j += 1
                            continue

                    current_level = my_stock.get_current_level()
                    # Check if the product can be placed in the current level
                    if prod_h <= current_level.height and prod_w + current_level.length <= stock_w:
                        if prod_w > prod_h and prod_w <= current_level.height and prod_h + current_level.length <= stock_w:
                            pos_x, pos_y = current_level.length, current_level.pos
                            current_level.place_product(prod_h)
                            prod_size = prod_size[::-1]
                        else:
                            pos_x, pos_y = current_level.length, current_level.pos
                            current_level.place_product(prod_w)
                    # Try stock rotation
                    elif prod_w <= current_level.height and prod_h + current_level.length <= stock_w:
                        pos_x, pos_y = current_level.length, current_level.pos
                        current_level.place_product(prod_h)
                        prod_size = prod_size[::-1]
                    # Move to the next product if the item cannot be placed in the current level
                    else:
                        j += 1
                        continue

                # If the product is placed, break the loop
                if pos_x is not None and pos_y is not None:
                    break

                j += 1

            # If a product is placed, break the loop to return the action
            if pos_x is not None and pos_y is not None:
                stock_idx = sorted_stocks_indices[i]
                break

            # If no product can be placed in the stock, continue to process
            # If the level does not reach the stock's height, move to the next level
            if my_stock.can_add_new_level(products, sorted_products_indices):
                my_stock.is_new_level = True
            else:
                i += 1  # Move to the next stock if cannot create a new level

        if self.can_reset(products):
            self.reset()
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


class BestFitStrategy(CuttingStockStrategy):
    def __init__(self):
        super().__init__()
        self.bestfit_stock_sheets = []
        self.bestfit_used_stocks = set()
        self.bestfit_is_resetted = False

    def get_action(self, observation, info):
        items = []
        num_item = 0
        for item in observation["products"]:
            if item["quantity"] > 0:
                items.append(item["size"])
                num_item += item["quantity"]

        # print(f"{num_item=}")
        if num_item <= 1:
            self.bestfit_is_resetted = True
        items.sort(
            key=lambda p: p[0] * p[1],
            reverse=True
        )

        stocks = observation["stocks"]
        stock_sizes = [self._get_stock_size_(stock) for stock in stocks]
        stock_sizes = [(int(w), int(h)) for w, h in stock_sizes]
        stock_sizes = sorted(
            enumerate(stock_sizes),
            key=lambda s: s[1][0],
            reverse=True
        )
        # stock_sizes = enumerate(stock_sizes)

        for item_width, item_height in items:
            # Try placing the item in an existing sheet
            for sheet in self.bestfit_stock_sheets:
                if sheet.can_fit(item_width, item_height):
                    x, y, w, h = sheet.place_item(item_width, item_height)
                    ans = {
                        "stock_idx": sheet.idx,
                        "size": np.array([w, h]),
                        "position": (x, y)
                    }
                    self.bestfit_stock_sheets.sort(
                        key=lambda s: s._total_unused_space_(),
                        reverse=True
                    )
                    if (self.bestfit_is_resetted == True):
                        self.bestfit_stock_sheets = []
                        self.bestfit_used_stocks = set()
                        self.bestfit_is_resetted = False
                    return ans
                elif sheet.can_fit(item_height, item_width):
                    x, y, w, h = sheet.place_item(item_height, item_width)
                    ans = {
                        "stock_idx": sheet.idx,
                        "size": np.array([w, h]),
                        "position": (x, y)
                    }
                    self.bestfit_stock_sheets.sort(
                        key=lambda s: s._total_unused_space_(),
                        reverse=True
                    )
                    if (self.bestfit_is_resetted == True):
                        self.bestfit_stock_sheets = []
                        self.bestfit_used_stocks = set()
                        self.bestfit_is_resetted = False
                    return ans

            # If still not placed, create a new sheet
            for idx, (stock_width, stock_height) in stock_sizes:
                if (stock_width, stock_height) not in self.bestfit_used_stocks and (
                    (item_width <= stock_width and item_height <= stock_height) or (
                        item_height <= stock_width and item_width <= stock_height)
                ):
                    new_sheet = self.StockSheet(idx, stock_width, stock_height)
                    placement = new_sheet.place_item(item_width, item_height)
                    if placement is None:
                        continue
                    x, y, w, h = placement

                    self.bestfit_stock_sheets.append(new_sheet)
                    self.bestfit_used_stocks.add((stock_width, stock_height))
                    ans = {
                        "stock_idx": new_sheet.idx,
                        "size": np.array([w, h]),
                        "position": (x, y)
                    }
                    self.bestfit_stock_sheets.sort(
                        key=lambda s: s._total_unused_space_(),
                        reverse=True
                    )
                    if (self.bestfit_is_resetted == True):
                        self.bestfit_stock_sheets = []
                        self.bestfit_used_stocks = set()
                        self.bestfit_is_resetted = False
                    return ans

    class StockSheet:
        def __init__(self, idx, width, height):
            self.idx = idx
            self.width = width
            self.height = height
            # List of available spaces (x, y, w, h)
            self.remaining_spaces = [(0, 0, width, height)]

        def can_fit(self, item_width, item_height):
            """Check if an item can fit in any remaining space."""
            for space in self.remaining_spaces:
                _, _, space_width, space_height = space
                if item_width <= space_width and item_height <= space_height:
                    return True
            return False

        def place_item(self, item_width, item_height):
            """Place the item in the best-fitting space and update the remaining spaces."""
            best_space = None
            min_waste = float('inf')
            rotated = False

            for space in self.remaining_spaces:
                x, y, space_width, space_height = space
                if item_width <= space_width and item_height <= space_height:
                    waste = (space_width - item_width) * \
                        (space_height - item_height)
                    if waste < min_waste:
                        min_waste = waste
                        best_space = space

            rot_w, rot_h = item_height, item_width
            for space in self.remaining_spaces:
                x, y, space_width, space_height = space
                if rot_w <= space_width and rot_h <= space_height:
                    waste = (space_width - rot_w) * (space_height - rot_h)
                    if waste < min_waste:
                        rotated = True
                        min_waste = waste
                        best_space = space

            if best_space:
                x, y, space_width, space_height = best_space
                # Place the item
                if rotated == True:
                    item_width, item_height = rot_w, rot_h

                # Update remaining spaces
                self.remaining_spaces.remove(best_space)
                self.remaining_spaces.append(
                    (x + item_width, y, space_width - item_width, item_height))  # Right space
                self.remaining_spaces.append(
                    (x, y + item_height, space_width, space_height - item_height))  # Bottom space

                self.remaining_spaces = [
                    s for s in self.remaining_spaces if s[2] > 0 and s[3] > 0]
                return (x, y, item_width, item_height)

            return None

        def _total_unused_space_(self):
            ans = 0
            for space in self.remaining_spaces:
                ans += space[2] * space[3]
                return ans


class Level:
    def __init__(self, height, pos):
        self.height = height
        self.pos = pos
        self.length = 0

    def place_product(self, product_width: int) -> None:
        self.length += product_width


class Stock:
    def __init__(self, index: int, width: int, height: int, levels: tuple[Level, ...]):
        self.index = index
        self.width = width
        self.height = height
        self.levels: tuple[Level, ...] = levels
        self.is_new_level: bool = True
        self.is_full: bool = False

    def can_add_new_level(self, products, indices) -> bool:
        used_height = sum(level.height for level in self.levels)
        for idx in indices:
            prod = products[idx]
            prod_h = prod["size"][1]
            if prod_h + used_height <= self.height and prod["quantity"] > 0:
                return True
        return False

    def can_place(self, product_height: int) -> bool:
        if not self.levels:
            return True

        current_level = self.levels[-1]
        return current_level.pos + current_level.height + product_height <= self.height

    def add_new_level(self, height: int) -> None:
        if not self.levels:
            new_pos = 0
        else:
            new_pos = self.levels[-1].pos + self.levels[-1].height
        new_level = Level(height, new_pos)
        self.levels = (*self.levels, new_level)

    def get_current_level(self) -> Level:
        return self.levels[-1]


class StockManager:
    def __init__(self):
        self.stocks: list[Stock] = []

    def add_stock(self, stock: Stock) -> None:
        self.stocks.append(stock)

    @property
    def stock_indices(self) -> tuple[int, ...]:
        return tuple(stock.index for stock in self.stocks)

    def get_most_recent_stock(self) -> Stock:
        return self.stocks[-1]
