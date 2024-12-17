import numpy as np
from policy import Policy


class Policy2353205_2352674_2352035_2353040_2352646__Group54(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = Policy2352674(policy_id)
        elif policy_id == 2:
            self.policy = Policy2353205(policy_id)
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class Policy2352674(Policy):
    def __init__(self, policy_id = 1):
        super().__init__()
        self.policy_id = policy_id
        self.current_stock_index = 0
        self.sorted_stocks = []
        self.remaining_products = []
        self.total_product_area = 0
        self.remaining_area = 0

    def reset(self, observation = None):
        self.current_stock_index = 0
        self.sorted_stocks = []
        self.remaining_products = []
        self.total_product_area = 0
        self.remaining_area = 0

        if observation:
            self.total_product_area = sum(
                np.prod(product["size"]) * product["quantity"] for product in observation["products"]
            )
            self.remaining_area = self.total_product_area

            self.sorted_stocks = sorted(
                enumerate(observation["stocks"]),
                key=lambda x: self._calculate_stock_area(x[1]),
                reverse=True,
            )
            self.remaining_products = sorted(
                observation["products"],
                key=lambda x: np.prod(x["size"]),
                reverse=True,
            )

    def _calculate_stock_area(self, stock):
        stock_width, stock_height = self._get_stock_size_(stock)
        return stock_width * stock_height

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock_h, stock_w = stock.shape

        if pos_x + prod_w > stock_h or pos_y + prod_h > stock_w:
            return False

        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)

    def _greedy_place(self):
        """
        Greedy approach to find an initial solution by choosing the largest product first.
        """
        for stock_idx, stock in self.sorted_stocks[self.current_stock_index:]:
            stock_width, stock_height = self._get_stock_size_(stock)

            for product in self.remaining_products:
                if product["quantity"] == 0:
                    continue

                product_width, product_height = product["size"]

                for orientation in [(product_width, product_height), (product_height, product_width)]:
                    rotated_width, rotated_height = orientation

                    if stock_width >= rotated_width and stock_height >= rotated_height:
                        for x in range(stock_width - rotated_width + 1):
                            for y in range(stock_height - rotated_height + 1):
                                if self._can_place_(stock, (x, y), (rotated_width, rotated_height)):
                                    product["quantity"] -= 1
                                    self.remaining_area -= np.prod((rotated_width, rotated_height))
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": [rotated_width, rotated_height],
                                        "position": (x, y),
                                    }
        return None

    def _branch_and_bound(self):
        """
        Apply Branch and Bound to improve the solution.
        """
        best_solution = None
        best_remaining_area = self.remaining_area

        for stock_idx, stock in self.sorted_stocks:
            stock_width, stock_height = self._get_stock_size_(stock)

            for product in self.remaining_products:
                if product["quantity"] == 0:
                    continue

                product_width, product_height = product["size"]

                for orientation in [(product_width, product_height), (product_height, product_width)]:
                    rotated_width, rotated_height = orientation

                    if stock_width >= rotated_width and stock_height >= rotated_height:
                        for x in range(stock_width - rotated_width + 1):
                            for y in range(stock_height - rotated_height + 1):
                                if self._can_place_(stock, (x, y), (rotated_width, rotated_height)):
                                    remaining_area_after = (
                                        stock_width * stock_height
                                        - (rotated_width * rotated_height)
                                    )
                                    if remaining_area_after < best_remaining_area:
                                        best_solution = {
                                            "stock_idx": stock_idx,
                                            "size": [rotated_width, rotated_height],
                                            "position": (x, y),
                                        }
                                        best_remaining_area = remaining_area_after
        return best_solution

    def get_action(self, observation, info):
        if info["filled_ratio"] == 0:
            self.reset(observation)

        # Try greedy placement first
        action = self._greedy_place()
        if action:
            return action

        # Use Branch and Bound if Greedy fails
        action = self._branch_and_bound()
        if action:
            return action

        # No valid placement found
        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
    

class Policy2353205(Policy):
    def __init__(self, policy_id = 2):
        super().__init__()
        self.policy_id = policy_id

    def get_action(self, observation, info):
        stocks = observation['stocks']
        products = observation['products']
        stock_sizes = []

        # Tính toán diện tích của từng stock
        for j, stock in enumerate(stocks):
            length, width = self.calculate_valid_area(stock)
            area = length * width
            stock_sizes.append((j, area))

        # Sắp xếp stock theo diện tích từ lớn đến nhỏ nhưng giữ chỉ mục
        sorted_stocks = sorted(stock_sizes, key=lambda x: x[1], reverse=True)
        stock_idx_sort = [stock[0] for stock in sorted_stocks]
        
        # Sắp xếp các sản phẩm theo diện tích từ lớn đến nhỏ
        products_sorted = sorted(products, key=lambda x: x['size'][0] * x['size'][1], reverse=True)
        
        # Duyệt qua các stock và sản phẩm để thực hiện hành động
        for j in stock_idx_sort:
            stock = stocks[j]
            for product in products_sorted:
                if product['quantity'] > 0:
                    w, h = product['size']
                    # Tìm các vị trí hợp lệ để cắt sản phẩm
                    for x in range(stock.shape[1]): 
                        for y in range(stock.shape[0]):
                            if (stock[x, y] != -1):
                                continue
                            if self.can_place(stock, (w, h), (x, y)):
                                return {
                                    "stock_idx": j,
                                    "size": (w, h),
                                    "position": (x, y)
                                }
                            if self.can_place(stock,(h, w), (x, y)):
                                return {
                                    "stock_idx": j,
                                    "size": (h, w),
                                    "position": (x, y)
                                }
        return None
                    
    def calculate_valid_area(self, stock):
        valid_positions = np.where(stock == -1)

        if valid_positions[0].size == 0 or valid_positions[1].size == 0:
            return 0, 0

        length = valid_positions[0].max() - valid_positions[0].min() + 1
        width = valid_positions[1].max() - valid_positions[1].min() + 1

        return length, width

    def can_place(self, stock, product_size, position):
        w, h = product_size
        x, y = position
        if x + w > stock.shape[0] or y + h > stock.shape[1]:
            return False
        # Kiểm tra các ô trong kho chưa bị chiếm dụng (giá trị == -1)
        if np.any(stock[x:x + w, y:y + h] != -1):
            return False
        return True

