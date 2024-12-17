from policy import Policy
import numpy as np

class Policy2310596_2310745_2313406_2313739_2211891_(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._apply_ffda_policy(observation, info)
        elif self.policy_id == 2:
            return self._apply_greedy_policy(observation, info)

    # FFDA Policy
    def _apply_ffda_policy(self, observation, info):
        list_prods = sorted(
            observation["products"],
            # key=lambda prod: max(prod["size"]), # Sắp xếp theo chiều cao giảm dần
            key=lambda prod: prod["size"][0] * prod["size"][1], # Sắp xếp theo diện tích giảm dần
            reverse=True,
        )

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                placed = False

                # Duyệt qua từng tấm lớn
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Tìm vị trí khả dụng đầu tiên (không xoay)
                    position = self._find_first_fit(stock, prod_size)
                    if position:
                        placed = True
                        pos_x, pos_y = position
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

                    # Thử xoay sản phẩm
                    position = self._find_first_fit(stock, prod_size[::-1])
                    if position:
                        placed = True
                        pos_x, pos_y = position
                        prod_size = prod_size[::-1]
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

                # Nếu không đặt được vào tấm nào, thêm một tấm mới
                if not placed:
                    observation["stocks"].append(self._create_new_stock())
                    stock_idx = len(observation["stocks"]) - 1
                    pos_x, pos_y = 0, 0
                    return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _find_first_fit(self, stock, prod_size):
        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None

    # Greedy First Fit Decrease Policy
    def _apply_greedy_policy(self, observation, info):
        
        stocks = observation['stocks']
        products = observation['products']

        # Chuyển đổi products thành danh sách và sắp xếp theo thứ tự kích thước giảm dần
        products = sorted(products, key=lambda x: x['size'][0]*x['size'][1], reverse=True)

        for stock_idx, stock in enumerate(stocks):
            for product in products:
                # Tìm vị trí cắt tối ưu
                pos_x, pos_y = self.find_optimal_position(stock, product['size'])
                if pos_x != -1 and pos_y != -1:
                    self.cut_product(stock, product['size'], pos_x, pos_y, product)
                    product['quantity'] -= 1
                    return {"stock_idx": stock_idx, "size": product['size'], "position": (pos_x, pos_y)}
        return None

    def _create_new_stock(self):
        # Tạo tấm lớn mới với kích thước tối đa
        return np.full((10, 10), -1)  # 10x10 là ví dụ kích thước tấm lớn
    
    def find_optimal_position(self, stock: np.ndarray, product_size: tuple) -> tuple:
        
        max_width, max_height = stock.shape
        for x in range(max_width - product_size[0] + 1):
            for y in range(max_height - product_size[1] + 1):
                if self._can_place_(stock, (x, y), product_size):   #can_cut_product(stock, product_size, x, y):
                    return x, y
        return -1, -1
    
    def cut_product(self, stock: np.ndarray, product_size: tuple, pos_x: int, pos_y: int, product: dict) -> None:
        
        pw, ph = product_size
        product_id = product.get('id', -1)
        stock[pos_x:pos_x+pw, pos_y:pos_y+ph] = product_id
    
    