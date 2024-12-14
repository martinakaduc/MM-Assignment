

import numpy as np
from abc import abstractmethod
from typing import List, Dict, Tuple
from scipy import ndimage

class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        """
        Lấy chiều rộng và chiều cao sử dụng được của vật liệu.
        Giả sử -2 đại diện cho các vùng không sử dụng được.
        """
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        """
        Kiểm tra xem một sản phẩm có thể được đặt tại vị trí cụ thể hay không.
        Giả sử -1 đại diện cho các ô trống.
        """
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        if (pos_x + prod_w > stock.shape[0] or 
            pos_y + prod_h > stock.shape[1]):
            return False

        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)


class Policy2210xxx(Policy):
    def __init__(self, 
                 waste_penalty: float = 0.1, 
                 rotation_enabled: bool = True,
                 fragmentation_weight: float = 0.3, 
                 policy_id: int = 1):
        """
        Khởi tạo chính sách cắt dải nâng cao.
        
        Args:
            waste_penalty: Hệ số phạt cho không gian vật liệu không sử dụng
            rotation_enabled: Cho phép xoay 90 độ các sản phẩm
            fragmentation_weight: Trọng số của sự phân mảnh trong tính toán lãng phí
            policy_id: Xác định thuật toán hoặc phương pháp heuristic nào được sử dụng
        """
        self.current_stock_index = 0
        self.waste_penalty = waste_penalty
        self.rotation_enabled = rotation_enabled
        self.fragmentation_weight = fragmentation_weight
        self.policy_id = policy_id
        self.stock_placements = {}

    def _get_usable_stock_dimensions(self, stock: np.ndarray) -> Tuple[int, int]:
        """
        Lấy chiều rộng và chiều cao sử dụng được của vật liệu.
        Giả sử -2 đại diện cho các vùng không sử dụng được.
        """
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_product(self, stock: np.ndarray, position: Tuple[int, int], product_size: Tuple[int, int]) -> bool:
        """
        Kiểm tra xem một sản phẩm có thể được đặt tại vị trí cụ thể hay không.
        Giả sử -1 đại diện cho các ô trống.
        """
        pos_x, pos_y = position
        prod_w, prod_h = product_size

        # Kiểm tra xem sản phẩm có nằm trong giới hạn vật liệu hay không
        if (pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]):
            return False

        # Kiểm tra xem tất cả các ô mục tiêu có trống không
        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)
    def _calculate_wasting(self, stock: np.ndarray, placed_product_size: Tuple[int, int]) -> float:
        """
        Tính điểm lãng phí nâng cao bằng cách xem xét nhiều yếu tố.
        """
        # Tổng diện tích sử dụng được
        sum_usable_area = np.sum(stock == -1)
        
        # Diện tích của sản phẩm đã đặt
        placed_area = np.prod(placed_product_size)
        
        # Diện tích trống còn lại
        empty_area = sum_usable_area - placed_area
        
        # Tính điểm phân mảnh
        frag_scores = self.frag_calculating(stock)
        
        # Tính tỷ lệ lãng phí với nhiều yếu tố
        waste_ratio = max(0, empty_area / sum_usable_area)
        
        # Kết hợp tỷ lệ lãng phí với sự phân mảnh
        combined_waste = (
            waste_ratio * (1 - self.fragmentation_weight) + 
            frag_scores * self.fragmentation_weight
        )
        
        return combined_waste * (1 + self.waste_penalty)

    def frag_calculating(self, stock: np.ndarray) -> float:
        """
        Đánh giá sự phân mảnh của các không gian trống.
        """
        # Xác định các vùng trống
        bin_st = (stock == -1).astype(int)
        labeled_array, num_regions = ndimage.label(bin_st)
        
        # Nếu không có vùng trống nào, trả về 0
        if num_regions == 0:
            return 0
        
        # Tính kích thước trung bình của các vùng trống
        empty_sizes = np.bincount(labeled_array.ravel())[1:]
        avg_size = np.mean(empty_sizes)
        
        # Điểm phân mảnh: số vùng càng nhiều thì mức độ phân mảnh càng cao
        return num_regions / (avg_size + 1)
    def find_best_place(self, stock: np.ndarray, observation) -> Dict:
        """
        Tìm vị trí hiệu quả nhất để đặt sản phẩm trong một vật liệu.
        
        Args:
            stock: Biểu diễn lưới của vật liệu
            products: Danh sách các sản phẩm cần đặt
        
        Returns:
            Dictionary chứa vị trí tốt nhất hoặc None
        """
        stock_w, stock_h = self._get_usable_stock_dimensions(stock)
        best_place = None
        min_waste = float('inf')

        
        products = observation["products"]
        for product in products:
            if(product['quantity'] == 0):
                continue
            size = product["size"]
            
            prod_w, prod_h = size

            if stock_w < prod_w or stock_h < prod_h:
                continue

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_product(stock, (x, y), size):
                        temp_stock = stock.copy()
                        temp_stock[x:x+prod_w, y:y+prod_h] = 1
                        
                        waste_score = self._calculate_wasting(temp_stock, size)
                            
                        if waste_score < min_waste:
                            min_waste = waste_score
                            best_place = {
                                "product": product,
                                "size": size,
                                "position": (x, y)
                            }
                        
                
            if stock_w < prod_h or stock_h < prod_w:
                continue
            for x in range(stock_w - prod_h + 1):
                for y in range(stock_h - prod_w + 1):
                    if self._can_place_product(stock, (x, y), size[::-1]):

                        size = size[::-1]
                        
                        temp_stock_ = stock.copy()
                        temp_stock_[x:x+prod_h, y:y+prod_w] = 1
                            
                        waste_score = self._calculate_wasting(temp_stock_, size[::-1])
                            
                        if waste_score < min_waste:
                            min_waste = waste_score
                            product["size"] = size
                            best_place = {
                                "product": product,
                                "size": size,
                                "position": (x, y)
                           }
        
        print(best_place)
        return best_place

    def is_stock_fully_used(self, stock: np.ndarray, placed_products: List[Dict]) -> bool:
        """
        Kiểm tra xem vật liệu có được sử dụng đầy đủ hay không.
        """
        remain_area = np.sum(stock == -1)
        max_size = max(
            [np.prod(prod['size']) for prod in placed_products], 
            default=0
        )
        return remain_area < max_size * 1.5

    def get_action(self, observation: Dict, info: Dict) -> Dict:
        """
        Xác định chiến lược đặt sản phẩm.
        """
        if self.policy_id == 1:
            return self._policy_1(observation)
        elif self.policy_id == 2:
            return self._policy_2(observation)
        else:
            # Mặc định: không thực hiện đặt
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _policy_1(self, observation: Dict) -> Dict:
        """
        Policy 1: Best-Fit
        """
        current_stock = observation["stocks"][self.current_stock_index]
        
        # Khởi tạo theo dõi sản phẩm cho vật liệu hiện tại
        if self.current_stock_index not in self.stock_placements:
            self.stock_placements[self.current_stock_index] = []

        # Tìm vị trí đặt tốt nhất cho vật liệu hiện tại
        placement = self.find_best_place(current_stock, observation)

        if placement:
            return {
                "stock_idx": self.current_stock_index,
                "size": placement['size'],
                "position": placement['position']
            }
        else:
            self.current_stock_index += 1

        # Nếu không còn khả năng đặt trong vật liệu hiện tại
        if self.current_stock_index >= len(observation["stocks"]):
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _policy_2(self, observation: Dict) -> Dict:
        """
        Policy 2: First-Fit Decreasing
        """
        list_of_products = observation["products"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        prods_sorted = sorted(
            (prod for prod in list_of_products if prod["quantity"] > 0),
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
        )

        for stock_idx, stock in sorted(
            enumerate(observation["stocks"]), key=lambda s: np.sum(s[1] != -2), reverse=True
        ):
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in prods_sorted:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (pos_x, pos_y)
                                    }

        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
