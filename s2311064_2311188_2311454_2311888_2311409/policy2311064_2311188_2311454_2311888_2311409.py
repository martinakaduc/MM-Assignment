from policy import Policy
import numpy as np

class Policy2311064_2311188_2311454_2311888_2311409(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = Heuristic()
        elif policy_id == 2:
            self.policy = brandandbound()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)


class Heuristic(Policy):
    def __init__(self, policy_id=1):

        # Danh sách chiến lược đặt sản phẩm
        self.placement_strategies = [
            self._place_with_density_optimization,
            self._place_with_compactness_scoring,
            self._place_with_minimal_fragmentation
        ]
        
    def get_action(self, observation, info):
        """Chọn vị trí đặt sản phẩm trong kho."""
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Sắp xếp kho theo mức độ trống
                sorted_stocks = sorted(
                    enumerate(observation["stocks"]),
                    key=lambda x: self._calculate_stock_occupancy(x[1])
                )

                for stock_idx, stock in sorted_stocks:
                    for strategy in self.placement_strategies:
                        result = strategy(stock, stock_idx, prod_size, observation)
                        if result:
                            return result  # Dừng lại ngay khi tìm thấy vị trí phù hợp

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _calculate_stock_occupancy(self, stock):
        """Tính phần trăm diện tích bị chiếm dụng của kho."""
        total_space = stock.size
        occupied_space = np.count_nonzero(stock != -1)
        return 1 - (occupied_space / total_space)

    def find_best_placement(self, stock, stock_idx, prod_size, score_function):
        """Hàm chung để tìm vị trí tốt nhất dựa trên hàm tính điểm."""
        best_score = float('-inf')
        best_position = None

        for x in range(stock.shape[0] - prod_size[0] + 1):
            for y in range(stock.shape[1] - prod_size[1] + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    score = score_function(stock, (x, y), prod_size)
                    if score > best_score:
                        best_score = score
                        best_position = (x, y)

        return {"stock_idx": stock_idx, "size": prod_size, "position": best_position} if best_position else None

    def _place_with_density_optimization(self, stock, stock_idx, prod_size, observation):
        """Chiến lược đặt sản phẩm tối ưu hóa mật độ."""
        return self.find_best_placement(
            stock, stock_idx, prod_size, 
            lambda s, p, ps: self._calculate_proximity_score(s, p, ps) + self._calculate_edge_score(s.shape[0], s.shape[1], p[0], p[1], ps[0], ps[1])
        )

    def _place_with_compactness_scoring(self, stock, stock_idx, prod_size, observation):
        """Chiến lược đặt sản phẩm tập trung vào độ gọn."""
        return self.find_best_placement(
            stock, stock_idx, prod_size, 
            lambda s, p, ps: self._calculate_fragmentation(s, p, ps)
        )

    def _place_with_minimal_fragmentation(self, stock, stock_idx, prod_size, observation):
        """Chiến lược đặt sản phẩm để giảm thiểu phân mảnh."""
        return self.find_best_placement(
            stock, stock_idx, prod_size, 
            lambda s, p, ps: -self._calculate_fragmentation(s, p, ps)
        )

    def _calculate_proximity_score(self, stock, position, prod_size):
        """Tính điểm gần các sản phẩm đã đặt."""
        x, y = position
        prod_w, prod_h = prod_size
        neighbors = self.get_neighbors(stock, position, prod_size, pad=1)
        return -np.count_nonzero(neighbors != -1)

    def _calculate_edge_score(self, stock_w, stock_h, x, y, prod_w, prod_h):
        """Tính điểm gần các cạnh."""
        edge_score = min(x, y) + min(stock_w - (x + prod_w), stock_h - (y + prod_h))
        return edge_score

    def _calculate_fragmentation(self, stock, position, prod_size):
        """Tính phân mảnh gây ra bởi việc đặt sản phẩm."""
        x, y = position
        prod_w, prod_h = prod_size

        temp_stock = stock.copy()
        temp_stock[x:x+prod_w, y:y+prod_h] = 0

        return self._count_isolated_spaces(temp_stock)

    def _count_isolated_spaces(self, stock):
        """Đếm số lượng ô trống bị phân mảnh trong kho."""
        visited = np.zeros_like(stock, dtype=bool)
        isolated_count = 0

        def dfs(x, y):
            if x < 0 or x >= stock.shape[0] or y < 0 or y >= stock.shape[1] or visited[x, y] or stock[x, y] != -1:
                return 0
            visited[x, y] = True
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(x + dx, y + dy)
            return 1

        for x in range(stock.shape[0]):
            for y in range(stock.shape[1]):
                if stock[x, y] == -1 and not visited[x, y]:
                    isolated_count += dfs(x, y)

        return isolated_count

    def get_neighbors(self, stock, position, prod_size, pad=1):
        """Lấy ô lân cận xung quanh vị trí sản phẩm."""
        x, y = position
        prod_w, prod_h = prod_size
        neighbors = stock[max(0, x - pad):x + prod_w + pad, max(0, y - pad):y + prod_h + pad]
        return neighbors

    def _can_place_(self, stock, position, prod_size):
        """Kiểm tra xem có thể đặt sản phẩm tại vị trí không."""
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        for x in range(pos_x, pos_x + prod_w):
            for y in range(pos_y, pos_y + prod_h):
                if x >= stock.shape[0] or y >= stock.shape[1] or stock[x, y] != -1:
                    return False  # Thoát sớm
        return True

class brandandbound(Policy):
    def __init__(self, policy_id=1):
        super().__init__()

        self.best_choice = None
        self.prd_cnt = 1  # Khởi tạo bộ đếm cho ID sản phẩm

    def filter_and_sort_products(self, products):
        #lọc và sắp xếp sản phẩm cần cắt
        return sorted(
            (prod for prod in products if prod["quantity"] > 0), 
            key=lambda prod: prod["size"][0] * prod["size"][1], 
            reverse=True
    )

    def get_action(self, observation, info):
        # Lọc và sắp xếp sản phẩm cần cắt
        products = self.filter_and_sort_products(observation["products"])

        # Đặt lại trạng thái tìm kiếm ban đầu
        self.pst_best = None
        self.wstmin = float('inf')

        # Lặp qua tất cả các tấm nguyên liệu có sẵn
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w == 0 or stock_h == 0:
                continue

            # Khởi động thuật toán Branch and Bound cho từng tấm nguyên liệu
            self.b_and_b(stock, products, 0, stock_idx=i)

        # Trả về thông tin của vị trí và kích thước sản phẩm tối ưu đã tìm được
        return self.pst_best or {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


    def b_and_b(self, stock, products, wa_crt, stock_idx):
        # Điều kiện cắt tỉa: nếu lãng phí hiện tại lớn hơn lãng phí nhỏ nhất đã biết
        if wa_crt >= self.wstmin:
            return

        # Biến trạng thái đã duyệt
        stock = np.array(stock)
        state_vst = set()

        # Tạo key cho trạng thái hiện tại
        st_key = self._generate_state_key(stock, products)
        if st_key in state_vst:
            return
        state_vst.add(st_key)

        # Kiểm tra nếu tất cả sản phẩm đã được đặt
        if all(prod["quantity"] == 0 for prod in products):
            if wa_crt < self.wstmin:
                self.best_choice = np.copy(stock)
                self.wstmin = wa_crt
            return

        # Lặp qua từng sản phẩm
        for idx, prod in enumerate(products):
            if prod["quantity"] > 0:
                self._process_product_for_stock(stock, prod, products, wa_crt, stock_idx)

    def _generate_state_key(self, stock, products):
        """Tạo key trạng thái từ stock và danh sách sản phẩm."""
        return (tuple(stock.flatten()), tuple(prod["quantity"] for prod in products))

    def _process_product_for_stock(self, stock, prod, products, wa_crt, stock_idx):
        """Xử lý một sản phẩm cho tấm nguyên liệu."""
        prod_w, prod_h = prod["size"]
        stock_w, stock_h = self._get_stock_size_(stock)

        # Kiểm tra xem sản phẩm có thể đặt vừa tấm nguyên liệu hay không
        if stock_w < prod_w or stock_h < prod_h:
            return

        # Duyệt các vị trí khả thi để đặt sản phẩm
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod["size"]):
                    self._place_product(stock, prod, (x, y), products, wa_crt, stock_idx)

    def _place_product(self, stock, prod, position, products, wa_crt, stock_idx):
        """Đặt sản phẩm vào tấm nguyên liệu và gọi đệ quy."""
        x, y = position
        prod_w, prod_h = prod["size"]

        # Thực hiện thay đổi trực tiếp trên `stock`
        stock[x:x + prod_w, y:y + prod_h] = 1  # Đặt sản phẩm

        # Giảm số lượng sản phẩm và tính lãng phí
        prod["quantity"] -= 1
        new_waste = np.sum(stock == 0)  # Tính số khoảng trống

        # Cập nhật thông tin của giải pháp tốt nhất
        if new_waste < self.wstmin:
            self.pst_best = {
                "stock_idx": stock_idx,
                "size": prod["size"],
                "position": (x, y)
            }
            self.wstmin = new_waste

        # Gọi đệ quy
        self.b_and_b(stock, products, new_waste, stock_idx)

        # Quay lui (backtrack)
        stock[x:x + prod_w, y:y + prod_h] = 0  # Hoàn tác
        prod["quantity"] += 1

