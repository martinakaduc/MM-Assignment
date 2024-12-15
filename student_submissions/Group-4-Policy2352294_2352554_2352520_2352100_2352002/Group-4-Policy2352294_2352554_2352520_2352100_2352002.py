from policy import Policy
import numpy as np

class Policy2352294_2352554_2352520_2352100_2352002(Policy):
    def __init__(self,  policy_id=1):
        # Kiểm tra ID chính sách hợp lệ
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        
        self.policy_id = policy_id
        self.patterns = []  # Danh sách các patterns hiện tại
        self.branch_stack = []  # Ngăn xếp để lưu trữ thông tin nhánh
        self.best_solution = None  # Lưu trữ giải pháp tốt nhất
        self.best_solution_score = None  # Lưu trữ điểm số của giải pháp tốt nhất

    def get_action(self, observation, info):
        # Lấy thông tin sản phẩm và kho từ observation
        products = observation["products"]
        stocks = observation["stocks"]
        
        # Chọn logic chính sách dựa trên policy_id
        if self.policy_id == 1:
            return self._policy_1_logic(products, stocks)  # Sử dụng logic cho chính sách 1
        elif self.policy_id == 2:
            return self._policy_2_logic(products, stocks)  # Sử dụng logic cho chính sách 2

    def _policy_1_logic(self, products, stocks):
        """Logic cho chính sách 1: Sử dụng cả column generation và branch and bound."""
        # Khởi tạo patterns nếu chưa có
        if not self.patterns:
            self._initialize_patterns(stocks)

        # Sắp xếp sản phẩm theo diện tích giảm dần và tỷ lệ kích thước
        products = sorted(products, 
                        key=lambda x: (x["size"][0] * x["size"][1], 
                                     max(x["size"][0]/x["size"][1], x["size"][1]/x["size"][0])), 
                        reverse=True)

        for prod in products:
            if prod["quantity"] > 0:
                best_action, best_waste = self._find_best_action(prod, stocks)
                if best_action["stock_idx"] == -1:
                    branch = self._branch()  # Tạo nhánh mới nếu không tìm thấy hành động tốt nhất
                    if branch["stock_idx"] != -1:
                        branch_action = self._create_branch(branch)
                        branch_waste = self._calculate_waste(branch_action, stocks, prod["size"])
                        if branch_waste < best_waste:
                            best_action = branch_action
                            self.patterns.append(branch_action)  # Thêm hành động nhánh vào patterns

        # Nếu không tìm thấy vị trí hợp lệ, tạo một pattern mới
        if best_action["stock_idx"] != -1:
            return best_action
        return self._generate_new_pattern(products, stocks)  # Tạo pattern mới nếu không tìm thấy

    def _find_best_action(self, prod, stocks):
        """Tìm hành động tốt nhất cho sản phẩm."""
        original_size = prod["size"]
        rotated_size = [original_size[1], original_size[0]]  # Kích thước xoay của sản phẩm
        
        best_action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        best_waste = float("inf")  # Khởi tạo waste tốt nhất
        
        for size in [original_size, rotated_size]:
            prod["size"] = size
            action = self._solve_master_problem(prod, stocks)  # Giải bài toán mẹ để tìm hành động
            if action["stock_idx"] != -1:
                waste = self._calculate_waste(action, stocks, size)  # Tính waste cho hành động
                if waste < best_waste:
                    pos_bonus = 1 if action["position"][0] == 0 or action["position"][1] == 0 else 0
                    adjusted_waste = waste * (1 - 0.1 * pos_bonus)  # Điều chỉnh waste dựa trên vị trí
                    if adjusted_waste < best_waste:
                        best_waste = adjusted_waste
                        best_action = action

        # Khôi phục kích thước gốc
        prod["size"] = original_size
        return best_action, best_waste

    def _policy_2_logic(self, products, stocks):
        """Logic cho chính sách 2: Chỉ sử dụng column generation."""
        # Khởi tạo patterns nếu chưa có
        if not self.patterns:
            self._initialize_patterns2(stocks)

        best_action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        best_waste = float("inf")  # Khởi tạo waste tốt nhất

        # Duyệt qua từng sản phẩm để tìm hành động tốt nhất
        for prod in products:
            if prod["quantity"] > 0:
                # Giải bài toán mẹ cho chính sách 2 để tìm hành động
                action = self._solve_master_problem2(prod, stocks)
                if action["stock_idx"] != -1:
                    stock = stocks[action["stock_idx"]]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    # Tính waste bằng cách lấy diện tích kho trừ đi diện tích sản phẩm
                    waste = (stock_w * stock_h) - (prod["size"][0] * prod["size"][1])

                    # Cập nhật hành động tốt nhất nếu waste thấp hơn
                    if waste < best_waste:
                        best_waste = waste
                        best_action = action

        # Nếu không tìm thấy hành động tốt nhất, tạo một pattern mới
        if best_action["stock_idx"] == -1:
            new_action = self._generate_new_pattern2(products, stocks)
            return new_action
        return best_action

    def _initialize_patterns(self, stocks):
        """Khởi tạo các patterns cơ bản dựa trên kho."""
        for stock_idx, stock in enumerate(stocks):
            self.patterns.append({
                "stock_idx": stock_idx,
                "pos": [],  # Chưa đặt sản phẩm nào
            })

    def _solve_master_problem(self, prod, stocks):
        """Giải bài toán mẹ bằng cách kiểm tra các patterns hiện tại."""
        prod_size = prod["size"]
        prod_w, prod_h = prod_size

        # Duyệt qua các patterns hiện tại để tìm vị trí phù hợp cho sản phẩm
        for pattern in self.patterns:
            stock_idx = pattern["stock_idx"]
            stock = stocks[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            # Kiểm tra xem sản phẩm có thể được đặt vào kho không
            if stock_w >= prod_w and stock_h >= prod_h:
                position = self._find_position_in_pattern(pattern, stock, prod_size)
                if position:
                    posX, posY = position
                    return {"stock_idx": stock_idx, "size": prod_size, "position": (posX, posY)}

        # Nếu không tìm thấy patterns phù hợp, trả về action mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_position_in_pattern(self, pattern, stock, prod_size):
        """Tìm vị trí phù hợp trong pattern hiện tại."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Duyệt qua các vị trí có thể trong kho để tìm vị trí phù hợp
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None

    def _generate_new_pattern(self, products, stocks):
        """Sinh pattern mới cho chính sách 1."""
        for prod in products:
            if prod["quantity"] > 0:
                new_action = self._generate_column(prod, stocks)
                if new_action:
                    return new_action

        # Nếu không tìm được giải pháp
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _generate_column(self, prod, stocks):
        """Sinh cột mới dựa trên sản phẩm và kho hiện tại với tối ưu hóa."""
        prod_size = prod["size"]
        prod_w, prod_h = prod_size
        best_pattern = None
        best_waste = float("inf")

        # Duyệt qua từng kho để tìm vị trí tốt nhất cho sản phẩm
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Kiểm tra xem sản phẩm có thể được đặt vào kho không
            if stock_w >= prod_w and stock_h >= prod_h:
                position = self._find_position_in_stock(stock, prod_size)
                if position:
                    # Tính waste của pattern
                    waste = (stock_w * stock_h) - (prod_w * prod_h)
                    # Cập nhật nếu waste thấp hơn
                    if waste < best_waste:
                        best_waste = waste
                        best_pattern = {
                            "stock_idx": stock_idx,
                            "size": prod_size,
                            "position": position
                        }

        return best_pattern if best_pattern else {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_position_in_stock(self, stock, prod_size):
        """Tìm vị trí trong kho để đặt sản phẩm."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Duyệt qua các vị trí có thể trong kho
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None

    def _branch(self):
        """Phân nhánh dựa trên các patterns hiện tại."""
        if not self.branch_stack:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Lấy nhánh đầu tiên từ ngăn xếp
        branch = self.branch_stack.pop()
        return branch

    def _create_branch(self, pattern):
        """Tạo các nhánh mới với nhiều cách đặt khác nhau."""
        stock_idx = pattern["stock_idx"]
        # Thêm nhiều vị trí và kích thước hơn để tối ưu không gian
        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2)]
        sizes = [[1, 1], [2, 1], [1, 2], [2, 2], [3, 1], [1, 3]]
        
        best_branch = {
            "stock_idx": pattern["stock_idx"],
            "size": [1, 1],
            "position": (0, 0)
        }
        best_score = 0
        
        # Duyệt qua các vị trí và kích thước để tìm nhánh tốt nhất
        for pos in positions:
            for size in sizes:
                branch = {
                    "stock_idx": stock_idx,
                    "size": size,
                    "position": pos
                }
                # Cải thiện cách tính điểm để cân bằng giữa tỷ lệ lấp đầy và waste
                area_ratio = (size[0] * size[1]) / ((pos[0] + size[0]) * (pos[1] + size[1]))
                edge_penalty = 0.1 if (pos[0] == 0 or pos[1] == 0) else 0  # Ưu tiên đặt sát cạnh
                compactness = 1 / (pos[0] + pos[1] + 1)  # Ưu tiên đặt gần gốc tọa độ
                score = area_ratio + edge_penalty + compactness
                
                if score > best_score:
                    best_score = score
                    best_branch = branch
        
        return best_branch

    def _initialize_patterns2(self, stocks):
        """Khởi tạo các patterns cơ bản dựa trên kho hiện tại."""
        for stock_idx, stocks in enumerate(stocks):
            self.patterns.append({
                "stock_idx": stock_idx,
                "pos": [],  # Chưa đặt sản phẩm nào
            })
            
    def _solve_master_problem2(self, prod, stocks):
        """Giải bài toán mẹ để tìm vị trí tốt nhất cho sản phẩm."""
        prod_size = prod["size"]
        prod_w, prod_h = prod_size
        
        # Duyệt qua các patterns hiện tại
        for pattern in self.patterns:
            stock_idx = pattern["stock_idx"]
            stock = stocks[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            
            # Kiểm tra xem sản phẩm có thể được đặt vào kho không
            if stock_w >= prod_w and stock_h >= prod_h:
                # Tìm vị trí tốt nhất để đặt sản phẩm
                position = self._find_position_in_pattern2(pattern, stock, prod_size)
                if position:
                    posX, posY = position
                    return {"stock_idx": stock_idx, "size": prod_size, "position": (posX, posY)}
        
        # Nếu không tìm thấy vị trí hợp lệ, trả về action mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def _find_position_in_pattern2(self, pattern, stock, prod_size):
        """Tìm vị trí tốt nhất để đặt sản phẩm trong kho."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        
        # Duyệt qua các vị trí có thể trong kho
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None
    
    def _generate_new_pattern2(self, products, stocks):
        """Sinh pattern mới nếu không tìm thấy vị trí hợp lệ."""
        for prod in products:
            if prod["quantity"] > 0:
                new_action = self._generate_new_pattern2_helper(prod, stocks)
                if(new_action["stock_idx"] != -1):
                    return new_action
        # Nếu không tìm thấy vị trí hợp lệ, trả về action mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def _generate_new_pattern2_helper(self, prod, stocks):
        """Sinh pattern mới nếu không tìm thấy vị trí hợp lệ."""
        prod_size = prod["size"]
        prod_w, prod_h = prod_size
        
        # Duyệt qua các kho hiện tại
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            # Kiểm tra xem sản phẩm có thể được đặt vào kho không
            if stock_w >= prod_w and stock_h >= prod_h:
                # Thêm một pattern mới với sản phẩm được đặt tại vị trí (0, 0)
                new_pattern = {
                    "stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}
                
                self.patterns.append(new_pattern)
                return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}
        # Nếu không tìm thấy vị trí hợp lệ, trả về action mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _get_stock_size_(self, stock):
        """Trả về kích thước của kho."""
        if isinstance(stock, np.ndarray):
            return stock.shape[1], stock.shape[0]  # width, height
        else:
            raise ValueError("Invalid stock format: {}".format(stock))

    def _calculate_waste(self, action, stocks, prod_size):
        """Tính toán waste cho một action cụ thể."""
        stock = stocks[action["stock_idx"]]
        stock_w, stock_h = self._get_stock_size_(stock)
        used_area = prod_size[0] * prod_size[1]
        total_area = stock_w * stock_h
        return total_area - used_area
