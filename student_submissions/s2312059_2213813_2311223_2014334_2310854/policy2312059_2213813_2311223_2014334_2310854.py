from policy import Policy
import numpy as np


class Policy2312059_2213813_2311223_2014334_2310854(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            # AFPTAS Policy
            self.policy_id = policy_id
            self.epsilon = 0.5  # Tham số epsilon
            self.threshold = None
            self.avg_stock_width = None
            self.max_stock_width = None
            self.min_stock_width = None
            self.current_stock = None
            self.actions = []
            pass
        elif policy_id == 2:
            # Guillotine Cutting với First Fit Decreasing Height Policy
            self.policy_id = policy_id
            self.stocks_available_regions = None
            pass

    def get_action(self, observation, info):
        # Student code here
        #POLICY 1:
        if (self.policy_id == 1): 
            action = self.policy_1_action(observation, info)
            return action
        #POLICY 2:
        if (self.policy_id == 2): 
            action = self.policy_2_action(observation, info)
            return action
        pass

    # Student code here
    # You can add more functions if needed

# functions for POLICY 1:
    def policy_1_action(self, observation, info):
        """
        get_action() function của policy_1:
        Asymptotic Fully Polynomial Approximation Scheme (AFPTAS)
        """
        products = observation["products"]
        stocks = observation["stocks"]

        #Tính self.epsilon và avg_stock_width
        if not self.avg_stock_width:
            self.avg_stock_width = sum(self._get_stock_size_(s)[0] for s in stocks) / len(stocks)
            self.epsilon = (
                sum(p["size"][0] * p["size"][1] * p["quantity"] for p in products) / 
                sum(self._get_stock_size_(s)[0] * self._get_stock_size_(s)[1] for s in stocks)
            )
            # print(self.epsilon)

        if len(products) < 2 and np.sum([p["quantity"] > 0 for p in products]) < 10:
            action = self.best_fit_packing(products, stocks)
            if action: return action

        # Phân loại sản phẩm thành wide, narrow dựa trên epsilon và avg_stock_width
        wide_prods, narrow_prods = self.classify_items(products, self.avg_stock_width)

        if np.sum([p["quantity"] > 0 for p in wide_prods]) > 0: 
            action = self.first_fit_decreasing(wide_prods, stocks)
            # print("w")
        else:  # Nếu hết wide_prods -> sang narrow_prods
            if not self.current_stock:
                self.current_stock = 0
            for stock_idx, stock in enumerate(stocks[self.current_stock:], start=self.current_stock):
                action = self.fill_remaining_space(stock, narrow_prods, stock_idx)
                if action is not None:
                    self.actions.append(action)
                    break
            self.current_stock = stock_idx
        if not action:
            action = self.best_fit_packing(narrow_prods, stocks)

        if not action:
            return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
        # print(action)
        return action

    def classify_items(self, products, avg_stock_width):
        """
        Phân loại sản phẩm thành wide và narrow dựa trên ngưỡng epsilon.
        """
        threshold = self.epsilon / (2 + self.epsilon) * avg_stock_width
        self.threshold = threshold
        wide_prods = [p for p in products if p["size"][0] > threshold]
        narrow_prods = [p for p in products if p["size"][0] <= threshold]
        # print({"len_nr_prods" : len(narrow_prods), "len_wi_prods" : len(wide_prods)})
        return wide_prods, narrow_prods

    def first_fit_decreasing(self, wide_prods, stocks):
        """
        Đóng gói sản phẩm 'wide' theo chiến lược First Fit Decreasing.
        """
        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_wide_prods = sorted(wide_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for i, prod in enumerate(sorted_wide_prods):
            if prod["quantity"] <= 0:
                continue
            prod_w, prod_h = prod["size"]
            for stock_idx, stock in enumerate(stocks):
                position = self.find_position_min_waste(stock, (prod_w, prod_h))
                if position:
                    action = {
                        "type" : "wide",
                        "stock_idx": stock_idx,
                        "prod_idx" : i,
                        "size": prod["size"],
                        "position": position,
                    }
                    self.actions.append(action)
                    return action

    def fill_remaining_space(self, stock, narrow_prods, stock_idx):
        """
        Lấp đầy khoảng trống bằng các sản phẩm nhỏ.
        """
        narrow_prods = sorted(narrow_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for i, prod in enumerate(narrow_prods):
            if prod["quantity"] <= 0:
                continue
            prod_w, prod_h = prod["size"]
            position = self.find_position_min_waste(stock, (prod_w, prod_h))
            if position:
                action = {
                    "type": "narrow",
                    "stock_idx": stock_idx,
                    "prod_idx" : i,
                    "size": prod["size"],
                    "position": position,
                }
                return action
        return None
    
    #
    # BEST_FIT_PACKING
    #

    def best_fit_packing(self, products, stocks):
        """
        Chiến lược đóng gói sản phẩm với tiêu chí Best Fit.
        """
        products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for i, prod in enumerate(products):
            if prod["quantity"] <= 0:
                continue
            prod_w, prod_h = prod["size"]

            best_stock_idx = None
            best_position = None
            best_score = float("inf")  # Score thấp nhất là tốt nhất
            
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Tìm vị trí phù hợp đầu tiên trong kho
                position = self.find_position_min_waste(stock, prod["size"])
                if position:
                    # Tính điểm dựa trên tiêu chí:
                    total_area = stock_w * stock_h
                    used_area = np.sum(stock >= 0) + prod_w * prod_h
                    waste = total_area - used_area
                    
                    score = waste

                    if score < best_score:
                        best_score = score
                        best_stock_idx = stock_idx
                        best_position = position

            if best_stock_idx is not None and best_position is not None:
                action = {
                    "stock_idx": best_stock_idx,
                    "size": prod["size"],
                    "prod_idx" : i,
                    "position": best_position,
                }
                self.actions.append(action)
                return action  # Trả về ngay khi tìm được
        return None

    def find_position_min_waste(self, stock, prod_size):
        """
        Tìm vị trí khả thi với lãng phí không gian tối thiểu.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        
        # Kiểm tra nếu vùng không đủ lớn
        if stock_w < prod_w or stock_h < prod_h:
            # Thử xoay item
            if stock_w >= prod_h and stock_h >= prod_w:
                prod_w, prod_h = prod_h, prod_w
            else:
                return None

        # Duyệt qua các vị trí khả thi và tính vị trí mà ít lãng phí diện tích nhất
        best_position = None
        min_waste = float("inf")

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    # Tính lãng phí không gian
                    total_area = stock_w * stock_h
                    used_area = np.sum(stock >= 0) + prod_w * prod_h
                    waste = total_area - used_area
                    if waste < min_waste:
                        min_waste = waste
                        best_position = (x, y)

        return best_position

    def filled_ratio(self, observation):
        """
        Tính ra các số liệu phục vụ quan sát kết quả.
        """
        stocks = observation["stocks"]
        used_area = 0
        total_area = 0
        used_stocks = np.sum(np.any(stock >= 0) for stock in stocks)

        for stock in stocks: 
            stock_w, stock_h = self._get_stock_size_(stock)
            if (np.any(stock >= 0)): total_area += stock_w * stock_h
            used_area += np.sum(stock >= 0)  # Số lượng ô đã sử dụng
        return (used_area, total_area, used_stocks) 
    
# functions for POLICY 2:
    def policy_2_action(self, observation, info):
        """
        get_action() function của policy_1 
        (Guillotine Cutting với FFDH Policy)
        """
        products = observation["products"]
        stocks = observation["stocks"]

        # Khởi t
        # Tạo stocks_available_regions:
        if not self.stocks_available_regions:
            self.stocks_available_regions = [[{"size" : self._get_stock_size_(stock), 
                                            "position" : (0, 0)}] for stock in stocks]
        action = self.first_fit_decreasing_height(stocks, products)
        # print(action)
        if not action:
            return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
        return action
    
    def first_fit_decreasing_height(self, stocks, items):
        """
        Cắt các stock theo chiến lược First Fit Decreasing Height.
        """
        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_items = sorted(items, key=lambda p: p["size"][0] * p["size"][1], reverse=True)
        for i, item in enumerate(sorted_items):
            if item["quantity"] <= 0:
                continue
            item_w, item_h = item["size"]
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                # Nếu diện tích của stock < item 
                if item_w*item_h > stock_w*stock_h: continue
                # Nếu diện tích phần trống < item
                remaining_area = np.sum(stock == -1) 
                if remaining_area < item_h * item_w: continue
                action = self.guillotine_cut(stock, stock_idx, item, i)
                if not action: continue
                return action
        pass

    def guillotine_cut(self, stock, stock_idx, item, item_idx):
        """
        Thực hiện Guillotine Cut trên stock.
        """
        for region_idx, region in enumerate(self.stocks_available_regions[stock_idx]):
            item_w, item_h = item["size"]
            stock_w, stock_h = self._get_stock_size_(stock)
            region_w, region_h = region["size"]
            
            # Kiểm tra nếu vùng không đủ lớn
            if region_w < item_w or region_h < item_h:
                # Thử xoay item
                if region_w >= item_h and region_h >= item_w:
                    item_w, item_h = item_h, item_w
                else:
                    continue
            # Kiểm tra xem stock có đặt được item ko
            if not self._can_place_(stock, region["position"], item["size"]):
                continue

            # Cập nhật vùng khả dụng
            self.stocks_available_regions[stock_idx].pop(region_idx)
            if region_w > item_w:
                self.stocks_available_regions[stock_idx].append({
                    "size": (region_w - item_w, item_h),
                    "position": (region["position"][0] + item_w, region["position"][1])
                })
            if region_h > item_h:
                self.stocks_available_regions[stock_idx].append({
                    "size": (region_w, region_h - item_h),
                    "position": (region["position"][0], region["position"][1] + item_h)
                })
            
            # Sắp xếp lại vùng khả dụng theo diện tích
            self.stocks_available_regions[stock_idx].sort(
                key=lambda p: p["size"][0] * p["size"][1], 
            )

            # Trả về action
            return {
                "stock_idx": stock_idx,
                "size": (item_w, item_h),
                "position": region["position"],
            }

        # Không tìm thấy position phù hợp
        return None

