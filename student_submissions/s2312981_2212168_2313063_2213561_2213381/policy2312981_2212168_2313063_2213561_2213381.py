from policy import Policy
import random
import numpy as np
class Policy2312981_2212168_2313063_2213561_2213381(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        if self.policy_id == 2:
            self.alpha = 91  # Tỉ lệ lựa chọn utilization của TwoSGP 
            self.beta = 91   # Tỉ lệ lựa chọn utilization của ThreeSHP 
            #! New parameters
            self.best_plan = None   # Placeholder for the best cutting plan
            self.is_best_plan = False # True if the best plan is exist 
            self.best_plan_iterator = 0 
            self.best_utimization = 0
            
            self.used_stocks = [] #Lưu trữ các stock đã hoàn thành (có bản cắt)
            self.previous_observation = None # Chứa observation cũ 

    def get_action(self, observation, info):
        """
        Decide action based on the selected policy ID.
        """
        if self.policy_id == 1:
            return self._policy_1(observation, info)
        elif self.policy_id == 2:
            return self._policy_2(observation, info)
        else:
            pass

### POLICY 1
    def _policy_1(self, observation, info):
        """
        Implement the First Fit Decreasing Height (FFDH) algorithm with product rotation.
        """
        # Sort products by area (width * height) in descending order
        products = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
        
        stocks = observation["stocks"]  # Stocks from the observation
        stock_idx = -1  # Initialize selected stock index
        pos_x, pos_y = None, None  # Initialize placement position
        prod_size = None  # Initialize selected product size

        for prod in products:
            if prod["quantity"] <= 0:
                continue

            # Get product dimensions
            original_size = prod["size"]
            rotated_size = original_size[::-1]  # Rotated dimensions
            
            # Try both orientations
            for orientation in [original_size, rotated_size]:
                prod_w, prod_h = orientation
                placed = False
                
                # Iterate over stocks
                for idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Skip if the stock is too small
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Try placing the product
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), orientation):
                                stock_idx = idx
                                pos_x, pos_y = x, y
                                prod_size = orientation
                                placed = True
                                break
                        if placed:
                            break
                    
                    if placed:
                        break

                if placed:
                    break

            # If a valid placement is found, exit loop
            if stock_idx != -1:
                break

        # If no valid placement, return None
        if stock_idx == -1 or pos_x is None or pos_y is None:
            return None

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
### POLICY 2
#================================================================
# TWO STAGE GENERAL PATTERNS: GENERAL STRIPS - DẢI CHUNG #!(COMPLETED)
#================================================================
    def TwoSGP(self,observation): #Two Stage General Patterns
        """
            Purpose:    
                Xử lý các sản phẩm có số lượng ít, có kích thước khác nhau 
                
            Parameters:
                observation (dict):
                    - stocks: danh sách các tấm lớn (mỗi tấm chứa chiều rộng và chiều cao).
                    - products: danh sách sản phẩm (mỗi sản phẩm chứa kích thước và số lượng).
                    
            Returns:
                best_pattern, best_utilization: bản cắt tốt nhất và sự tận dụng của mẫu cắt tốt nhất cao nhất 
        """
        stocks = observation["stocks"]
        products = observation["products"]
    
        best_pattern = []  # Chứa bản cắt cuối cùng (tốt nhất)
        best_utilization = 0  # Sự tận dụng của bản cắt tốt nhất
    
        # Sắp xếp sản phẩm theo diện tích giảm dần
        products = sorted(products, key=lambda p: p['size'][0] * p['size'][1], reverse=True)
        for stock_idx, stock in enumerate(stocks):
            if stock_idx in self.used_stocks:
                continue
            stock_width, stock_height = self._get_stock_size_(stock)
            stock_grid = np.full((stock_width, stock_height), -1)  # Mảng giả đại diện cho stock
            current_pattern = []
            current_utilization = 0
            used_area = 0
            strip_start = 0

            while strip_start < stock_width:
                strip_width = stock_width - strip_start  # Tạo một dải mới với chiều rộng còn lại

                for product in products:
                    product_width, product_height = product['size']
                    quantity = product['quantity']
                    while quantity > 0:
                        placed = False
                        for i in range(strip_start, stock_width  - product_width + 1):
                        # for i in range(strip_start, strip_start + strip_width - product_width + 1):
                            for j  in range(stock_height - product_height + 1):
                                if quantity> 0 and self._can_place_(stock_grid, (i, j), (product_width, product_height)):
                                    # Đặt sản phẩm lên stock_grid
                                    stock_grid[i:i + product_width, j:j + product_height] = 1

                                    # Ghi lại thông tin cắt
                                    current_pattern.append({
                                        'stock_idx': stock_idx,
                                        'size': (product_width, product_height),
                                        'position': (i,j)
                                    })

                                    quantity -= 1
                                    placed = True
                                    break
                        if not placed:
                            break

                # Chuyển sang dải tiếp theo
                strip_start += strip_width
            # Tính tỷ lệ tận dụng
            used_area =np.sum(stock_grid==1) 
            current_utilization = (used_area / (stock_width * stock_height)) * 100
            
            # Cập nhật pattern tốt nhất nếu tận dụng cao hơn
            if current_utilization > best_utilization:
                best_pattern = current_pattern
                best_utilization = current_utilization
                if best_utilization >= self.alpha:
                    print(f"    |---TwoSHP: The stock {best_pattern[0]["stock_idx"]} has {best_utilization:.5f} % utilization.")
                    return best_pattern, best_utilization
        print(f"    |---TwoSHP: The stock {best_pattern[0]["stock_idx"]} has {best_utilization:.5f} % utilization.")
        return best_pattern, best_utilization

#================================================================
# THREE STAGE HOMOGENOUS PATTERNS: HOMOGENOUS STRIPS - DẢI ĐỒNG NHẤT #!(COMPLETED)
#================================================================
    def ThreeSHP(self, observation): 
        """
            Purpose:
                Xử lý các mẫu đồng nhất ba giai đoạn và trả về kết quả cắt.
                -> Xử lý các mẫu đồng nhất (Các mẫu có cùng kích thước, ưu tiên xử lý theo quantity)

            Parameters:
                observation: Dữ liệu đầu vào chứa thông tin về stocks và products

            Returns:
                best_utilization: Sự tận dụng của mẫu cắt tốt nhất
                best_plan: Mẫu cắt tốt nhất
            
        """

        stocks = observation["stocks"]
        products = observation["products"]
        best_pattern = []  # Chứa bản cắt cuối cùng (tốt nhất)
        best_utilization = 0  # Sự tận dụng của bản cắt tốt nhất
        # Sắp xếp sản phẩm theo quantity -> diện tích
        products = sorted(products, key=lambda x: (x['quantity'], x['size'][0] * x['size'][1]), reverse=True)
        
        for stock_idx, stock in enumerate(stocks):  # Duyệt qua từng stock
            stock_width, stock_height = self._get_stock_size_(stock)
            if stock_idx in self.used_stocks:
                continue
            current_pattern = []  # Chứa bản cắt hiện tại
            current_utilization = 0
            stock_grid = np.full((stock_width, stock_height), -1)  # Tạo mảng giả đại diện cho stock
            for product in products:
                prod_w, prod_h = product['size']
                prod_q = product['quantity']

                while prod_q > 0:  # Cắt cho đến khi hết số lượng sản phẩm này
                    placed = False
                    for orientation in [(prod_w, prod_h), (prod_h, prod_w)]:  # Kiểm tra cả hai hướng cắt
                        w, h = orientation

                        for x in range(stock_width - w + 1):
                            for y in range(stock_height - h + 1):
                                if self._can_place_(stock_grid, (x, y), (w,h)):
                                    # Đánh dấu vùng cắt trên stock grid
                                    stock_grid[x:x+w, y:y+h] = 1

                                    # Thêm thông tin cắt vào danh sách hiện tại
                                    current_pattern.append({
                                        "stock_idx": stock_idx,
                                        "size": (int(w), int(h)),
                                        "position": (x, y)
                                    })
                                    prod_q -= 1
                                    placed = True
                                    break
                            if placed:
                                break
                        if placed:
                            break
                    if not placed:  # Không còn chỗ để cắt sản phẩm này
                        break
            # Tính tỷ lệ tận dụng
            total_area_used =  np.sum(stock_grid == 1)
            current_utilization = (total_area_used / ( stock_width * stock_height)) * 100 

            # Cập nhật pattern tốt nhất nếu tận dụng cao hơn
            if current_utilization > best_utilization:
                best_pattern = current_pattern
                best_utilization = current_utilization
                if best_utilization >= self.beta:
                    print(f"    |---ThreeSHP: The stock {best_pattern[0]["stock_idx"]} has {best_utilization:.5f} % utilization.")
                    return best_pattern, best_utilization

        print(f"    |---ThreeSHP: The stock {best_pattern[0]["stock_idx"]} has {best_utilization:.5f} % utilization.")
        return best_pattern, best_utilization

#!================================================================
#! RUN ISHP ALGORITHM: ITERATIVE SINGLY HEURISTIC PATTERN
#!================================================================
    def run(self, observation):
        # Khởi tạo biến chứa mẫu cắt tốt nhất và độ tối ưu tốt nhất
        self.best_plan =[]
        self.best_utimization = 0
        self.is_best_plan = False
        sgp_plan,sgp_utimi = [], 0
        shp_plan,shp_utimi = [], 0
        
        # Tạo mẫu cắt 2SGP và mẫu cắt 3SHP
        sgp_plan,sgp_utimi = self.TwoSGP(observation) 
        shp_plan,shp_utimi = self.ThreeSHP(observation) 
        # So sánh hai mẫu cắt và trả về mẫu cắt tốt hơn 
        if sgp_plan or shp_plan:
            self.is_best_plan = True
            if sgp_utimi>=shp_utimi:
                self.best_plan, self.best_utimization = sgp_plan,sgp_utimi
            else:
                self.best_plan, self.best_utimization =  shp_plan,shp_utimi
        else:
            self.is_best_plan = False



    def _policy_2(self, observation, info):
        # compare the previous observation and current observation
        if not self.previous_observation or not np.array_equal(self.previous_observation["stocks"], observation["stocks"]):
            self.best_plan = []
            self.is_best_plan = False 
            self.best_plan_iterator = 0
            self.best_utimization = 0
            self.used_stocks = [] 
            self.previous_observation = observation
        # Chạy thuật toán ISHP 
        if not self.is_best_plan: # Nếu chưa có best_plan:
            self.run(observation)
            if self.best_plan:
                print(f"    |===]Best plan: The stock {self.best_plan[0]['stock_idx']} has {self.best_utimization:.5f}% utilization")
                for cut in self.best_plan:
                    print(f"        |---]Stock_idx: {cut['stock_idx']}, size: {cut['size']}, position: {cut['position']}")
        
        # Kiểm tra best_plan có tồn tại không.
        if self.is_best_plan:
            if self.best_plan_iterator < len(self.best_plan): # Lặp lại đến khi duyệt hết best_plan
                cut = self.best_plan[self.best_plan_iterator] # best plan này là cho toàn bộ hay cho 1 stock nhỉ
                self.best_plan_iterator += 1
                return cut 

            if self.best_plan_iterator == len(self.best_plan) and len(self.best_plan) != 0: 
                self.used_stocks.append(self.best_plan[0]["stock_idx"]) # Lấy giá trị stock_idx từ phần tử đầu tiên của best_plan                
                self.best_plan = []
                self.is_best_plan = False
                self.best_utimization = 0
                self.best_plan_iterator = 0
        else:
            self.is_best_plan = False
            print("Best Plan is not exist!")
        return {"stock_idx": -1, "size": [0,0], "position": (-1, -1)}

