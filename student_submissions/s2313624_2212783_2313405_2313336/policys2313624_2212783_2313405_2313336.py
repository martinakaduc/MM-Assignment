from policy import Policy
import numpy as np

class Policy2313624_2212783_2313405_2313336(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = policy_id
            pass
        elif policy_id == 2:
            self.policy_id = policy_id
            pass

    # Student code here
    # You can add more functions if needed

    def get_action(self, observation, info):
    
        if self.policy_id == 1:
            list_prods = observation["products"]
            stocks = observation["stocks"]

            # mảng này chứa các mẫu cắt
            patterns = []

            # khởi tạo chi phí tối thiểu
            min_cost = float("inf")
            best_action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

            # vòng lặp giải thuật column generation
            while True:
                # tạo mẫu cắt mới dựa 
                new_pattern, cost = self._generate_cutting_pattern(list_prods, stocks)

                # nếu không tìm được mẫu cắt mới hoặc chi phí tệ hơn thì out
                if new_pattern is None or cost >= min_cost:
                    break

                # thêm vào các mẫu cắt
                patterns.append(new_pattern)

                # cập nhật chi phi min và action tốt nhất
                min_cost = cost
                best_action = new_pattern["action"]

            return best_action
        elif self.policy_id == 2:
            best_pattern = None
            self.cut_patterns = self._generate_columns(observation)    #Gọi _generate_columns để tạo danh sách các mẫu cắt
            
            #Chọn best_pattern từ danh sách (Chọn mẫu đầu tiên để tối uy thời gian)
            for pattern in self.cut_patterns:
                prod, stock, pos_x, pos_y = pattern
                if best_pattern is None:
                    best_pattern = pattern

            if best_pattern is not None:
                prod, stock, pos_x, pos_y = best_pattern
                stock_idx = -1
                for idx, s in enumerate(observation["stocks"]):    #Vòng lặp để tìm chỉ số stock_idx tương ứng với kho trong observation.
                    if self._get_stock_size_(s) == self._get_stock_size_(stock):
                        stock_idx = idx
                        break
                
                return {"stock_idx": stock_idx, "size": prod["size"], "position": (pos_x, pos_y)}    #Trả về thông tin hành động 
            else:
                
                return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}    #Trả về hành động mặc định nếu không tìm được pattern
            
    def _find_valid_position(self, stock, prod, rotated=False):    
    #Tìm một vị trí hợp lệ trong kho stock để đặt sản phẩm prod.
        
        prod_w, prod_h = prod["size"]    #Lấy kích thước sản phẩm 
        stock_w, stock_h = self._get_stock_size_(stock)    #Lấy kích thước kho hàng
        
        #Nếu rotated == True, hoán đổi chiều rộng và chiều cao của sản phẩm
        if rotated:
            prod_w, prod_h = prod_h, prod_w
            
        #Dừng ngay nếu sản phẩm lớn hơn kho
        if prod_w > stock_w or prod_h > stock_h:
            return None, None
        
        #Duyệt qua từng vị trí trong kho, kiểm tra nếu có thể đặt sản phẩm ở đó bằng _can_place_. Nếu có, trả về vị trí (x, y)
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    return x, y
                
        #Nếu không tìm được vị trí nào, trả về None, None
        return None, None    
    
    
    def _generate_columns(self, observation):
    #Tạo danh sách các mẫu cắt bằng cách duyệt qua từng sản phẩm và kho hàng.  
        
        #Sắp xếp sản phẩm theo diện tích giảm dần  
        sorted_products = sorted(
            observation["products"],
            key=lambda p: p["size"][1] * p["size"][0],
            reverse=True
        )
        
        columns = []    #Tạo danh sách mẫu rỗng 
        
        #Vòng lặp duyệt qua từng vị trí trong kho hàng
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            #Vòng lặp duyệt qua từng sản phẩm 
            for prod in sorted_products:
                prod_w, prod_h = prod["size"]
                if prod["quantity"] > 0:
                    best_fit_pos = None
                    best_fit = float("inf")

                    #Xoay sản phẩm để tìm vị trí tốt nhất
                    for rotated in [False, True] if prod_w != prod_h else [False]:    #Nếu kích thước 2 cạnh bằng nhau không cần xoay
                        pos_x, pos_y = self._find_valid_position(stock, prod, rotated=rotated)
                        if pos_x is not None and pos_y is not None:
                            remaining_area = (stock_w * stock_h) - (prod_w * prod_h)
                            if best_fit is None or remaining_area < best_fit:         
                                best_fit = remaining_area
                                best_fit_pos = (pos_x, pos_y, rotated)
                            if best_fit == 0:
                                break

                    if best_fit_pos:
                        pos_x, pos_y, rotated = best_fit_pos
                        prod_copy = prod.copy()                             #Tạo bản sao độc lập để tránh làm ảnh hưởng đến dữ liệu sản phẩm
                        if rotated:
                            prod_copy["size"] = (prod_h, prod_w)            #Cập nhật lại kích thước sản phẩm nếu xoay
                        prod_copy["quantity"] -= 1                          #Giảm số lượng sản phẩm
                        columns.append((prod_copy, stock, pos_x, pos_y))    #Thêm danh sách vào mẫu
                    else:
                        continue  #Nếu không tìm được vị trí hợp lệ
        return columns
   
    def _generate_cutting_pattern(self, list_prods, stocks):
        # tìm một mẫu cắt mới bằng cách duyệt qua các sản phẩm và stocks
        # trả về mẫu cắt mới và chi phí
        min_trim_loss = float("inf")
        best_pattern = None

        # sắp xếp theo kích thước giảm dần
        sorted_prods = sorted(list_prods, key=lambda x: max(x["size"]), reverse=True)

        # duyệt qua từng sản phẩm
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # duyệt qua từng stock
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # kiểm tra xem sản phẩm fit vào stock không
                    if stock_w < prod_w and stock_h < prod_h:
                        continue

                    # duyệt qua tất cả các vị trí có thể đặt sản phẩm
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break

                    # nếu vị trí hợp lệ thì tính trim_loss
                    if pos_x is not None and pos_y is not None:
                        # tạo bản sao của stock để cập nhật
                        stock_copy = stock.copy()
                        stock_copy[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] = -2  # -2 là ô đã sử dụng

                        # tính trim loss
                        trim_loss = self._calculate_trim_loss(stock_copy)

                        # nếu trim loss thấp hơn, cập nhật mẫu
                        if trim_loss < min_trim_loss:
                            min_trim_loss = trim_loss
                            best_pattern = {
                                "action": {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y),
                                },
                                "trim_loss": trim_loss
                            }

        if best_pattern is not None:
            return best_pattern, min_trim_loss
        else:
            return None, float("inf")


    # tính toán trim loss
    def _calculate_trim_loss(self, stock):
        trim_loss = np.sum(stock == -1)  # ô trống được đánh dấu là -1
        return trim_loss


   

