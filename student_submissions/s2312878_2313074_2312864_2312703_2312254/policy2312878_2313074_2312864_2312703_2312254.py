from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2312878_2313074_2312864_2312703_2312254(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        if policy_id == 1:
            pass
        elif policy_id == 2:
            self.is_1st_call = True
            self.prod_queue = []

    def get_action(self, observation, info):
        # Student code here
    #FIRST ALGORITHM    
        if self.policy_id == 1:
            
            list_prods = observation["products"]
            stocks = observation["stocks"]

            def get_best_priority(prod):
                best_priority = float("inf")
                best_stock_idx = -1
                best_waste = float("inf")
                
                for idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod["size"]
                    
                    # Tính toán trim loss cho cả 2 hướng đặt
                    waste1 = (stock_w * stock_h) - (prod_w * prod_h)
                    waste2 = (stock_w * stock_h) - (prod_h * prod_w)
                    current_waste = min(waste1, waste2)

                    # Tính toán các giá trị thường dùng
                    prod_area = prod_w * prod_h
                    stock_area = stock_w * stock_h
                    max_prod_dim = max(prod_w, prod_h)
                    min_prod_dim = min(prod_w, prod_h)
                    max_stock_dim = max(stock_w, stock_h)
                    min_stock_dim = min(stock_w, stock_h)

                    priority = []

                    # Rule 1: Nếu diện tích sản phẩm >= 85% diện tích tấm
                    if prod_area >= 0.85 * stock_area:
                        priority.append(1)
                    # Rule 2: Nếu cả hai kích thước của sản phẩm > 50% kích thước tương ứng của tấm
                    # Ưu tiên cao nhất vì đảm bảo sử dụng hiệu quả không gian
                    if ((prod_w > 0.5 * stock_w and prod_h > 0.5 * stock_h) or 
                        (prod_h > 0.5 * stock_w and prod_w > 0.5 * stock_h)):
                        priority.append(2)

                    # Rule 3: Nếu một kích thước của sản phẩm gần bằng kích thước tương ứng của tấm
                    # (trong khoảng 90-100%)
                    if (max_prod_dim >= 0.9 * max_stock_dim or 
                        min_prod_dim >= 0.9 * min_stock_dim):
                        priority.append(3)

                    # Rule 4: Nếu tỷ lệ kích thước sản phẩm tương đồng với tỷ lệ tấm
                    # (độ chênh lệch < 10%)
                    aspect_ratio_diff = abs(prod_w/prod_h - stock_w/stock_h)
                    if aspect_ratio_diff < 0.1:
                        priority.append(4)

                    # Rule 5: Ưu tiên sản phẩm có ít trim loss
                    # (dưới 15% diện tích tấm)
                    if current_waste < 0.15 * stock_area:
                        priority.append(5)

                    # Rule 6: Nếu sản phẩm vừa khít theo một chiều
                    # (trong khoảng 95-100% của chiều tương ứng)
                    if (max(prod_w/stock_w, prod_h/stock_h) > 0.95 or 
                        max(prod_w/stock_h, prod_h/stock_w) > 0.95):
                        priority.append(6)

                    # Rule 7: Nếu diện tích sản phẩm >= 50% diện tích tấm
                    # Rule phụ để đảm bảo sử dụng hiệu quả trong trường hợp không thỏa các rule trên
                    if prod_area >= 0.5 * stock_area:
                        priority.append(7)
                    
                    current_priority = min(priority) if priority else float("inf")
                    if current_priority < best_priority or (current_priority == best_priority and current_waste < best_waste):
                        best_priority = current_priority
                        best_stock_idx = idx
                        best_waste = current_waste
                
                return (best_priority, best_waste, best_stock_idx)

            # Sort products based on best priority across all stocks
            sorted_prods_with_priority = []
            for prod in list_prods:
                if prod["quantity"] > 0:  # Chỉ xét các sản phẩm có quantity > 0
                    priority, waste, best_idx = get_best_priority(prod)
                    sorted_prods_with_priority.append((prod, priority, waste, best_idx))

            # Sắp xếp theo priority và waste
            sorted_prods_with_priority.sort(key=lambda x: (x[1], x[2]))

            # Thử đặt từng sản phẩm
            for prod, _, _, best_idx in sorted_prods_with_priority:
                prod_size = prod["size"]
                
                # Thử đặt vào best stock trước
                if best_idx != -1:
                    stock = stocks[best_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for orientation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
                        prod_w, prod_h = orientation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                        return {
                                            "stock_idx": best_idx,
                                            "size": (prod_w, prod_h),
                                            "position": (x, y)
                                        }

                # Nếu không đặt được vào best stock, thử các stock khác
                for i, stock in enumerate(stocks):
                    if i == best_idx:  # Bỏ qua stock đã thử
                        continue
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for orientation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
                        prod_w, prod_h = orientation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                        return {
                                            "stock_idx": i,
                                            "size": (prod_w, prod_h),
                                            "position": (x, y)
                                        }

            return {"stock_idx": -1, "size": (0, 0), "position": (None, None)}
    


    
    #SECOND ALGORITHM
        elif self.policy_id == 2:
            # If this is the 1st call, sort the product base on their areas in the descending order
            if self.is_1st_call:
                self.is_1st_call = False
                if "products" in observation:  # If there exist products in observation
                    self.prod_queue = sorted(
                        observation["products"],  # List of products
                        key=lambda product: product["size"][0] * product["size"][1],  # Sort by the area
                        reverse=True)  # Descending order
                # print("prod_queue:", self.prod_queue)

            # If this is not the 1st call
            # Step 1: take the 1st product in the prod_queue
            # The 1st product is always the 1 with the largest area
            product = self.prod_queue[0]
            # print("prod:", product)

            # Step 2: stock selection (need to be worked on to find the best stock)
            # Try to put the product on every possible stock, calculate the wastage and
            # choose the stock with the smallest wastage
            stock, stock_idx, pos_x, pos_y, is_rotated = self.choose_stock(observation, product)
            stock_W, stock_L = self._get_stock_size_(stock)
            prod_w, prod_l = product["size"]

            # Step 3: Guillotine cut (ưu tiên đặt theo chiều ngang trước)
            # Vì bài toán cho phép xoay product nên trước hết ta sẽ ưu tiên chiều nào dài hơn để xếp vào
            if is_rotated:
                if self.prod_queue[0]["quantity"] - 1 == 0:
                    self.prod_queue.pop(0)
                if len(self.prod_queue) == 0:  # If all the products is place, reset for a new patch
                    self.is_1st_call = True
                return {"stock_idx": stock_idx, "size": product["size"][::-1], "position": (pos_x, pos_y)}
            elif not is_rotated:
                if self.prod_queue[0]["quantity"] - 1 == 0:
                    self.prod_queue.pop(0)
                if len(self.prod_queue) == 0:  # If all the products is place, reset for a new patch
                    self.is_1st_call = True
                return {"stock_idx": stock_idx, "size": product["size"], "position": (pos_x, pos_y)}
            else:  # If not found the position
                return {"stock_idx": -1, "size": product["size"], "position": (0, 0)}

    def choose_stock(self, observation, product):
        min_wastage = float("inf")
        chosen_stock_idx = -1
        return_pos_x, return_pos_y = None, None
        chosen_stock = None
        prod_w, prod_l = product["size"]
        is_rotated = None

        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_W, stock_L = self._get_stock_size_(stock)
            wastage = float("inf")  # The current stock's wastage if the product is placed here
            rotated = 0  # Odd number means rotated = True and False otherwise
            pos_x, pos_y = None, None
            if prod_w >= prod_l:
                long = prod_w
                short = prod_l
            else:
                rotated += 1
                long = prod_l
                short = prod_w
            if stock_W >= long and stock_L >= short:
                flag = False  # Use to break the outer for loop when found the wastage
                for y in range(stock_L - short + 1):
                    for x in range(stock_W - long + 1):
                        if self._can_place_(stock, (x, y), (long, short)):
                            pos_x, pos_y = x, y
                            wastage = self.calc_wastage(stock, prod_w, prod_l)
                            flag = True
                            break
                    if flag:
                        break
            elif stock_W >= short and stock_L >= long:  # Second case if the product does not fit
                # NOTE: we have to rotate the product 90 degree in this case
                rotated += 1
                flag = False  # Use to break the outer for loop when found the wastage
                for y in range(stock_L - long + 1):
                    for x in range(stock_W - short + 1):
                        if self._can_place_(stock, (x, y), (short, long)):
                            pos_x, pos_y = x, y
                            wastage = self.calc_wastage(stock, prod_w, prod_l)
                            flag = True
                            break
                    if flag:
                        break

            if float(wastage) < min_wastage:
                min_wastage = wastage
                chosen_stock_idx = stock_idx
                chosen_stock = stock
                is_rotated = (rotated % 2 == 1)
                return_pos_x, return_pos_y = pos_x, pos_y

        # print("chosen_stock_idx:", chosen_stock_idx, "(pos_x, pos_y):",
        #     return_pos_x, return_pos_y, "rotated:", is_rotated)
        return chosen_stock, chosen_stock_idx, return_pos_x, return_pos_y, is_rotated

    def calc_wastage(self, stock, prod_w, prod_l):
        stock_free_place = np.count_nonzero(stock == -1)  # Count the number of -1 as for the free area
        return stock_free_place - prod_w * prod_l

    # Student code here
    # You can add more functions if needed

