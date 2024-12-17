from policy import Policy
import numpy as np

class Policy2310335_2311915_2310491_2310453(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.stockid = 0
            self.prodid = 0
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._get_action_1(observation, info)
        elif self.policy_id == 2:
            return self._get_action_2(observation, info)

    #Policy 1: First fit
    def _get_action_1(self, observation, info):
        # Student code here
        # Lấy danh sách sản phẩm và sắp xếp theo kích thước giảm dần (diện tích lớn nhất trước)
        list_prods = sorted(
            observation["products"],
            key=lambda prod: (prod["size"][0] * prod["size"][1],max(prod["size"][0]/prod["size"][1], prod["size"][1]/prod["size"][0])),
            reverse=True,
        )
        if all([np.sum(stock==-1).item()+np.sum(stock==-2).item()==10000 for stock in observation["stocks"]]):
            self.prodid = 0
            self.stockid = 0
        while self.stockid != len(observation["stocks"]):
            if all(prod["quantity"] == 0 for prod in list_prods):
                return {"stock_idx": -1, "size": [0, 0],"position": (0, 0)}
            elif self.prodid == len(list_prods):
                self.prodid = 0
                self.stockid += 1
                continue
            elif list_prods[self.prodid]["quantity"]==0:
                self.prodid +=1
                continue

            stock_w, stock_h = self._get_stock_size_(observation["stocks"][self.stockid])
            prod_w, prod_h = list_prods[self.prodid]["size"]
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(observation["stocks"][self.stockid], [x, y], list_prods[self.prodid]["size"]):
                        return {"stock_idx": self.stockid, "size": list_prods[self.prodid]["size"], "position": (x, y)} 
                    
            # Kiểm tra khả năng đặt sản phẩm sau khi xoay 90 độ
            for x in range(stock_w - prod_h + 1):
                for y in range(stock_h - prod_w + 1):
                    if self._can_place_(observation["stocks"][self.stockid], [x, y], list_prods[self.prodid]["size"][::-1]):
                        return {"stock_idx": self.stockid, "size": list_prods[self.prodid]["size"][::-1], "position": (x, y)} 
                    
            self.prodid +=1

        return {"stock_idx": -1, "size": [0, 0],"position": (0, 0)}
    #End of Policy 1


    #Policy 2: Best fit
    def _get_action_2(self, observation, info):
        # Student code here
        # Lấy danh sách sản phẩm và sắp xếp theo kích thước giảm dần (diện tích lớn nhất trước)
        list_prods = sorted(
            observation["products"],
            key=lambda prod: (prod["size"][0] * prod["size"][1],max(prod["size"][0]/prod["size"][1], prod["size"][1]/prod["size"][0])),
            reverse=True,
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Lặp qua từng sản phẩm theo thứ tự giảm dần
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                best_stock_idx = -1
                best_fit_gap = float('inf')  # Khoảng trống tốt nhất (nhỏ nhất)
                best_pos_x, best_pos_y = None, None
                best_size = prod_size

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Tính khoảng trống sau khi đặt sản phẩm
                                gap = np.sum(stock==-1) - (prod_w * prod_h)
                                if gap < best_fit_gap:
                                    best_fit_gap = gap
                                    best_stock_idx = i
                                    best_pos_x, best_pos_y = x, y
                                    best_size = prod_size
                                break

                    # Kiểm tra khả năng đặt sản phẩm sau khi xoay 90 độ
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                # Tính khoảng trống sau khi đặt sản phẩm
                                gap = np.sum(stock==-1) - (prod_h * prod_w)
                                if gap < best_fit_gap:
                                    best_fit_gap = gap
                                    best_stock_idx = i
                                    best_pos_x, best_pos_y = x, y
                                    best_size = prod_size[::-1]
                                break

                # Nếu tìm thấy tấm phù hợp nhất, lưu kết quả
                if best_stock_idx != -1:
                    stock_idx = best_stock_idx
                    pos_x, pos_y = best_pos_x, best_pos_y
                    prod_size = best_size
                    break
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}      
    #End of Policy 2
