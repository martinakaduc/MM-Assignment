from policy import Policy
import numpy as np

class Policy2312449_2312357_2312397_2312435_1652242(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here
        if policy_id == 1:
            self.policy=1
        elif policy_id == 2:
            self.policy=2
        
    def get_action(self, observation, info):

        if self.policy == 1:
            result = self.FIRST_FIT_DECREASING(observation, info)
        elif self.policy == 2:
            result = self.BEST_FIT_DECREASING(observation, info)

        return result
        
    def FIRST_FIT_DECREASING(self, observation, info):
        # Sắp xếp sản phẩm theo diện tích giảm dần (thuật toán First Fit Decreasing)
        list_prods = sorted(
            observation["products"], 
            key=lambda prod: prod["size"][0] * prod["size"][1], 
            reverse=True
        )
    
        #Sắp xếp số kho khả dụng theo diện tích từ nhỏ đến lớn (để tối ưu được trim_loss)
        list_stocks = list ( enumerate ( observation ["stocks"]) )
        list_stocks.sort ( key = lambda list : np . sum ( list [1] != -2) ,reverse = False )

        prod_size = [0, 0]  # Kích thước sản phẩm được chọn
        stock_idx = -1  # Chỉ số của kho phù hợp
        pos_x, pos_y = 0, 0  # Vị trí đặt sản phẩm

        # Duyệt qua các sản phẩm đã sắp xếp
        for prod in list_prods:
            if prod["quantity"] > 0:  # Chỉ xử lý sản phẩm còn tồn kho
                prod_size = prod["size"]
                for i, stock in list_stocks:
                    # Lấy kích thước kho
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    for rotate in [False, True]:  # Kiểm tra cả hai chiều (gốc và xoay)
                        if rotate:
                            prod_size = prod_size[::-1]  # Xoay kích thước sản phẩm
                            prod_w, prod_h = prod_size
                        if stock_w < prod_w or stock_h < prod_h:  # Bỏ qua nếu kho không đủ lớn
                            continue

                        pos_x, pos_y = None, None

                        # Tìm vị trí đầu tiên có thể đặt sản phẩm
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):  # Kiểm tra khả năng đặt
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:  # Nếu đã tìm thấy, dừng vòng lặp
                                break

                        if pos_x is not None and pos_y is not None:  # Nếu tìm được vị trí
                            stock_idx = i  # Lưu chỉ số kho
                            break
                    if pos_x is not None and pos_y is not None:
                        break  # Dừng tìm kiếm nếu đã đặt sản phẩm
                if pos_x is not None and pos_y is not None:
                    break  # Thoát vòng lặp nếu sản phẩm đã được đặt

        # Trả về hành động bao gồm vị trí đặt, kích thước, và chỉ số kho
        return {
            "stock_idx": stock_idx,
            "size": prod_size,
            "position": (pos_x, pos_y)
        }
    def BEST_FIT_DECREASING(self, observation, info):
        list_prods = observation["products"]
        list_stocks = observation["stocks"]

        # Tiền xử lý: Tính kích thước từng kho
        stock_sizes = [self._get_stock_size_(stock) for stock in list_stocks]

        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_prods = sorted(
            list_prods,
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
        list_stocks = list ( enumerate ( observation ["stocks"]) )

    # This line is stock sort : largest to smallest ( comment # it if you want to used normal stock )
        #list_stocks.sort ( key = lambda list : np . sum ( list [1] != -2) ,reverse = True )
        # Duyệt qua từng sản phẩm
        for prod in sorted_prods:
            if prod["quantity"] <= 0:
                continue

            prod_w, prod_h = prod["size"]
            best_result = {
                "size": [0, 0],
                "stock_idx": -1,
                "position": (None, None),
                "waste": float("inf")
            }

            # Duyệt qua từng kho
            for stock_idx, stock in list_stocks:
                stock_w, stock_h = stock_sizes[stock_idx]

                # Duyệt qua hai hướng: giữ nguyên và xoay 90 độ
                orientations = [(prod_w, prod_h), (prod_h, prod_w)]
                for current_w, current_h in orientations:
                    if current_w > stock_w or current_h > stock_h:
                        continue  # Bỏ qua nếu sản phẩm không vừa

                    # Duyệt qua từng vị trí tiềm năng
                    for x in range(stock_w - current_w + 1):
                        for y in range(stock_h - current_h + 1):
                            if self._can_place_(stock, (x, y), (current_w, current_h)):
                                # Tính toán lãng phí
                                waste = (stock_w * stock_h) - (current_w * current_h)

                                # Ưu tiên vị trí thấp hơn hoặc gần cạnh trái hơn
                                heuristic_value = (waste, y, x)

                                if waste < best_result["waste"] or (
                                    waste == best_result["waste"] and
                                    heuristic_value < best_result.get("heuristic_value", (float("inf"), float("inf"), float("inf")))
                                ):
                                    best_result = {
                                        "size": (current_w, current_h),
                                        "stock_idx": stock_idx,
                                        "position": (x, y),
                                        "waste": waste,
                                        "heuristic_value": heuristic_value,
                                    }

                                # Dừng sớm nếu tìm thấy vị trí tối ưu (lãng phí bằng 0)
                                if waste == 0:
                                    return {
                                        "size": best_result["size"],
                                        "stock_idx": best_result["stock_idx"],
                                        "position": best_result["position"],
                                    }

            # Nếu tìm thấy vị trí phù hợp nhất
            if best_result["stock_idx"] != -1:
                return {
                    "size": best_result["size"],
                    "stock_idx": best_result["stock_idx"],
                    "position": best_result["position"],
                }

        # Không tìm thấy vị trí phù hợp
        return {
            "size": [0, 0],
            "stock_idx": -1,
            "position": (None, None)
        }
        
    # Student code here
    # You can add more functions if needed

    