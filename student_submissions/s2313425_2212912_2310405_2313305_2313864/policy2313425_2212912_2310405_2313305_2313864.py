from policy import Policy


class policy2313425_2212912_2310405_2313305_2313864(Policy):
    def __init__(self):
        # Student code here
        pass

    def get_action(self, observation, info):
        # Student code here
        list_prods = observation["products"]

        # Khởi tạo giá trị mặc định
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        min_waste = float('inf')
        best_action = None

        # Bước 1: Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_prods = sorted(
            [(i, prod) for i, prod in enumerate(list_prods) if prod["quantity"] > 0],
            key=lambda x: (x[1]["size"][0] * x[1]["size"][1]), 
            reverse=True
        )

        if not sorted_prods:
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        # Duyệt qua từng sản phẩm đã sắp xếp
        for prod_idx, prod in sorted_prods:
            # Xem xét cả hai hướng của sản phẩm (gốc và xoay)
            orientations = [prod["size"]]
            if prod["size"][0] != prod["size"][1]:  # Chỉ thêm hướng xoay nếu không phải hình vuông
                orientations.append([prod["size"][1], prod["size"][0]])

            # Duyệt qua từng tấm vật liệu
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Thử từng hướng của sản phẩm
                for current_size in orientations:
                    prod_w, prod_h = current_size

                    # Bỏ qua nếu kích thước không phù hợp
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Tối ưu: Chỉ thử đặt ở các vị trí có khả năng
                    for x in range(0, stock_w - prod_w + 1, 1):  # Có thể tăng bước nhảy để tăng tốc
                        for y in range(0, stock_h - prod_h + 1, 1):
                            if not self._can_place_(stock, (x, y), current_size):
                                continue

                            # Tính toán lượng hao hụt
                            used_area = prod_w * prod_h
                            total_area = stock_w * stock_h
                            waste = total_area - used_area

                            # Cập nhật nếu tìm thấy giải pháp tốt hơn
                            if waste < min_waste:
                                min_waste = waste
                                best_action = {
                                    "stock_idx": i,
                                    "size": current_size,
                                    "position": (x, y)
                                }
                                
                                # Tối ưu: Nếu tìm được giải pháp đủ tốt, có thể dừng sớm
                                if waste < total_area * 0.1:  # Ngưỡng hao hụt 10%
                                    return best_action

        return best_action if best_action else {
            "stock_idx": stock_idx,
            "size": prod_size,
            "position": (pos_x, pos_y)
        }

    # Student code here
    # You can add more functions if needed
