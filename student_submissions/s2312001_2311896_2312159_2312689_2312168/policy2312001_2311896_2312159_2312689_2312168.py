from policy import Policy


class Policy2312001_2311896_2312159_2312689_2312168(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        
        self.policy_id = policy_id
        
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.best_fit(observation,info)
        elif self.policy_id == 2:
            return self.hff(observation,info)
        pass

    # Student code here
    def best_fit(self, observation, info):
        """
        Trả về hành động cắt: {"stock_idx": int, "size": [w, h], "position": (x, y)}
        """
        # Sắp xếp sản phẩm theo diện tích giảm dần
        list_prods = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda p: p["size"][0] * p["size"][1],  # Diện tích sản phẩm
            reverse=True,
        )

        # Duyệt qua từng sản phẩm
        
        
        for prod in list_prods:
            prod_size = prod["size"]  # Kích thước sản phẩm [w, h]
            prod_w, prod_h = prod_size
            
            best_x, best_y = None, None
            best_waste = float('inf')  # Lãng phí nhỏ nhất
            best_stock_idx = None
            best_rotation = None
            # Duyệt qua từng tấm
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                for rotation in [(prod_w, prod_h), (prod_h, prod_w)]:
                    rotated_prod_w, rotated_prod_h = rotation
                    # Kiểm tra nếu sản phẩm vừa với tấm
                    if stock_w >= rotated_prod_w and stock_h >= rotated_prod_h:
                        for x in range(stock_w - rotated_prod_w + 1):
                            for y in range(stock_h - rotated_prod_h + 1):
                                if self._can_place_(stock, (x, y), (rotated_prod_w, rotated_prod_h)):
                                    # Tính toán lãng phí trực tiếp
                                    remaining_w = stock_w - rotated_prod_w
                                    remaining_h = stock_h - rotated_prod_h
                                    waste = remaining_w * remaining_h

                                    # Chọn tấm có lãng phí nhỏ nhất và ưu tiên vị trí (x, y)
                                    if waste < best_waste or (waste == best_waste and (best_y is None or y < best_y or (y == best_y and x < best_x))):
                                        best_x, best_y = x, y
                                        best_waste = waste
                                        best_stock_idx = stock_idx
                                        best_rotation = rotation  # Cập nhật kiểu xoay của sản phẩm
                                       
            # Nếu tìm được vị trí tốt nhất cho sản phẩm này, trả về hành động
            if best_x is not None and best_y is not None:
                return {
                    "stock_idx": best_stock_idx,
                    "size": best_rotation,
                    "position": (best_x, best_y),
                }

        # Nếu không thể cắt được, trả về hành động không hợp lệ
        pass
    
    def hff(self, observation, info):
        """
        Trả về hành động cắt: {"stock_idx": int, "size": [w, h], "position": (x, y)}
        """
        # Bước 1: Sắp xếp sản phẩm theo diện tích giảm dần
        list_prods = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda p: p["size"][0] * p["size"][1],  # Diện tích sản phẩm
            reverse=True,
        )

        # Bước 2: Lặp qua từng sản phẩm
        for prod in list_prods:
            prod_size = prod["size"]  # Kích thước sản phẩm [w, h]
            prod_w, prod_h = prod_size

            # Bước 3: Duyệt qua các tấm để tìm tấm đầu tiên phù hợp
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Bước 4: Tìm vị trí đặt theo chiến lược kết hợp
                best_x, best_y = None, None
                best_waste = float('inf')  # Lãng phí nhỏ nhất
                best_rotation = None

                for rotation in [(prod_w, prod_h), (prod_h, prod_w)]:
                    rotated_prod_w , rotated_prod_h = rotation
                    # Kiểm tra nếu sản phẩm vừa với tấm
                    if stock_w < rotated_prod_w or stock_h < rotated_prod_h:
                        continue
                    for x in range(stock_w - rotated_prod_w + 1):
                        for y in range(stock_h - rotated_prod_h + 1):
                            if self._can_place_(stock, (x, y), (rotated_prod_w, rotated_prod_h)):
                                # Chiến lược Hybrid:
                                # Tính lãng phí (waste) dựa trên diện tích còn trống sau khi đặt sản phẩm
                                remaining_w = stock_w - rotated_prod_w
                                remaining_h = stock_h - rotated_prod_h
                                waste = remaining_w * remaining_h

                                # Ưu tiên vị trí (x, y) thấp nhất và giảm lãng phí
                                if waste < best_waste or (waste == best_waste and (best_y is None or y < best_y or (y == best_y and x < best_x))):
                                    best_x, best_y = x, y
                                    best_waste = waste
                                    best_rotation = rotation

                # Nếu tìm được vị trí tốt nhất, trả về hành động
                if best_x is not None and best_y is not None:
                    return {
                        "stock_idx": stock_idx,
                        "size": best_rotation,
                        "position": (best_x, best_y),
                    }

        # Nếu không tìm được vị trí phù hợp, trả về hành động không hợp lệ
        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
    # You can add more functions if needed
