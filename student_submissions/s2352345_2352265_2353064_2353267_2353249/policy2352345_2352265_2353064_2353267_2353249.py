from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2352345_2352265_2353064_2353267_2353249(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = ColumnGenerationPolicy()
        elif policy_id == 2:
            self.policy = HeuristicLFPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed
class ColumnGenerationPolicy(Policy):
    def __init__(self, max_iterations=100):
        super().__init__()
        self.current_patterns = []  # Danh sách các pattern hiện tại
        self.demand_met = False     # Cờ kiểm tra đáp ứng nhu cầu
        self.max_iterations = max_iterations  # Giới hạn số lần lặp
        self.A = None  # Khởi tạo thuộc tính A

    def get_action(self, observation, info):
        try:
            # Kiểm tra tính hợp lệ của observation
            if not observation or not isinstance(observation, dict):
                return self._create_default_action()

            # Kiểm tra keys cần thiết
            required_keys = ['products', 'stocks']
            for key in required_keys:
                if key not in observation:
                    return self._create_default_action()

            # Lấy sản phẩm và stock
            products = observation['products']
            stocks = observation['stocks']

            # Kiểm tra tính hợp lệ của products và stocks
            if not products or not stocks:
                return self._create_default_action()

            # Tìm sản phẩm chưa được đáp ứng đủ
            for product in products:
                # Bỏ qua sản phẩm đã đáp ứng đủ
                if product['quantity'] <= 0:
                    continue

                # Lấy kích thước sản phẩm
                prod_size = product['size']
                prod_w, prod_h = prod_size

                # Tìm stock phù hợp
                for stock_idx, stock in enumerate(stocks):
                    # Xác định kích thước thực tế của stock
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Kiểm tra kích thước stock có phù hợp không
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Tìm vị trí có thể đặt sản phẩm
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            # Kiểm tra có thể đặt sản phẩm không
                            if self._can_place_(stock, (x, y), prod_size):
                                # Tạo action
                                action = {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y)
                                }
                                
                                return action

            # Nếu không tìm được action phù hợp
            return self._create_default_action()

        except Exception as e:
            # Trả về action mặc định
            return self._create_default_action()

    def _create_default_action(self):
        """
        Tạo action mặc định an toàn
        """
        return {
            "stock_idx": 0,
            "size": (1, 1),
            "position": (0, 0)
        }

    def _initialize_patterns(self, observation):
        """
        Khởi tạo các pattern ban đầu dựa trên kích thước sản phẩm và stock
        """
        try:
            products = observation.get("products", [])
            stocks = observation.get("stocks", [])
            initial_patterns = []

            for prod in products:
                prod_w, prod_h = prod.get("size", (0, 0))
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Kiểm tra khả năng đặt sản phẩm
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pattern = {
                            "stock_idx": stock_idx,
                            "size": prod["size"],
                            "position": (0, 0),
                            "quantity": prod["quantity"]
                        }
                        initial_patterns.append(pattern)
            
            return initial_patterns
        
        except Exception as e:
            return []

    def _solve_subproblem(self, observation, dual_prices):
        """
        Sinh pattern mới bằng cách giải bài toán phụ (subproblem)
        """
        try:
            products = observation.get("products", [])
            stocks = observation.get("stocks", [])
            
            best_pattern = None
            min_reduced_cost = float('inf')

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                
                for prod in products:
                    prod_w, prod_h = prod.get("size", (0, 0))
                    
                    # Kiểm tra khả năng đặt sản phẩm
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    max_items_x = stock_w // prod_w
                    max_items_y = stock_h // prod_h
                    total_items = max_items_x * max_items_y

                    # Tính reduced cost
                    reduced_cost = 1 - sum(
                        dual_prices[i] * total_items 
                        for i in range(len(products))
                    )
                    
                    # Cập nhật pattern tốt nhất
                    if reduced_cost < min_reduced_cost:
                        min_reduced_cost = reduced_cost
                        best_pattern = {
                            "stock_idx": stock_idx,
                            "size": prod["size"],
                            "position": (0, 0),
                            "quantity": total_items
                        }

            return best_pattern if min_reduced_cost < 0 else None
        
        except Exception as e:
            return None

    def _solve_master_problem(self, observation):
        """
        Giải bài toán chính (master problem) sử dụng quy hoạch tuyến tính
        """
        try:
            num_products = len(observation.get("products", []))
            num_patterns = len(self.current_patterns)

            # Vector chi ```python
            # phí (mỗi pattern sử dụng 1 stock)
            c = np.ones(num_patterns)  
            
            # Ma trận ràng buộc
            self.A = np.zeros((num_products, num_patterns))
            
            # Vector nhu cầu
            b = np.array([prod["quantity"] for prod in observation.get("products", [])])

            # Điền ma trận ràng buộc
            for j, pattern in enumerate(self.current_patterns):
                for i, prod in enumerate(observation.get("products", [])):
                    if np.array_equal(prod["size"], pattern["size"]):
                        self.A[i, j] = pattern["quantity"]

            # Giải quyết bằng phương pháp linprog
            result = linprog(
                c, 
                A_eq=self.A, 
                b_eq=b, 
                bounds=(0, None), 
                method='highs'
            )
            
            if result.success:
                return result
            else:
                return None
        
        except Exception as e:
            return None

    def _adjust_solution(self, lp_solution, observation):
        """
        Điều chỉnh nghiệm để đáp ứng các ràng buộc nguyên
        """
        try:
            # Làm tròn nghiệm LP xuống số nguyên
            rounded_solution = np.floor(lp_solution.x).astype(int)
            
            # Tính toán nhu cầu chưa được đáp ứng
            unmet_demand = np.array([
                prod["quantity"] for prod in observation.get("products", [])
            ]) - np.dot(self.A, rounded_solution)

            # Điều chỉnh để đáp ứng nhu cầu
            for i in range(len(unmet_demand)):
                if unmet_demand[i] > 0:
                    for j in range(len(rounded_solution)):
                        if self.A[i, j] > 0:
                            rounded_solution[j] += unmet_demand[i] // self.A[i, j]
                            unmet_demand[i] = 0
                            break

            return rounded_solution
        
        except Exception as e:
            return None
        
class HeuristicLFPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info):
        # Kiểm tra tính hợp lệ của observation
        if not observation or not isinstance(observation, dict):
            return self._create_default_action()

        # Lấy sản phẩm và sắp xếp theo diện tích giảm dần
        products = observation.get('products', [])
        stocks = observation.get('stocks', [])
        if not products or not stocks:
            return self._create_default_action()

        products = list(products)  # Chuyển tuple thành list
        products.sort(key=lambda p: p['size'][0] * p['size'][1], reverse=True)


        # Duyệt qua từng sản phẩm theo thứ tự sắp xếp
        for product in products:
            if product['quantity'] <= 0:
                continue  # Bỏ qua sản phẩm đã đáp ứng đủ

            prod_size = product['size']
            prod_w, prod_h = prod_size

            # Duyệt qua từng kho
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Kiểm tra kích thước kho có đủ không
                if stock_w < prod_w or stock_h < prod_h:
                    continue

                # Tìm vị trí có thể đặt sản phẩm
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            # Tạo action
                            action = {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (x, y)
                            }
                            return action  # Trả về action đầu tiên tìm thấy

        # Nếu không tìm được action phù hợp
        return self._create_default_action()

    def _create_default_action(self):
        """
        Tạo action mặc định an toàn
        """
        return {
            "stock_idx": 0,
            "size": (1, 1),
            "position": (0, 0)
        }
