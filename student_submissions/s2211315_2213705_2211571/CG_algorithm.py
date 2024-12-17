from policy import Policy
import numpy as np
from scipy.optimize import linprog


class ColunmGeneraton(Policy):
    def __init__(self):
        """
        Khởi tạo policy.
        """
        super().__init__()

    def _can_place_(self, stock, position, prod_size):
        """
        Kiểm tra xem sản phẩm có thể được đặt tại một vị trí cụ thể trên vật liệu không.
        :param stock: Mảng numpy đại diện cho vật liệu hiện có.
        :param position: Vị trí (x, y) trên vật liệu để đặt sản phẩm.
        :param prod_size: Kích thước sản phẩm (length, width).
        :return: True nếu có thể đặt sản phẩm tại vị trí, ngược lại False.
        """
        pos_x, pos_y = position                  # `pos_x` và `pos_y` là tọa độ góc trên bên trái của sản phẩm trên vật liệu.
        prod_length, prod_width = prod_size      # `prod_length` và `prod_width` là chiều dài và chiều rộng của sản phẩm.
        stock_length, stock_width = stock.shape  # `stock_length` và `stock_width` là kích thước của vật liệu.

        if pos_x + prod_length > stock_length or pos_y + prod_width > stock_width:  # Kiểm tra nếu sản phẩm tràn ra ngoài giới hạn của vật liệu
            return False

        try:
            region = stock[pos_x:pos_x + prod_length, pos_y:pos_y + prod_width]  # Lấy vùng trên vật liệu tương ứng với vị trí sản phẩm
            return np.all(region == -1)  # Kiểm tra toàn bộ vùng có trống hay không
        except IndexError:
            return False # Trả về False nếu xảy ra lỗi IndexError (tràn ra ngoài mảng numpy)



    def _generate_initial_basis(self, products, roll_length, roll_width):
        """
        Sinh các mẫu cơ sở ban đầu từ các sản phẩm.
        :param products: Danh sách các sản phẩm cần cắt.
        :param roll_length: Chiều dài của vật liệu.
        :param roll_width: Chiều rộng của vật liệu.
        :return: Ma trận cơ sở ban đầu.
        """
        if len(products) == 0:
            print("Products list is empty in _generate_initial_basis.")
            raise ValueError("Danh sách sản phẩm trống.")


        basis = []   # Khởi tạo danh sách lưu trữ các mẫu cơ sở
        for i, prod in enumerate(products):  # Lặp qua từng sản phẩm trong danh sách sản phẩm
            prod_length, prod_width = prod["size"]  # Lấy kích thước của sản phẩm (length, width).
            # Tính số lượng tối đa sản phẩm có thể cắt theo chiều dài và chiều rộng của vật liệu
            max_count_length = roll_length // prod_length
            max_count_width = roll_width // prod_width
            max_count = max_count_length * max_count_width  # Tính tổng số lượng sản phẩm có thể cắt từ vật liệu
            pattern = [0] * len(products)   # Tạo một mẫu cơ sở với tất cả các giá trị ban đầu là 0
            pattern[i] = max_count   # Gán số lượng tối đa của sản phẩm `i` vào vị trí tương ứng trong mẫu cơ sở
            basis.append(pattern)    # Thêm mẫu cơ sở này vào danh sách
        return np.array(basis).T     # Chuyển danh sách các mẫu cơ sở thành numpy array và chuyển vị để phù hợp định dạng



    def _solve_master_problem(self, B, demands):
        """
        Giải bài toán chính (master problem) để tìm dual_values và chi phí giảm.
        :param B: Ma trận cơ sở.
        :param demands: Danh sách số lượng sản phẩm cần cắt.
        :return: Kết quả bài toán master và giá trị đối ngẫu.
        """
        if B.size == 0 or len(demands) == 0:    # Kiểm tra đầu vào: nếu ma trận B hoặc demands rỗng, báo lỗi.
            raise ValueError("Matrix B hoặc demands bị trống.")

        B = np.atleast_2d(B)    # Đảm bảo B là mảng 2 chiều.
        c = np.array([1] * B.shape[1], dtype=float)     # Hàm mục tiêu: mảng toàn số 1 với kích thước bằng số cột của B.
        A_ub = -B.T.astype(float)                       # Chuyển vị của B và đổi dấu để phù hợp với dạng bất đẳng thức.
        b_ub = -np.array(demands, dtype=float)          # Chuyển demands sang dạng âm để phù hợp với dạng bất đẳng thức.

        # In thông tin kích thước để debug
       # print(f"Trước đồng bộ: A_ub {A_ub.shape}, b_ub {b_ub.shape}, c {len(c)}")

        # Đồng bộ A_ub và c
        if A_ub.shape[1] != len(c):
            if A_ub.shape[1] > len(c):  # Nếu số cột của A_ub lớn hơn độ dài của c, bổ sung các số 0 vào c.
                c = np.append(c, [0] * (A_ub.shape[1] - len(c)))
            else:   # Nếu số cột của A_ub nhỏ hơn độ dài của c, bổ sung các cột 0 vào A_ub.
                A_ub = np.hstack((A_ub, np.zeros((A_ub.shape[0], len(c) - A_ub.shape[1]))))

        # Đồng bộ A_ub và b_ub
        if A_ub.shape[0] != len(b_ub):
            if A_ub.shape[0] > len(b_ub):   # Nếu số hàng của A_ub lớn hơn độ dài của b_ub, bổ sung các số 0 vào b_ub.
                b_ub = np.append(b_ub, [0] * (A_ub.shape[0] - len(b_ub)))
            else:   # Nếu số hàng của A_ub nhỏ hơn độ dài của b_ub, bổ sung các hàng 0 vào A_ub.
                A_ub = np.vstack((A_ub, np.zeros((len(b_ub) - A_ub.shape[0], A_ub.shape[1]))))

        # In thông tin kích thước sau khi đồng bộ
       # print(f"Sau đồng bộ: A_ub {A_ub.shape}, b_ub {b_ub.shape}, c {len(c)}")

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')      # Giải bài toán tuyến tính bằng phương pháp "highs".

        if res.success:     # Nếu bài toán giải thành công.
            dual_values = res.slack if hasattr(res, 'slack') else np.zeros_like(b_ub)       # Lấy giá trị đối ngẫu.
            if len(dual_values) != len(b_ub):
                # print(f"Đồng bộ dual_values: {len(dual_values)} -> {len(b_ub)}")
                if len(dual_values) < len(b_ub):        # Nếu dual_values ngắn hơn, bổ sung các số 0.
                    dual_values = np.append(dual_values, [0] * (len(b_ub) - len(dual_values)))
                else:   # Nếu dual_values dài hơn, cắt bớt.
                    dual_values = dual_values[:len(b_ub)]

            return res, dual_values     # Trả về kết quả của bài toán và giá trị đối ngẫu.
        else:
            raise ValueError(f"Master Problem không hội tụ. Thông tin lỗi: {res.message}")




    def _solve_knapsack(self, roll_length, roll_width, products, dual_values):
        """
        Giải bài toán túi đồ (knapsack) để tìm mẫu cắt mới tối ưu.
        :param roll_length: Chiều dài vật liệu.
        :param roll_width: Chiều rộng vật liệu.
        :param products: Danh sách sản phẩm cần cắt.
        :param dual_values: Giá trị đối ngẫu từ bài toán chính.
        :return: Mẫu cắt mới và chi phí giảm.
        """
        best_pattern = [0] * len(products)  # Khởi tạo mẫu cắt tối ưu ban đầu (tất cả bằng 0).
        best_reduced_cost = float('-inf')   # Khởi tạo chi phí giảm tốt nhất là âm vô cực.

        # Đồng bộ dual_values nếu cần
        if len(dual_values) != len(products):
            # print(f"Đồng bộ dual_values trong _solve_knapsack: {len(dual_values)} -> {len(products)}")
            if len(dual_values) < len(products):    # Nếu dual_values ngắn hơn, bổ sung các số 0.
                dual_values = np.append(dual_values, [0] * (len(products) - len(dual_values)))
            else:   # Nếu dual_values dài hơn, cắt bớt. Chỉ giữ lại các giá trị đầu tiên tương ứng với thứ tự của sản phẩm.
                dual_values = dual_values[:len(products)]
        # Lặp qua từng sản phẩm để xác định mẫu cắt tối ưu.
        for i, prod in enumerate(products):
            prod_length, prod_width = prod["size"]  # Lấy kích thước sản phẩm (dài và rộng).
            max_count_length = roll_length // prod_length   # Số lượng tối đa theo chiều dài.
            max_count_width = roll_width // prod_width      # Số lượng tối đa theo chiều rộng.
            max_count = max_count_length * max_count_width  # Tổng số lượng tối đa có thể cắt từ vật liệu.
            reduced_cost = 1 - np.dot(dual_values, [max_count if j == i else 0 for j in range(len(products))]) # Tính toán chi phí giảm (reduced cost) của sản phẩm này.
            # Cập nhật mẫu cắt tốt nhất nếu chi phí giảm của sản phẩm này lớn hơn chi phí tốt nhất hiện tại.
            if reduced_cost > best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_pattern = [max_count if j == i else 0 for j in range(len(products))]

        return best_pattern, best_reduced_cost     # Trả về mẫu cắt tối ưu và chi phí giảm tương ứng.


    def get_action(self, observation, info):
        """
        Tìm hành động tối ưu dựa trên thuật toán Column Generation.
        :param observation: Trạng thái hiện tại của môi trường.
        :param info: Thông tin bổ sung từ môi trường.
        :return: Hành động tối ưu.
        """
        # Lấy danh sách các stocks và products từ observation
        stocks = observation.get("stocks", None)
        products = observation.get("products", None)
        # Kiểm tra tính hợp lệ của observation
        if not stocks or not products:
            print("Invalid observation: Missing stocks or products.")
            return {"stock_idx": 0, "size": [0, 0], "position": [0, 0]}  # Hành động mặc định
        # Lấy kích thước của vật liệu đầu tiên (roll_length và roll_width)
        roll_length, roll_width = stocks[0].shape
        # Tạo danh sách số lượng sản phẩm cần cắt
        demands = [prod["quantity"] for prod in products]
        # Sinh cơ sở ban đầu (ma trận B)
        B = self._generate_initial_basis(products, roll_length, roll_width)
        previous_reduced_cost = float('inf')
        # Vòng lặp giải bài toán chính và bài toán con
        while True:
            try:
                # Giải bài toán chính để tìm dual_values
                res, dual_values = self._solve_master_problem(B, demands)
            except ValueError as e:
                print(f"Lỗi khi giải Master Problem: {str(e)}")
                break
            # Giải bài toán túi đồ để tìm mẫu cắt mới
            new_pattern, reduced_cost = self._solve_knapsack(roll_length, roll_width, products, dual_values)
            # Kiểm tra điều kiện dừng khi không thể cải thiện
            if reduced_cost <= 0:
                break
            # Nếu chi phí giảm không thay đổi đáng kể, thoát khỏi vòng lặp
            if abs(reduced_cost - previous_reduced_cost) < 1e-6:
                break
            # Cập nhật chi phí giảm trước đó
            previous_reduced_cost = reduced_cost
            # Chuyển đổi mẫu cắt mới thành mảng numpy và đồng bộ với ma trận B
            new_pattern = np.array(new_pattern).reshape(-1, 1)
            # Kiểm tra và điều chỉnh kích thước của ma trận B để đồng bộ với new_pattern
            if B.shape[0] != new_pattern.shape[0]:
                if B.shape[0] > new_pattern.shape[0]:
                    new_pattern = np.vstack((new_pattern, np.zeros((B.shape[0] - new_pattern.shape[0], 1))))
                else:
                    B = np.vstack((B, np.zeros((new_pattern.shape[0] - B.shape[0], B.shape[1]))))
            # Gộp new_pattern vào ma trận B
            B = np.hstack((B, new_pattern))
        
            # Tìm hành động khả thi dựa trên trạng thái hiện tại của stocks và products
            for stock_idx, stock in enumerate(stocks):
                for prod_idx, prod in enumerate(products):
                    prod_length, prod_width = prod["size"]
                    for x in range(roll_length - prod_length + 1):  # Duyệt theo trục x
                        for y in range(roll_width - prod_width + 1): # Duyệt theo trục y
                            # Kiểm tra xem sản phẩm có thể đặt tại vị trí (x, y) hay không
                            if self._can_place_(stock, (x, y), (prod_length, prod_width)) and prod["quantity"] > 0:
                                # Trả về hành động khi tìm thấy vị trí hợp lệ
                                return {
                                    "stock_idx": stock_idx,
                                    "size": (prod_length, prod_width),
                                    "position": (x, y),
                                }

        # Nếu không tìm thấy hành động, trả về giá trị mặc định
        print("No valid action found. Returning default action.")
        return {"stock_idx": -1, "size": [0, 0], "position": [0, 0]}