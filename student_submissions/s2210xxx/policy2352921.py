from policy import Policy
import numpy as np
from itertools import product

class Policy2352921_2353283_2352129_2352659_2352948_1(Policy):
    def __init__(self):
        self.G_max = 1
        self.alpha = 0.7
        self.beta = 1.02

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        # Initialize parameters
        G = 1
        N_pl_best = float("inf")
        values = [p["size"][0] * p["size"][1] for p in products]
        remains = [p["quantity"] for p in products]
        demands = remains[:]
        lengths = [p["size"][1] for p in products]
        widths = [p["size"][0] for p in products]
        temp = lengths[:]
        lengths += widths
        widths += temp
        best_plan = {"stock_idx": -1, "patterns": [], "frequencies": [], "plates": float("inf")}
        for stock_idx, stock in enumerate(stocks):
            while G <= self.G_max:
                stock_w, stock_h = self._get_stock_size_(stock)
                P = []
                F = []
                r_i = remains[:]
                while sum(r_i) > 0:
                    F_2SGP, N_2SGP, _, _, _ = self.generate_2SGP(
                        len(products), stock_h, stock_w, values, lengths, widths, r_i
                    )
                    F_3SHP, N_3SHP, _, _ = self.generate_3SHP(
                        len(products), stock_h, stock_w, values, lengths, widths, r_i
                    )
                    if F_2SGP >= F_3SHP:
                        candidate_pattern = N_2SGP
                    else:
                        candidate_pattern = N_3SHP
                    frequency = min(
                        r_i[i] // candidate_pattern[i] if candidate_pattern[i] > 0 else float("inf")
                        for i in range(len(products))
                    )
                    P.append(candidate_pattern)
                    F.append(frequency)
                    for i in range(len(products)):
                        r_i[i] -= frequency * candidate_pattern[i]
                    if sum(r_i) > 0:
                        values = self.correction_value(
                            len(products), stock_h, stock_w, values, lengths, widths, demands, r_i, candidate_pattern
                        )
                N_pl_cur = len(F)
                if N_pl_cur < N_pl_best:
                    N_pl_best = N_pl_cur
                    best_plan = {"stock_idx": stock_idx, "patterns": P, "frequencies": F, "plates": N_pl_best}
                G += 1
        if best_plan["patterns"]:
            for pattern, freq in zip(best_plan["patterns"], best_plan["frequencies"]):
                for i, quantity in enumerate(pattern):
                    if quantity > 0 and freq > 0:
                        size = products[i]["size"]
                        stock = stocks[best_plan["stock_idx"]]
                        position = self.find_position(stock, size)
                        if position:
                            return {
                                "stock_idx": best_plan["stock_idx"],
                                "size": size,
                                "position": position,
                            }
        return {"stock_idx": -1, "size": [0, 0], "position": [0, 0]}        
        

    def find_position(self, stock, size):
        stock_h, stock_w = stock.shape
        item_h, item_w = size

        for x in range(stock_h - item_h + 1):
            for y in range(stock_w - item_w + 1):
                if self._can_place_(stock, (x, y), size):
                    stock[x : x + item_h, y : y + item_w] = 1
                    return [x, y]

        return None

    def generate_2SGP(self, m, L, W, values, lengths, widths, remains):
        """
        Generate the maximum value for 2SGP using the recurrence relation.

        Parameters:
        m (int): Number of item types.
        L (int): Length of the plate.
        W (int): Width of the plate.
        values (list): List of values for each item type [v1, v2, ..., vm].
        lengths (list): List of lengths for each item type [l1, l2, ..., lm].
        widths (list): List of widths for each item type [w1, w2, ..., wm].
        remains (list): List of remaining demand for each item type [r1, r2, ..., rm].

        Returns:
        F_2SGP
        N_2SGP
        V_2SGP
        """
        K = 2 * m
        # Initialize DP tables
        f_2SGP = [[0] * (L + 1) for _ in range(K + 1)]
        n_2SGP = [[[0] * m for _ in range(L + 1)] for _ in range(K + 1)]

        # Compute X1 using itertools.product
        X1 = set()
        ranges = [range(L // l_k + 1) for l_k in lengths]  # Create range for each l_k

        for e in product(*ranges):  # Cartesian product of ranges
            x = sum(e_k * lengths[i] for i, e_k in enumerate(e))  # Compute the sum
            if x <= L:
                X1.add(x)

        # Generate strips
        for k in range(1, K + 1):
            for x in X1:
                max_value = f_2SGP[k - 1][x]
                best_combination = [0] * m

                for i in range(m):
                    l_k = lengths[k - 1]

                    if l_k <= x and remains[i] > 0:
                        temp_value = values[i] + f_2SGP[k][x - l_k]
                        if temp_value > max_value:
                            max_value = temp_value
                            best_combination = n_2SGP[k][x - l_k][:]  # Copy previous
                            best_combination[i] += 1

                f_2SGP[k][x] = max_value
                n_2SGP[k][x] = best_combination[:]

        # Initialize DP tables
        F_2SGP = [0] * (W + 1)  # DP table for maximum values across widths
        N_2SGP = [[0] * m for _ in range(W + 1)]  # Counts for each item type
        V_2SGP = [0] * (K + 1)

        # Compute Y1 using itertools.product
        Y1 = set()
        ranges = [range(W // w_k + 1) for w_k in widths]  # Create range for each w_k

        for e in product(*ranges):  # Cartesian product of ranges
            y = sum(e_k * widths[i] for i, e_k in enumerate(e))  # Compute the sum
            if y <= W:
                Y1.add(y)

        # Iterate over segment widths
        for y in Y1:
            max_value = F_2SGP[y - 1]  # Case 1: No new strip added
            best_combination = N_2SGP[y - 1][:]

            for k in range(1, K + 1):  # Case 2: Add a strip
                strip_value = 0
                strip_items = [0] * m

                # w_k = k if k <= m else k - m  # Width of the strip
                w_k = widths[k - 1] 

                if w_k <= y:
                    for i in range(m):
                        num_items = min(
                            n_2SGP[k][L][i],
                            remains[i] - N_2SGP[y - w_k][i]
                        )
                        strip_items[i] = num_items
                        strip_value += num_items * values[i]
                    
                    V_2SGP[k] = strip_value 
                        
                    temp_value = strip_value + F_2SGP[y - w_k]
                    if temp_value > max_value:
                        max_value = temp_value
                        best_combination = N_2SGP[y - w_k][:]
                        for i in range(m):
                            best_combination[i] += strip_items[i]

            # Update DP tables
            F_2SGP[y] = max_value
            N_2SGP[y] = best_combination[:]

        return F_2SGP[W], N_2SGP[W], V_2SGP[K], f_2SGP, n_2SGP

    def generate_3SHP(self, m, L, W, values, lengths, widths, remains):
        """
        Parameters:
        m
        L
        W
        values
        lengths
        widths
        remains

        Returns:
        F_3SHP
        N_3SHP
        V_3SHP
        T_3SHP
        """
        # Compute X2
        X2 = set()

        for k in range(len(lengths)):  # Loop over all l_k
            for e_k in range(L // lengths[k] + 1):  # Loop over valid e_k values
                x = e_k * lengths[k]
                X2.add(x)

        # Compute Y2
        Y2 = set()

        for k in range(len(widths)):  # Loop over all w_k
            for e_k in range(W // widths[k] + 1):  # Loop over valid e_k values
                y = e_k * widths[k]
                Y2.add(y)

        K = 2 * m
        F_3SHP = [[0] * (W + 1) for _ in range(L + 1)]
        n_3SHP = [[[0] * m for _ in range(L + 1)] for _ in range(K + 1)]  
        N_3SHP = [[[0] * m for _ in range(W + 1)] for _ in range(L + 1)]
        V_3SHP = [[0] * (L + 1) for _ in range(K + 1)]

        for k in range(1, K + 1):
            for x in X2:
                for i in range(m):
                    n_3SHP[k][x][i] = min(
                        x // lengths[i],
                        remains[i]
                    )

        # Generate homogeneous strips and segments
        for x in X2:
            for y in Y2:
                max_value = F_3SHP[x][y - 1]
                best_combination = N_3SHP[x][y - 1][:]

                for k in range(1, K + 1):
                    strip_value = 0
                    strip_items = [0] * m
                    w_k = widths[k - 1]

                    if w_k <= y:
                        for i in range(m):
                            num_items = min(
                                n_3SHP[k][x][i],
                                remains[i] - N_3SHP[x][y - w_k][i]
                            )
                            strip_items[i] = num_items
                            strip_value += num_items * values[i]

                        V_3SHP[k][x] = strip_value

                        temp_value = strip_value + F_3SHP[x][y - w_k]
                        if temp_value > max_value:
                            max_value = temp_value
                            best_combination = N_3SHP[x][y - w_k][:]
                            for i in range(m):
                                best_combination[i] += strip_items[i]

                # Update DP tables
                F_3SHP[x][y] = max_value
                N_3SHP[x][y] = best_combination[:]

        # Solve for 3SHP
        T_3SHP = 0
        for x in X2:
            T_3SHP = max(T_3SHP, F_3SHP[x][W] + F_3SHP[L - x][W])

        return F_3SHP[L][W], N_3SHP[L][W], V_3SHP[K][L], T_3SHP
    
    def correction_value(self, m, L, W, values, lengths, widths, demands, remains, n):
        """
        Parameters:
        m
        L
        W
        values
        widths
        demands
        remains
        n

        Returns:
        v_new
        """
        u = sum(n[i] * lengths[i] * widths[i] for i in range(m)) / (L * W)

        v_new = [0] * m

        for i in range(m):
            if n[i] == 0:
                g2 = 0
                g1 = 1
            else:
                g2 = self.alpha * n[i] / (demands[i] + remains[i])
                g1 = 1 - g2

            v_i = g1 * values[i] + g2 * ((lengths[i] * widths[i]) ** self.beta) / u
            v_new[i] = v_i
        
        return v_new
    
class Policy2352921_2353283_2352129_2352659_2352948_2(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        self.num_products = len(products)
        # Sort products by area (width * height) in decreasing order
        sorted_products = sorted(products, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        for product in sorted_products:
            if product["quantity"] <= 0:
                continue
            original_size = product["size"]
            orientations = [original_size, [original_size[1], original_size[0]]]
            for stock_idx, stock in enumerate(stocks):
                for size in orientations:
                    position = self.find_first_fit(stock, size)
                    if position is not None:
                        return {
                            "stock_idx": stock_idx,
                            "size": size,
                            "position": position,
                        }
            return {
                "stock_idx": 0,
                "size": original_size,
                "position": [0, 0],
            }
        # If all products are placed
        return {"stock_idx": 0, "size": [0, 0], "position": [0, 0]}
    
    def find_first_fit(self, stock, size):
        """
        Find the first position in the stock where the product of 'size' can fit.
        Returns the position [x, y] or None if it doesn't fit.
        """
        stock_width = np.sum(np.any(stock != -2, axis=1))
        stock_height = np.sum(np.any(stock != -2, axis=0))
        width, height = size

        for x in range(stock_width - width + 1):
            for y in range(stock_height - height + 1):
                if np.all(stock[x : x + width, y : y + height] == -1):
                    return [x, y]
        return None

    def select_best_action(self, stocks, products):
        for stock_idx, stock in enumerate(stocks):
            stock_width = np.sum(np.any(stock != -2, axis=1))
            stock_height = np.sum(np.any(stock != -2, axis=0))

            for product in products:
                size = product["size"]
                quantity = product["quantity"]
                if quantity > 0:
                    orientations = [size, [size[1], size[0]]]
              
                    for orientation in orientations:
                        width, height = orientation
                        for x in range(stock_width - width + 1):
                            for y in range(stock_height - height + 1):
                                if np.all(stock[x : x + width, y : y + height] == -1):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": orientation,
                                        "position": [x, y],
                                    }

        return {"stock_idx": 0, "size": [0, 0], "position": [0, 0]}