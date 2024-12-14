import numpy as np

from policy import Policy


class Policy2313962_2313966_2310207_2312371_2311145(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here
        super().__init__()
        self.policy_id = policy_id
        if policy_id == 1:
            self.ListOpt = []
            self.currentIndex = 0
        elif policy_id == 2:
            self.lastStockIndex = None
            self.usedStocks = set()

    # Policy 1
    # Improved Priority Heuristic policy
    def evaluatePriority(self, stock_W, stock_H, observation):
        list_prods = observation["products"]
        prioritized_list = []
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                if prod_w == stock_W and prod_h == stock_H:
                    priority_1 = 4
                elif prod_h == stock_H and prod_w < stock_W:
                    priority_1 = 3
                elif prod_w == stock_W and prod_h < stock_H:
                    priority_1 = 2
                elif prod_w < stock_W and prod_h < stock_H:
                    priority_1 = 1
                else:
                    priority_1 = 0
                # rotate
                prod_w, prod_h = prod_size[::-1]

                if prod_w == stock_W and prod_h == stock_H:
                    priority_2 = 4
                elif prod_h == stock_H and prod_w < stock_W:
                    priority_2 = 3
                elif prod_w == stock_W and prod_h < stock_H:
                    priority_2 = 2
                elif prod_w < stock_W and prod_h < stock_H:
                    priority_2 = 1
                else:
                    priority_2 = 0
                if priority_1 >= priority_2:
                    priority = priority_1
                else:
                    priority = priority_2
                    prod["size"] = prod_size[::-1]
                prioritized_list.append((prod, priority))
        return prioritized_list

    def recursivePacking(self, x, y, stock_W, stock_H, observation, stock_idx=0, m=1):
        results = []
        if stock_W <= 0 or stock_H <= 0:
            return results
        if all(prod["quantity"] == 0 for prod in observation["products"]):
            return results
        priority_list = self.evaluatePriority(stock_W, stock_H, observation)
        priority_list.sort(key=lambda p: (p[1], p[0]["size"][1]), reverse=True)
        for prod, priority in priority_list:
            if prod["quantity"] == 0:
                continue
            prod_w, prod_h = prod["size"]
            if priority == 4:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                break
            elif priority == 3:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                results.extend(
                    self.recursivePacking(
                        x + prod_w,
                        y,
                        stock_W - prod_w,
                        stock_H,
                        observation,
                        stock_idx,
                        m,
                    )
                )
                break
            elif priority == 2:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                results.extend(
                    self.recursivePacking(
                        x,
                        y + prod_h,
                        stock_W,
                        stock_H - prod_h,
                        observation,
                        stock_idx,
                        m,
                    )
                )
                break
            elif priority == 1:
                results.append(
                    {
                        "stock_idx": stock_idx,
                        "size": (prod_w, prod_h),
                        "position": (x, y),
                    }
                )
                prod["quantity"] -= 1
                min_w = min(
                    (
                        p["size"][0]
                        for p in observation["products"]
                        if p["quantity"] > 0
                    ),
                    default=0,
                )
                min_h = min(
                    (
                        p["size"][1]
                        for p in observation["products"]
                        if p["quantity"] > 0
                    ),
                    default=0,
                )
                if stock_W - prod_w < min_w:
                    results.extend(
                        self.recursivePacking(
                            x + prod_w,
                            y,
                            stock_W - prod_w,
                            prod_h,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                    results.extend(
                        self.recursivePacking(
                            x,
                            y + prod_h,
                            stock_W,
                            stock_H - prod_h,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                elif stock_H - prod_h < min_h:
                    results.extend(
                        self.recursivePacking(
                            x,
                            y + prod_h,
                            prod_w,
                            stock_H - prod_h,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                    results.extend(
                        self.recursivePacking(
                            x + prod_w,
                            y,
                            stock_W - prod_w,
                            stock_H,
                            observation,
                            stock_idx,
                            m,
                        )
                    )
                else:
                    hb2 = (stock_W - prod_w) * prod_h
                    vb1 = prod_w * (stock_H - prod_h)
                    if hb2 / vb1 >= m:
                        results.extend(
                            self.recursivePacking(
                                x,
                                y + prod_h,
                                prod_w,
                                stock_H - prod_h,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                        results.extend(
                            self.recursivePacking(
                                x + prod_w,
                                y,
                                stock_W - prod_w,
                                stock_H,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                    else:
                        results.extend(
                            self.recursivePacking(
                                x + prod_w,
                                y,
                                stock_W - prod_w,
                                prod_h,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                        results.extend(
                            self.recursivePacking(
                                x,
                                y + prod_h,
                                stock_W,
                                stock_H - prod_h,
                                observation,
                                stock_idx,
                                m,
                            )
                        )
                break
        return results

    def store_actions(self, observation):
        """Calculates and stores the optimal actions in `ListOpt`."""
        listQuantity = [prod["quantity"] for prod in observation["products"]]
        stockUsed = 1000
        mOp = 1
        stocks = [
            (
                idx,
                np.array(stock, dtype=np.int32),
                self._get_stock_size_(np.array(stock, dtype=np.int32)),
            )
            for idx, stock in enumerate(observation["stocks"])
        ]
        sorted_stocks = sorted(
            stocks,
            key=lambda x: x[2][0] * x[2][1],
            reverse=True,
        )
        for m in range(1, 50):
            indexList = []
            for i, stock, (stock_w, stock_h) in sorted_stocks:
                stock_W, stock_H = self._get_stock_size_(stock)
                indexList.extend(
                    self.recursivePacking(0, 0, stock_W, stock_H, observation, i, m)
                )
                if all(prod["quantity"] == 0 for prod in observation["products"]):
                    if stockUsed > i:
                        stockUsed = i
                        mOp = m
                        break
            for i, prod in enumerate(observation["products"]):
                prod["quantity"] = listQuantity[i]
            indexList.clear()
        self.ListOpt = []
        for i, stock, (stock_w, stock_h) in sorted_stocks:
            stock_W, stock_H = self._get_stock_size_(stock)
            self.ListOpt.extend(
                self.recursivePacking(0, 0, stock_W, stock_H, observation, i, mOp)
            )
        for i, prod in enumerate(observation["products"]):
            prod["quantity"] = listQuantity[i]

    # Policy 2
    # Greedy scoring policy
    def scorePosition(self, stock, position, prodSize, currentStockIndex):
        pos_x, pos_y = position
        prod_w, prod_h = prodSize
        score = 0
        if currentStockIndex not in self.usedStocks:
            score -= 100
        score -= pos_x**2 + pos_y**2
        adjacentCells = [
            (pos_x - 1, pos_y),
            (pos_x + prod_w, pos_y),
            (pos_x, pos_y - 1),
            (pos_x, pos_y + prod_h),
        ]
        filledNeighbors = sum(
            1
            for x, y in adjacentCells
            if 0 <= x < stock.shape[0] and 0 <= y < stock.shape[1] and stock[x, y] != -1
        )
        score += filledNeighbors * 20
        for x in range(pos_x, pos_x + prod_w):
            for y in range(pos_y, pos_y + prod_h):
                if stock[x, y] == -1:
                    score -= 5
        return score

    def greedyScoring(self, observation, info):
        listProds = sorted(
            observation["products"],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True,
        )
        stocks = [
            (
                idx,
                np.array(stock, dtype=np.int32),
                self._get_stock_size_(np.array(stock, dtype=np.int32)),
            )
            for idx, stock in enumerate(observation["stocks"])
        ]
        sortedStocks = sorted(
            stocks,
            key=lambda x: x[2][0] * x[2][1],
            reverse=True,
        )
        prodSize = [0, 0]
        stockIndex = -1
        pos_x, pos_y = 0, 0
        for product in listProds:
            if product["quantity"] == 0:
                continue
            prod_w, prod_h = product["size"]
            for stockIndex, stock, (stock_w, stock_h) in sortedStocks:
                validPositions = []
                for rotatedSize in [(prod_w, prod_h), (prod_h, prod_w)]:
                    validPositions += [
                        ((x, y), rotatedSize)
                        for x in range(stock_w - rotatedSize[0] + 1)
                        for y in range(stock_h - rotatedSize[1] + 1)
                        if self._can_place_(stock, (x, y), rotatedSize)
                    ]
                bestOption = None
                bestScore = -float("inf")
                for position, size in validPositions:
                    score = self.scorePosition(stock, position, size, stockIndex)
                    if score > bestScore:
                        bestScore = score
                        bestOption = {
                            "stock_idx": stockIndex,
                            "size": size,
                            "position": position,
                        }
                if bestOption:
                    self.lastStockIndex = bestOption["stock_idx"]
                    self.usedStocks.add(self.lastStockIndex)
                    return bestOption
        return {"stock_idx": stockIndex, "size": prodSize, "position": (pos_x, pos_y)}

    # get_action function
    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            """Returns the next action from the precomputed `ListOpt`."""
            if not self.ListOpt:
                self.store_actions(
                    observation
                )  # Compute actions if not already computed
            if self.currentIndex < len(self.ListOpt):
                action = self.ListOpt[self.currentIndex]
                self.currentIndex += 1
                return action
            # Return None when all actions are exhausted
            else:
                self.currentIndex = 0
                self.ListOpt.clear()
                self.store_actions(observation)
                action = self.ListOpt[self.currentIndex]
                self.currentIndex += 1
                return action
        elif self.policy_id == 2:
            return self.greedyScoring(observation, info)

    # Student code here
    # You can add more functions if needed
