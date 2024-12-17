from policy import Policy
import random
import math


class Policy2352492_2352693_2352852_2352960_2352919(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.temperature = 100.0
        self.cooling_rate = 0.955
        self.min_temperature = 0.1
        self.iterations = 2000

        self.population_size = 100
        self.generations = 50
        self.mutation_rate = 0.1
        self.elitism = 0.1

        self.best = []
        self.idx = -1

        self.id = policy_id

    class SkylinePacker:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.segments = [(0, 0, width)]

        def place(self, w, h):
            best_x = None
            best_y = None
            best_waste = float('inf')

            for i, (xs, yh, xe) in enumerate(self.segments):
                seg_width = xe - xs
                if seg_width >= w and yh + h <= self.height:
                    waste = yh + (seg_width - w)*0.001
                    if waste < best_waste:
                        best_waste = waste
                        best_x = xs
                        best_y = yh

            if best_x is None:
                return None

            self._update_skyline(best_x, best_x+w, best_y+h)
            return (best_x, best_y)

        def _update_skyline(self, start, end, new_height):
            new_segments = []
            for (xs, yh, xe) in self.segments:
                if xe <= start or xs >= end:
                    new_segments.append((xs, yh, xe))
                else:
                    if xs < start:
                        new_segments.append((xs, yh, start))
                    if xe > end:
                        new_segments.append((end, yh, xe))
            new_segments.append((start, new_height, end))
            new_segments.sort(key=lambda s: s[0])
            merged = [new_segments[0]]
            for seg in new_segments[1:]:
                if seg[1] == merged[-1][1] and seg[0] == merged[-1][2]:
                    merged[-1] = (merged[-1][0], merged[-1][1], seg[2])
                else:
                    merged.append(seg)
            self.segments = merged

    def get_action(self, observation, info):
        if (self.id == 1):
            return self._get_action_GA_(observation, info)
        elif (self.id == 2):
            return self._get_action_SA_(observation, info)
        else:
            pass

    def _get_action_SA_(self, observation, info):
        if self.best:
            if self.idx < len(self.best):
                action = self.best[self.idx]
                self.idx += 1
                return {"stock_idx": action["stock_idx"],
                        "size": action["size"],
                        "position": action["position"]}
            else:
                self.best = []
                self.idx = -1
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        products = observation["products"]
        stocks = observation["stocks"]

        full_list = [(i, False) for i, p in enumerate(products)
                     for _ in range(p["quantity"])]

        if not full_list:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        full_list.sort(
            key=lambda x: products[x[0]]["size"][0]*products[x[0]]["size"][1], reverse=True)

        current_solution = full_list[:]
        placed, current_score = self._compute_solution(
            products, stocks, current_solution)

        best_solution = current_solution[:]
        best_score = current_score
        for it in range(self.iterations):
            new_solution = self._neighbor(current_solution)
            _, new_score = self._compute_solution(
                products, stocks, new_solution)
            delta = new_score - current_score
            if delta < 0:
                current_solution = new_solution
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_solution = new_solution[:]
            else:
                if random.random() < math.exp(-delta/self.temperature):
                    current_solution = new_solution
                    current_score = new_score

            self._cool_down()

        final_placed, _ = self._compute_solution(
            products, stocks, best_solution)
        res = []
        for (p_i, s_i, pw, ph, x, y) in final_placed:
            res.append({"stock_idx": s_i, "size": [
                       pw, ph], "position": (x, y)})

        self.best = res
        self.idx = 0

        if self.best:
            action = self.best[self.idx]
            self.idx += 1
            if (self.idx == len(self.best)):
                self.best = []
                self.idx = -1
            return action
        else:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _compute_solution(self, products, stocks, solution):
        placed = []
        skylines = []
        stock_sizes = []
        for s in stocks:
            sw, sh = self._get_stock_size_(s)
            skylines.append(self.SkylinePacker(sw, sh))
            stock_sizes.append((sw, sh))

        for (p_idx, rot) in solution:
            pw, ph = products[p_idx]["size"]
            if rot:
                pw, ph = ph, pw
            area = pw*ph
            pos_found = False
            for s_idx, sl in enumerate(skylines):
                pos = sl.place(pw, ph)
                if pos is not None:
                    x, y = pos
                    placed.append((p_idx, s_idx, pw, ph, x, y))
                    pos_found = True
                    break
            if not pos_found:
                pass

        trim_losses = []
        for s_idx, s in enumerate(stocks):
            s_area = self._get_stock_area_(stocks, s_idx)
            used_area = 0
            for p_i, st_i, pw, ph, xx, yy in placed:
                if st_i == s_idx:
                    used_area += pw*ph
            if used_area > 0:
                trim_loss = (s_area - used_area)/s_area
                trim_losses.append(trim_loss)
        if trim_losses:
            avg_trim = sum(trim_losses)/len(trim_losses)
        else:
            avg_trim = 1.0

        return placed, avg_trim

    def _neighbor(self, solution):
        sol = solution[:]
        if random.random() < 0.5:
            if len(sol) > 1:
                i, j = random.sample(range(len(sol)), 2)
                sol[i], sol[j] = sol[j], sol[i]
        else:
            i = random.randint(0, len(sol)-1)
            p_idx, rot = sol[i]
            sol[i] = (p_idx, not rot)
        return sol

    def _cool_down(self):
        self.temperature = max(self.min_temperature,
                               self.temperature * self.cooling_rate)

    def _get_action_GA_(self, observation, info):
        if not self.best:
            import time
            start = time.time()
            list_prods = list(observation["products"])
            stocks = observation["stocks"]

            if not list_prods:
                return {
                    "stock_idx": -1,
                    "size": [0, 0],
                    "position": (0, 0)
                }

            for prod in list_prods:
                w, h = prod["size"]
                prod["area"] = w*h
            list_prods.sort(key=lambda p: p["area"], reverse=True)

            population = self._initialize_population(
                list_prods, stocks, observation)

            for generation in range(self.generations):
                fitness = self._evaluate_fitness(
                    population, stocks, observation)
                population = self._evolve_population(
                    population, fitness, stocks, observation)

            best_individual = min(
                population, key=lambda idx: self._evaluate_individual(idx, stocks, observation))
            trim_loss = self._evaluate_trim_loss(
                best_individual, stocks, observation)
            filled_ratio = self._evaluate_filled_ratio(
                best_individual, stocks, observation)

            products = observation["products"]
            product_sizes = [prod["size"] for prod in products]
            self.best = [
                {
                    "stock_idx": stock_idx,
                    "size": [prod_w, prod_h],
                    "position": (pos_x, pos_y)
                }
                for prod_idx, stock_idx, prod_w, prod_h, pos_x, pos_y in best_individual
            ]
            self.idx = 0

        res = self.best[self.idx]
        self.idx += 1
        if self.idx == len(self.best):
            self.best = []
            self.idx = -1
        return res

    def _initialize_population(self, products, stocks, observation):
        population = []
        num_stocks = len(stocks)

        def can_place(occupied_grid, prod_w, prod_h, start_x, start_y):
            stock_h = len(occupied_grid)
            stock_w = len(occupied_grid[0]) if stock_h > 0 else 0
            if start_x + prod_w > stock_w or start_y + prod_h > stock_h:
                return False
            for y in range(start_y, start_y + prod_h):
                for x in range(start_x, start_x + prod_w):
                    if occupied_grid[y][x]:
                        return False
            return True

        def place_product(occupied_grid, prod_w, prod_h, stock_w, stock_h):
            for pos_y in range(stock_h - prod_h + 1):
                for pos_x in range(stock_w - prod_w + 1):
                    if can_place(occupied_grid, prod_w, prod_h, pos_x, pos_y):
                        for y in range(pos_y, pos_y + prod_h):
                            for x in range(pos_x, pos_x + prod_w):
                                occupied_grid[y][x] = True
                        return pos_x, pos_y
            return None, None

        product_order = list(enumerate(products))
        stock_order = list(range(num_stocks))

        shuffle = False

        while len(population) < self.population_size:
            occupied_grids = {}
            stock_sizes = {}
            for i in range(num_stocks):
                sw, sh = self._get_stock_size_(stocks[i])
                stock_sizes[i] = (sw, sh)
                occupied_grids[i] = [[False]*sw for _ in range(sh)]

            individual = []
            if not shuffle:
                shuffle = True
            else:
                random.shuffle(stock_order)

            placed_all = True
            for prod_idx, prod in product_order:
                orig_w, orig_h = prod["size"]
                quantity = prod["quantity"]

                for _ in range(quantity):
                    if random.choice([True, False]):
                        prod_w, prod_h = orig_h, orig_w
                    else:
                        prod_w, prod_h = orig_w, orig_h

                    placed = False
                    for stock_idx in stock_order:
                        stock_w, stock_h = stock_sizes[stock_idx]

                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = place_product(occupied_grids[stock_idx],
                                                         prod_w, prod_h,
                                                         stock_w, stock_h)
                            if pos_x is not None and pos_y is not None:
                                individual.append(
                                    (prod_idx, stock_idx, prod_w, prod_h, pos_x, pos_y))
                                placed = True
                                break
                    if not placed:
                        placed_all = False
                        break
                if not placed_all:
                    break

            if placed_all:
                population.append(individual)

        return population

    def _evaluate_fitness(self, population, stocks, observation):
        return [
            self._evaluate_individual(individual, stocks, observation)
            for individual in population
        ]

    def _evaluate_individual(self, individual, stocks, observation):
        stock_areas = {stock_idx: self._get_stock_area_(
            stocks, stock_idx) for stock_idx in range(len(stocks))}

        used_areas = {stock_idx: 0 for stock_idx in range(len(stocks))}
        for prod_idx, stock_idx, prod_w, prod_h, pos_x, pos_y in individual:
            used_areas[stock_idx] += prod_w * prod_h

        trim_losses = []
        for stock_idx, stock_area in stock_areas.items():
            if used_areas[stock_idx] > 0:
                trim_loss = (stock_area - used_areas[stock_idx]) / stock_area
                trim_losses.append(trim_loss)

        return sum(trim_losses)/len(trim_losses) if trim_losses else 0

    def _evaluate_trim_loss(self, individual, stocks, observation):
        stock_areas = {stock_idx: self._get_stock_area_(
            stocks, stock_idx) for stock_idx in range(len(stocks))}

        used_areas = {stock_idx: 0 for stock_idx in range(len(stocks))}
        for prod_idx, stock_idx, prod_w, prod_h, pos_x, pos_y in individual:
            used_areas[stock_idx] += prod_w * prod_h

        trim_losses = []
        for stock_idx, stock_area in stock_areas.items():
            if used_areas[stock_idx] > 0:
                trim_loss = (stock_area - used_areas[stock_idx]) / stock_area
                trim_losses.append(trim_loss)
        return sum(trim_losses) / len(trim_losses) if trim_losses else 0

    def _evaluate_filled_ratio(self, individual, stocks, observation):
        used_stock_indices = set(
            stock_idx for _, stock_idx, _, _, _, _ in individual)
        num_stocks_used = len(used_stock_indices)
        total_num_stocks = len(stocks)
        return num_stocks_used / total_num_stocks if total_num_stocks else 0

    def _evolve_population(self, population, fitness, stocks, observation):
        new_population = []

        num_elite = int(self.elitism * self.population_size)
        elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[
            :num_elite]
        new_population.extend([population[i] for i in elite_indices])

        while len(new_population) < self.population_size:
            parent1 = self._select_parent(population, fitness)
            parent2 = self._select_parent(population, fitness)
            offspring = self._crossover(parent1, parent2, observation)
            mutated_offspring = self._mutate(offspring, stocks, observation)
            new_population.append(mutated_offspring)

        return new_population

    def _select_parent(self, population, fitness):
        tournament_size = 5
        import random
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = min(indices, key=lambda i: fitness[i])
        return population[best_idx]

    def _crossover(self, parent1, parent2, observation):
        offspring = []
        used_positions = {stock_idx: []
                          for stock_idx in range(len(observation["stocks"]))}

        for gene_idx in range(max(len(parent1), len(parent2))):
            if gene_idx < len(parent1) and gene_idx < len(parent2):
                gene = parent1[gene_idx] if random.random(
                ) < 0.5 else parent2[gene_idx]
            elif gene_idx < len(parent1):
                gene = parent1[gene_idx]
            elif gene_idx < len(parent2):
                gene = parent2[gene_idx]
            else:
                continue

            prod_idx, stock_idx, prod_w, prod_h, pos_x, pos_y = gene

            if not any(self._is_overlap(pos_x, pos_y, prod_w, prod_h, ox, oy, ow, oh)
                       for ox, oy, ow, oh in used_positions[stock_idx]):
                offspring.append(gene)
                used_positions[stock_idx].append(
                    (pos_x, pos_y, prod_w, prod_h))
            else:
                placed = False

                stock_w, stock_h = self._get_stock_size_(
                    observation["stocks"][stock_idx])
                for new_pos_y in range(stock_h - prod_h + 1):
                    for new_pos_x in range(stock_w - prod_w + 1):
                        if not any(self._is_overlap(new_pos_x, new_pos_y, prod_w, prod_h, ox, oy, ow, oh)
                                   for ox, oy, ow, oh in used_positions[stock_idx]):
                            offspring.append(
                                (prod_idx, stock_idx, prod_w, prod_h, new_pos_x, new_pos_y))
                            used_positions[stock_idx].append(
                                (new_pos_x, new_pos_y, prod_w, prod_h))
                            placed = True
                            break
                    if placed:
                        break

                if not placed:
                    prod_w, prod_h = prod_h, prod_w
                    for new_pos_y in range(stock_h - prod_h + 1):
                        for new_pos_x in range(stock_w - prod_w + 1):
                            if not any(self._is_overlap(new_pos_x, new_pos_y, prod_w, prod_h, ox, oy, ow, oh)
                                       for ox, oy, ow, oh in used_positions[stock_idx]):
                                offspring.append(
                                    (prod_idx, stock_idx, prod_w, prod_h, new_pos_x, new_pos_y))
                                used_positions[stock_idx].append(
                                    (new_pos_x, new_pos_y, prod_w, prod_h))
                                placed = True
                                break
                        if placed:
                            break

                if not placed:
                    rotate = random.choice([True, False])
                    if rotate:
                        prod_w, prod_h = prod_h, prod_w

                    for s_idx in range(len(observation["stocks"])):
                        stock_w, stock_h = self._get_stock_size_(
                            observation["stocks"][s_idx])
                        for pos_y in range(stock_h - prod_h + 1):
                            for pos_x in range(stock_w - prod_w + 1):
                                if not any(self._is_overlap(pos_x, pos_y, prod_w, prod_h, ox, oy, ow, oh)
                                           for ox, oy, ow, oh in used_positions[s_idx]):
                                    offspring.append(
                                        (prod_idx, s_idx, prod_w, prod_h, pos_x, pos_y))
                                    used_positions[s_idx].append(
                                        (pos_x, pos_y, prod_w, prod_h))
                                    placed = True
                                    break
                            if placed:
                                break
                        if placed:
                            break

        return offspring

    def _is_overlap(self, pos_x, pos_y, prod_w, prod_h, other_pos_x, other_pos_y, other_prod_w, other_prod_h):
        return not (pos_x + prod_w <= other_pos_x or pos_x >= other_pos_x + other_prod_w or
                    pos_y + prod_h <= other_pos_y or pos_y >= other_pos_y + other_prod_h)

    def _mutate(self, individual, stocks, observation):
        if random.random() < self.mutation_rate and individual:
            remove_index = random.randint(0, len(individual) - 1)
            prod_idx, stock_idx, prod_w_, prod_h_, old_pos_x, old_pos_y = individual.pop(
                remove_index)
            stock_order = list(range(len(stocks)))

            placed = False
            orig_w, orig_h = prod_w_, prod_h_

            for try_rotate in [True, False]:
                prod_w, prod_h = (orig_h, orig_w) if try_rotate else (
                    orig_w, orig_h)
                for s_idx in stock_order:
                    sw, sh = self._get_stock_size_(stocks[s_idx])
                    occupied = self._get_occupied_positions(
                        individual, s_idx, observation)
                    for new_pos_y in range(sh - prod_h + 1):
                        for new_pos_x in range(sw - prod_w + 1):
                            if not any(self._is_overlap(new_pos_x, new_pos_y, prod_w, prod_h, ox, oy, ow, oh)
                                       for ox, oy, ow, oh in occupied):
                                individual.insert(
                                    remove_index, (prod_idx, s_idx, prod_w, prod_h, new_pos_x, new_pos_y))
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

                if placed:
                    break

            if not placed:
                individual.insert(
                    remove_index, (prod_idx, stock_idx, prod_w_, prod_h_, old_pos_x, old_pos_y))

        return individual

    def _get_occupied_positions(self, individual, stock_idx, observation):
        return [(pos_x, pos_y, prod_w, prod_h)
                for p_idx, s_idx, prod_w, prod_h, pos_x, pos_y in individual
                if s_idx == stock_idx]

    def _get_stock_area_(self, stocks, s_idx):
        size = self._get_stock_size_(stocks[s_idx])
        return size[0] * size[1]
