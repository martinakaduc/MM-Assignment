from policy import Policy
import numpy as np

class Policy2352189_2352390_2352170_2352598_2352592(Policy):
    def __init__(self, policy_id = 1):
        self.population_size = 10   
        self.mutation_rate = 0.7 
        self.elite_size_select = 1
        self.generation_count = 1
        self.fitness_list = []
        self.actions = []
        self.products = self.stocks = []
        self.policy_id = policy_id
        
    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.GeneticAlgorithm(observation, info)
        elif self.policy_id == 2:
            return self.BestFitDecreasing(observation, info)

    def new_individual(self):
        product_counts = [0] * len(self.products)
        sheets = []

        indices = list(range(len(self.products)))

        
        if np.random.random() < self.mutation_rate:
            np.random.shuffle(indices)
        else:
            # Dùng FFD
            indices = sorted(indices, key=lambda p: self.products[p]["size"][0] * self.products[p]["size"][1], reverse=True)
        for stock in self.stocks:
            sheet = stock.copy()
            W, H = (
                np.sum(np.any(stock != -2, axis=1)),
                np.sum(np.any(stock != -2, axis=0)),
            )

            empty_map = np.full((H, W), True)

            for i in indices:
                product = self.products[i]
                w, h = product["size"]
                if product_counts[i] >= product["quantity"]:
                    continue
                for x in range(W - w + 1):
                    for y in range(H - h + 1):
                        if empty_map[y : y + h, x : x + w].all():
                            sheet[y : y + h, x : x + w] = i
                            empty_map[y : y + h, x : x + w] = False
                            product_counts[i] += 1
                            if product_counts[i] >= product["quantity"]:
                                break
                    if product_counts[i] >= product["quantity"]:
                        break

            sheets.append(sheet)

        fitness = self.evaluate_fitness((sheets, product_counts))
        self.fitness_list.append(fitness)
        return sheets, product_counts

    def init(self):
        self.population = []
        for _ in range(self.population_size):
            self.population.append(self.new_individual())

    def evaluate_fitness(self, individual):
        filled_ratio = []
        trim_loss = []

        for _, stock in enumerate(individual[0]):
            if np.all(stock<0) == False:
                tl = (stock == -1).sum() / (stock != -2).sum()
                trim_loss.append(tl)
                filled_ratio.append(1)
            else:
                filled_ratio.append(0)
        filled_ratio = np.sum(filled_ratio) / len(filled_ratio)
        trim_loss = np.mean(trim_loss).item() if trim_loss else 1
        return (1-filled_ratio)*(1-trim_loss)

    def crossover(self):
        elite_size = self.elite_size_select
        ranked_population = sorted(
            self.population, key=lambda x: self.evaluate_fitness(x), reverse=True
        )
        elite = ranked_population[:elite_size]
        offspring = [self.new_individual() for _ in range(self.population_size - elite_size)]
        self.population = elite + offspring

    def evolve(self):
        for _ in range(self.generation_count):
            self.calculate_fitness()
            self.crossover()

    def GeneticAlgorithm(self, observation, info):
        if len(self.actions) > 0:
            action = self.actions[0]
            self.actions.pop(0)
            return action
        else:        
            self.stocks = observation["stocks"]
            self.products = observation["products"]
            prods_length=0
            for prods_ in self.products:
                prods_length+=prods_["quantity"]

            self.init()
            self.evolve()
            self.population = sorted(self.population, key=lambda x: self.evaluate_fitness(x), reverse=True)

            sheets, product_counts = self.population[0]

            for i, _ in enumerate(observation["stocks"]):
                sheet = sheets[i]
                W = np.sum(np.any(sheet != -2, axis=1))
                H = np.sum(np.any(sheet != -2, axis=0))
                for x in range(W):
                    for y in range(H):
                        pos_val = sheet[y][x]
                        if pos_val >= 0:
                            curr_prod = self.products[pos_val]
                            w, h = curr_prod["size"]
                            sheet[y:y + h, x:x + w] = -1
                            action = {
                                "stock_idx": i,
                                "size": [w, h],
                                "position": (x, y)
                            }
                            self.actions.append(action)

            if len(self.actions) > 0:
                action = self.actions[0]
                self.actions.pop(0)
                return action

            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def BestFitDecreasing(self, observation, info):
        if info["filled_ratio"] == 0:
            self.stocks = sorted(observation["stocks"],key=lambda p: (w := self._get_stock_size_(p))[0] * w[1])
            self.products = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse = True)

        for product_index, product in enumerate(self.products):
            if product["quantity"] == 0: continue
            for stock_index, stock in enumerate(self.stocks):
                if product["quantity"] == 0: break
                W, H = self._get_stock_size_(stock)
                w, h = product["size"]
                
                #print([np.count_nonzero(stock == -1) for stock in self.stocks])
                
                for y in range(H - min(w, h) + 1):
                    for x in range(W - min(w, h) + 1):
                        if stock[y, x] >= 0: x += w
                        
                        if np.all(stock[y:y+h, x:x+w] == -1):
                            stock[y:y+h, x:x+w] = product_index
                            product["quantity"] -= 1
                            self.heapifyStocks(stock_index)                
                            return {"stock_idx": stock_index, "size": [w, h], "position": (x, y)}
                            
                        elif np.all(stock[y:y+w, x:x+h] == -1):
                            stock[y:y+w, x:x+h] = product_index
                            product["quantity"] -= 1
                            self.heapifyStocks(stock_index)
                            return {"stock_idx": stock_index, "size": [h, w], "position": (x, y)}
                        
                        if product["quantity"] == 0: break
                    if product["quantity"] == 0: break         
                    
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
    def heapifyStocks(self, index):
        l, r = 0, len(self.stocks) - 1
        m = (l + r) // 2
        while l < r:
            m = (l + r) // 2
            if np.count_nonzero(self.stocks[m] == -1) < np.count_nonzero(self.stocks[index] == -1): l = m + 1
            elif np.count_nonzero(self.stocks[m] == -1) == np.count_nonzero(self.stocks[index] == -1): r = m
            else: r = m - 1
            
        if index < m: m += 1
        self.stocks.insert(m, self.stocks.pop(index))