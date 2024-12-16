import numpy as np
from policy import Policy

class Policy2352082_2352424_2352425_2352831_2352931(Policy):
    def __init__(self,policy_id = 1):
        assert policy_id in [1,2]
        # 1 for Genetic Algorithm, 2 for Simulated Annealing
        if policy_id == 1:
            self.policy_id =policy_id
            self.population_size = 4
            self.num_generations = 5
            self.mutation_rate = 0.05
            self.crossover_rate = 0.8
            self.elite_size = 2
            self.best_solution = None
        elif policy_id == 2:
            self.policy_id = 2
            self.temperature = 10000
            self.cooling_rate = 0.95
            self.max_iter = 1000
                
    def get_action(self, observation, info):       
        if self.policy_id ==1:
            self.train(observation)
            action =self._decode_action(self.best_solution,observation)
           
            return action
        elif self.policy_id == 2:
            current_solution = self._generate_initial_solution(observation)
            current_cost = self._calculate_cost(observation, current_solution)

            best_solution = current_solution
            best_cost = current_cost

            for _ in range(self.max_iter):
                neighbor_solution = self._generate_neighbor(observation, current_solution)
                neighbor_cost = self._calculate_cost(observation, neighbor_solution)

                if self._acceptance_probability(current_cost, neighbor_cost):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost
            
                if neighbor_cost < best_cost:
                    best_solution = neighbor_solution
                    best_cost = neighbor_cost

                self.temperature *= self.cooling_rate
                if self.temperature < 1e-3:
                    break
            return best_solution
    def _get_stock_area(self, stock):
        w, h = self._get_stock_size_(stock)
        return w * h

    def _get_utilization(self, stock):
        return np.sum(stock != -1) / self._get_stock_area(stock)
    def _get_sorted_stocks(self, observation):
        return list(range(len(observation["stocks"])))
       
    
    def _create_bottom_left_individual_1(self, observation, sorted_stocks):
        
        individual = []
        for prod in observation["products"]:
            if prod["quantity"] <= 0:
                individual.append((-1, 0, 0, False))
                continue
            
            placed = False
            for stock_idx in sorted_stocks:
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod["size"]
                rotated_w,rotated_h = prod_h,prod_w
                for pos_y in range(max(stock_h-prod_h+1,stock_h-rotated_h+1)):
                    for pos_x in range(max(stock_w-prod_w+1,stock_w-rotated_w+1)):
                        if stock[pos_x][pos_y] != -1:
                            continue
                        if (stock_w >= prod_w and stock_h >= prod_h and 
                            pos_x + prod_w <= stock_w and pos_y + prod_h <= stock_h):
                            if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                                individual.append((stock_idx, pos_x, pos_y, False))
                                placed = True
                                break
                            
                        if not placed:  
                            rotated_w, rotated_h = prod_h, prod_w    
                            if (stock_w >= rotated_w and stock_h >= rotated_h and 
                                pos_x + rotated_w <= stock_w and pos_y + rotated_h <= stock_h):                        
                                if self._can_place_(stock, (pos_x, pos_y), (rotated_w, rotated_h)):
                                    individual.append((stock_idx, pos_x, pos_y, True))
                                    placed = True
                                    break
                       
                                           
                                
                    if placed:
                        break
                if placed:
                    break
                
            if not placed:
                individual.append((-1, 0, 0, False))
            
        return individual
    def _create_bottom_left_individual_2(self, observation, sorted_stocks):
        
        individual = []
        for prod in observation["products"]:
            if prod["quantity"] <= 0:
                individual.append((-1, 0, 0, False))
                continue
            
            placed = False
            for stock_idx in sorted_stocks:
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod["size"]
                rotated_w,rotated_h = prod_h,prod_w
                for pos_x in range(max(stock_w-prod_w+1,stock_w-rotated_w+1)):
                    for pos_y in range(max(stock_h-prod_h+1,stock_h-rotated_h+1)):
                        if stock[pos_x][pos_y] != -1:
                            continue
                        if (stock_w >= prod_w and stock_h >= prod_h and 
                            pos_x + prod_w <= stock_w and pos_y + prod_h <= stock_h):
                            if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                                individual.append((stock_idx, pos_x, pos_y, False))
                                placed = True
                                break
                            
                        if not placed:  
                            rotated_w, rotated_h = prod_h, prod_w
                            if (stock_w >= rotated_w and stock_h >= rotated_h and 
                                pos_x + rotated_w <= stock_w and pos_y + rotated_h <= stock_h):
                                if self._can_place_(stock, (pos_x, pos_y), (rotated_w, rotated_h)):
                                    individual.append((stock_idx, pos_x, pos_y, True))
                                    placed = True
                                    break

                    if placed:
                        break
                if placed:
                    break
                
            if not placed:
                individual.append((-1, 0, 0, False))
            
        return individual
    
    def _initialize_population(self, observation):
        population = []
        sorted_stocks = self._get_sorted_stocks(observation)
        individual_1 = self._create_bottom_left_individual_1(observation,sorted_stocks)
        individual_2 = self._create_bottom_left_individual_2(observation,sorted_stocks)
        population.append(individual_1)
        population.append(individual_2)
        for _ in range(self.population_size):
            
            population.append(individual_1)
            population.append(individual_2)
       
        return population

    def _fitness(self, individual, observation):
        total_placed_area = 0
       
        edge_contact = 0  
        piece_contact = 0 
        utilization = 0
        for i, (stock_idx, x, y,rotated) in enumerate(individual):
            if stock_idx == -1:
                continue
                
            prod = observation["products"][i]
            stock = observation["stocks"][stock_idx]
            prod_w, prod_h = (prod["size"][1], prod["size"][0]) if rotated else prod["size"]
            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                
                stock_w, stock_h = self._get_stock_size_(stock)
                
                piece_area = prod_w * prod_h
                total_placed_area += piece_area
                
                
                if x == 0 or x + prod_w == stock_w:
                    edge_contact += prod_h
                if y == 0 or y + prod_h == stock_h:
                    edge_contact += prod_w
                    
                piece_contact += self._calculate_piece_contacts(
                    stock, (x, y), (prod_w, prod_h)
                )
                utilization += self._get_utilization(stock)
        fitness = (
            total_placed_area * 1.0 + 
            edge_contact * 0.3 +     
            piece_contact * 0.3       
        )
        
        return  fitness*utilization  
    
    def _calculate_piece_contacts(self, stock, position, size):
        x, y = position
        w, h = size
        contact = 0

        for dx in range(w):
            if y > 0 and 0 <= x + dx < len(stock[0])  and stock[y - 1][x + dx] != -1:
                contact += 1
            if y + h < len(stock) and 0 <= x + dx < len(stock[0]) and stock[y + h][x + dx] != -1:
               contact += 1

        for dy in range(h):
            if x > 0  and 0 <= y + dy < len(stock) and stock[y + dy][x - 1] != -1:
                contact += 1
            if x + w < len(stock[0]) and 0 <= y + dy < len(stock) and stock[y + dy][x + w] != -1:
                contact += 1

        return contact
    def train(self,observation):
        population = self._initialize_population(observation)
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in range(self.num_generations):
            fitness_scores = [
                self._fitness(ind, observation) for ind in population
            ]
            if(generation==self.num_generations-1):
                break
            current_best = max(fitness_scores)
            if current_best>best_fitness:
                best_fitness = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            if generations_without_improvement > 3:
                self.mutation_rate = 0.1
            else:
                self.mutation_rate =0.05
            elite = sorted(zip(population,fitness_scores),key=lambda x : x[1],reverse=True)[:self.elite_size]
            elite_solutions = [e[0] for e in elite]
            new_population = elite_solutions.copy()
            while len(new_population)< self.population_size:
                parent1,parent2 = self._select_parents(population,fitness_scores)
                if np.random.rand()<self.crossover_rate:
                    child1,child2 = self._crossover(parent1,parent2)
                else:
                    child1,child2 = parent1,parent2
                new_population.extend([self._mutate(child,observation) for child in [child1,child2]]) 
            population = new_population[:self.population_size]
        
        best_idx = fitness_scores.index(max(fitness_scores))
        self.best_solution = population[best_idx]

    def _crossover(self,parent1,parent2):
        if len(parent1) != len(parent2):
            return parent1,parent2
        if len(parent1) < 2:
            return parent1,parent2
        indices = np.random.choice(len(parent1),size = 2,replace=False)
        points =sorted(indices)
        def create_child(p1,p2,points):
            child=[]
            used_positions={idx:[] for idx in range(100)}
            for i in range(len (p1)):
                if points[0] <= i <= points[1]:
                    gene = p2[i]
                else:
                    gene = p1[i]
                if gene[0] != -1:
                    valid = True
                    for used_stock,used_x,used_y,used_rotate in used_positions[gene[0]]:
                        if(used_x == gene[1] and used_y == gene[2] and used_rotate == gene[3]):
                            valid = False
                            break
                    if valid:
                        child.append(gene)
                        used_positions[gene[0]].append((gene[0],gene[1],gene[2],gene[3]))
                    else:
                        child.append((-1,0,0,False))
                else:
                    child.append((-1,0,0,False))
            return child

        child1 = create_child(parent1,parent2,points)
        child2 = create_child(parent2,parent1,points)
        
        return child1,child2
    
    def _mutate(self,individual,observation):
        
        mutated = list(individual)
        mutation_type = np.random.rand()
        for i in range(len(mutated)):
            if np.random.rand() <self.mutation_rate:
               
                if mutation_type < 0.3:
                    if mutated[i][0] != -1:
                        stock = observation["stocks"][mutated[i][0]]
                        stock_w,stock_h = self._get_stock_size_(stock)
                        prod = observation["products"][i]
                        prod_w, prod_h = (prod["size"][1], prod["size"][0]) if mutated[i][3] else prod["size"]
                        max_shift = 3
                        original_x,original_y = mutated[i][1],mutated[i][2]
                        
                        for _ in range(5):
                            shift_x = np.random.randint(-max_shift,max_shift)
                            shift_y = np.random.randint(-max_shift,max_shift)
                            new_x = max(0,min(original_x+shift_x,stock_w-prod_w))
                            new_y = max(0,min(original_y+shift_y,stock_h-prod_h))
                            if stock[new_x][new_y] != -1:
                                continue
                            if self._can_place_(stock,(new_x,new_y),(prod_w,prod_h)):
                                mutated[i] = (mutated[i][0],new_x,new_y,mutated[i][3])
                                break
                            else:
                                prod_w,prod_h = prod_h,prod_w
                                if(new_x+prod_w<=stock_w and new_y+prod_h<=stock_h and self._can_place_(stock,(new_x,new_y),(prod_w,prod_h))):
                                    mutated[i] = (mutated[i][0],new_x,new_y,not mutated[i][3])
                                    break
                elif mutation_type < 0.6:  
                    prod = observation["products"][i]
                    if prod["quantity"] > 0:
                        sorted_stocks = self._get_sorted_stocks(observation)
                        placed = False
                        
                        for stock_idx in sorted_stocks:
                            stock = observation["stocks"][stock_idx]
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod["size"]
                            rotated_w,rotated_h = prod_h,prod_w
                            for pos_y in range(max(stock_h-prod_h+1,stock_h-rotated_h+1)):
                                for pos_x in range(max(stock_w-prod_w+1,stock_w-rotated_w+1)):
                                    if stock[pos_x][pos_y] != -1:
                                        continue
                                    if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                                        mutated[i] = (stock_idx, pos_x, pos_y, False)
                                        placed = True
                                        break
                                
                                    if not placed:
                                        rotated_w, rotated_h = prod_h, prod_w
                                        if self._can_place_(stock, (pos_x, pos_y), (rotated_w, rotated_h)):
                                            mutated[i] = (stock_idx, pos_x, pos_y, True)
                                            placed = True
                                            break
                                if placed:
                                    break
                            if placed:
                                break
                else:
                    prod = observation["products"][i]
                    if prod["quantity"] > 0:
                        sorted_stocks = self._get_sorted_stocks(observation)
                        placed = False
                        
                        for stock_idx in sorted_stocks:
                            stock = observation["stocks"][stock_idx]
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod["size"]
                            rotated_w,rotated_h = prod_h,prod_w
                            for pos_x in range(max(stock_w-prod_w+1,stock_w-rotated_w+1)):
                                for pos_y in range(max(stock_h-prod_h+1,stock_h-rotated_h+1)):
                                   
                                    if stock[pos_x][pos_y] != -1:
                                        continue
                                    if (stock_w >= prod_w and stock_h >= prod_h and 
                                        pos_x + prod_w <= stock_w and pos_y + prod_h <= stock_h):
                                        if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                                            mutated[i] = (stock_idx, pos_x, pos_y, False)
                                            placed = True
                                            break
                                
                                    if not placed:
                                        rotated_w, rotated_h = prod_h, prod_w
                                        if (stock_w >= prod_w and stock_h >= prod_h and 
                                            pos_x + prod_w <= stock_w and pos_y + prod_h <= stock_h):
                                            if self._can_place_(stock, (pos_x, pos_y), (rotated_w, rotated_h)):
                                                mutated[i] = (stock_idx, pos_x, pos_y, True)
                                                placed = True
                                                break
                                if placed:
                                    break
                            if placed:
                                break
                    
        return mutated
    
    def _select_parents(self,population,fitness_scores):
        tournament_size = len(population)//2
        
        def tournament():
            candidate_idx = np.random.choice(len(population),size = tournament_size,replace=False)
            
            candidate_fitness = [fitness_scores[i] for i in candidate_idx]
            
            selected_idx = candidate_idx[candidate_fitness.index(max(candidate_fitness))]
            return population[selected_idx]
        return tournament(),tournament()
    
    def _decode_action(self, individual, observation):
        if not individual:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        best_action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        best_idx = -1
        best_score = -1  
        for idx, (stock_idx, x, y, rotated) in enumerate(individual):
            if stock_idx == -1:
                continue
            if idx >= len(observation["products"]):
                continue
            prod = observation["products"][idx]
            if prod["quantity"] <= 0:
                continue
           
            stock = observation["stocks"][stock_idx]
            prod_w, prod_h = (prod["size"][1], prod["size"][0]) if rotated else prod["size"]

            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
             
                score = 0            
                piece_contact = self._calculate_piece_contacts(stock, (x, y), (prod_w, prod_h))
                score += piece_contact * 2
            
                if x == 0:
                    score += 3
                if y == 0:
                    score += 3
                
                piece_area = prod_w * prod_h
                score += piece_area / 100  
                
                utilization = self._get_utilization(stock)
                
                score *= utilization            
           
                if score > best_score or best_action == {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}:
                    best_idx = idx
                    best_action = {
                    "stock_idx": stock_idx,
                    "size": [prod_w, prod_h],
                    "position": (x, y)
                    }
                    best_score = score
        
        return best_action
   
    def _generate_initial_solution(self, observation):
        list_prods = sorted(
            [p for p in observation["products"] if p["quantity"] > 0],
            key=lambda p: (p["size"][0] * p["size"][1], -min(p["size"])),
            reverse=True
        )
        
        for prod in list_prods:
            prod_size = prod["size"]
            rotated_size = (prod_size[1],prod_size[0])
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                for x in range(max(stock_w-prod_size[0]+1,stock_w-rotated_size[0]+1)):
                    for y in range(max(stock_h-prod_size[1]+1,stock_h-rotated_size[1]+1)):
                        if stock[x][y] != -1:
                            continue
                        if (stock_w >= prod_size[0] and stock_h >= prod_size[1] and 
                            x + prod_size[0] <= stock_w and y + prod_size[1] <= stock_h):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": i, "size": prod_size, "position": (x, y)}

                        if (stock_w >= rotated_size[0] and stock_h >= rotated_size[1] and 
                                x + rotated_size[0] <= stock_w and y + rotated_size[1] <= stock_h):
                            if self._can_place_(stock, (x, y), rotated_size):
                                return {"stock_idx": i, "size": rotated_size, "position": (x, y)}

        return {"stock_idx": -1, "size": (0, 0), "position": (None, None)}
    def _generate_neighbor(self, observation, current_solution):
        neighbor_solution = current_solution.copy()
        stock_idx = neighbor_solution["stock_idx"]
        product_size = neighbor_solution["size"]
        position = neighbor_solution["position"]
        x, y = position

        stock = observation["stocks"][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)

        if np.random.rand() < 0.5:
            rotated_size = (product_size[1], product_size[0])
            if stock_w >= rotated_size[0] and stock_h >= rotated_size[1]:
                neighbor_solution["size"] = rotated_size
        else:
            new_position,new_size = self._find_new_position(stock, product_size)
            if new_position is not None:
                neighbor_solution["position"],neighbor_solution["size"] = new_position,new_size
                

        return neighbor_solution

    def _find_new_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        rotated_w,rotated_h = prod_size[1],prod_size[0]
        for y in range(max(stock_h-prod_size[1]+1,stock_h-rotated_h+1)):
            for x in range(max(stock_w-prod_size[0]+1,stock_w-rotated_w+1)):
            
                if (stock_w >= prod_size[0] and stock_h >= prod_size[1]):
                    if stock[x][y] == -1 and self._can_place_(stock, (x, y), prod_size):
                        return (x, y),(prod_size[0],prod_size[1])
                if (stock_w>= rotated_w and stock_h >=rotated_h):
                    if stock[x][y] == -1 and self._can_place_(stock,(x,y),(rotated_w,rotated_h)):
                        return (x,y),(rotated_w,rotated_h)    
                    
                    
                    
        return None

    def _calculate_cost(self, observation, solution):
        total_placed_area = 0
        total_waste = 0
        edge_contact = 0  
        piece_contact = 0 
        utilization = 0

        stock_idx = solution["stock_idx"]
        prod_size = solution["size"]
        position = solution["position"]
        x, y = position

        if stock_idx == -1:
            return 0
        
        prod = observation["products"][0]  
        stock = observation["stocks"][stock_idx]
        prod_w, prod_h = prod_size

        if self._can_place_(stock, (x, y), (prod_w, prod_h)):
            stock_w, stock_h = self._get_stock_size_(stock)

            piece_area = prod_w * prod_h
            total_placed_area += piece_area
            total_waste += (stock_w * stock_h - piece_area)
            if x == 0 or x + prod_w == stock_w:
                edge_contact += prod_h
            if y == 0 or y + prod_h == stock_h:
                edge_contact += prod_w

            piece_contact += self._calculate_piece_contacts(stock, (x, y), (prod_w, prod_h))
            utilization += self._get_utilization(stock)

        cost = (
            total_placed_area +
            edge_contact * 0.5 +
            piece_contact * 0.5
        )
        return -cost * utilization
    def _acceptance_probability(self, current_cost, neighbor_cost):
        if neighbor_cost < current_cost:
            return True
        delta = (current_cost - neighbor_cost) / self.temperature
        return np.random.rand() < np.exp(delta)                
                
                        
        
    
            
                
                


