from policy import Policy
import numpy as np

class Policy2352013_2352223_2352922_2353037_2353219(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = Greedy()
        elif policy_id == 2:
            self.policy = Genetic()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class Greedy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        if info["filled_ratio"] == 0:
            stocks = np.array(observation["stocks"]) == -1
            s_size = np.vstack((np.sum(stocks[:,:,0], axis=1), np.sum(stocks[:,0], axis=1))).T

            s_rotate = s_size[:,0] > s_size[:,1]
            s_size[s_rotate] = s_size[s_rotate,::-1]
            stocks[s_rotate] = np.transpose(stocks[s_rotate], (0,2,1))

            p_size = np.array([product["size"] for product in observation["products"]])
            p_size.sort(axis=1)
            products = np.column_stack((p_size, np.array([prod["quantity"] for prod in observation["products"]])))[np.argsort(-p_size[:,1])]

            self.actions = []

            for i in np.argsort(-s_size[:,1]):
                for product in products:
                    x_range = range(s_size[i, 1] - product[1] + 1)
                    for y in range(s_size[i, 0] - product[0] + 1):
                        for x in x_range:
                            if np.all(stocks[i, y:y + product[0], x:x + product[1]]):
                                stocks[i, y:y + product[0], x:x + product[1]].fill(False)
                                product[2] -= 1
                                self.actions.append({"stock_idx": i, "size": product[[1,0]], "position": (x, y)} if s_rotate[i]
                                                    else {"stock_idx": i, "size": product[[0,1]], "position": (y, x)})
                                if product[2] == 0: break
                        if product[2] == 0: break
                products = products[products[:,2] > 0]

                for product in products:
                    x_range = range(s_size[i, 1] - product[0] + 1)
                    for y in range(s_size[i, 0] - product[1] + 1):
                        for x in x_range:
                            if np.all(stocks[i, y:y + product[1], x:x + product[0]]):
                                stocks[i, y:y + product[1], x:x + product[0]].fill(False)
                                product[2] -= 1
                                self.actions.append({"stock_idx": i, "size": product[[0,1]], "position": (x, y)} if s_rotate[i]
                                                    else {"stock_idx": i, "size": product[[1,0]], "position": (y, x)})
                                if product[2] == 0: break
                        if product[2] == 0: break
                products = products[products[:,2] > 0]
                if len(products) == 0: break

        return self.actions.pop()
    
class Genetic(Policy):
    def __init__(self):
        self.best_chromosome = None     # the best chromosome
        self.best_cut = None            # the guillotine cut of the best chromosome
        self.best_score = 1             # the score of the best guillotine cut
        self.idx = 0

    # A. Encoding
    def Encoding(self):
        self.sheetsCode = np.empty(len(self.sheets)) # sheet code
        for i in range(len(self.sheets)):
            self.sheetsCode[i] = i
        total_pieces = sum([x[2] for x in self.pieces])
        self.piecesCode = np.zeros(total_pieces, dtype=int) # pieces' code
        pieces_cnt = 0
        for i in range(len(self.pieces)):
            for _ in range(self.pieces[i][2]):
                self.piecesCode[pieces_cnt] = i
                pieces_cnt += 1
        self.sheets_length = len(self.sheets)
        self.pieces_length = len(self.piecesCode)
        self.chromosomes_length = self.sheets_length + self.pieces_length
        # "the population size is set as 30∼40 of the length of a chromosome in this study"
        # -> TOO SLOW! We set it to something small
        self.population_size = 20
    
    # B. Initial Population
    def InitPopulation(self):
        self.chromosomes = np.empty((self.population_size, self.chromosomes_length), dtype=int)
        chromosomes = self.chromosomes
        for i in range(self.population_size):
            # chromosomes are chosen randomly
            sheets = np.random.permutation(self.sheetsCode)
            pieces = np.random.permutation(self.pieces_length)
            chromosomes[i] = np.concatenate((sheets, pieces))
            #print(self.chromosomes[i])

    # C. Guillotine Cut Process and D. Heuristic Placement Method for Multiple Sheets
    # Check to see if (a, b) is closer to the bottom-left (i.e (0, 0)) than (c, d)
    def CloserToBottomLeft(self, a, b, c, d):
        dist1 = a*a + b*b
        dist2 = c*c + d*d
        return (dist1 < dist2) or (dist1 == dist2 and a < c)

    # Returns the intersection of two rectangles 
    # [(x1, y1), (x2, y2)] and [(x3, y3), (x4, y4)]
    def Intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if x1 >= x4 or x2 <= x3 or y1 >= y4 or y2 <= y3:
            return None
        return max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)

    # Difference and Elimination processess
    # Described in: Lai, K. K., & Chan, J. W. M. (1997). Developing a simulated annealing algorithm for the cutting stock problem.
    def DifElim(self, inter, ers):
        # Difference of [(x1, y1), (x2, y2)] and [(x3, y3), (x4, y4)], denoted as [(x1, y1), (x2, y2)] - [(x3, y3), (x4, y4)], 
        # (assuming that x1 <= x3 <= x4 <= x2 and y1 <= y3 <= y4 <= y2) equals to the following four:
        # [(x1, y1), (x3, y2)]; [(x4, y1), (x2, y2)]; [(x1, y1), (x2, y3)]; [(x1, y4), (x2, y2)]
        # We calculate [(ers[0], ers[1]), (ers[2], ers[3])] - [(inter[0], inter[1]), (inter[2], inter[3])]
        # We also eliminate the infinitely thin ERSs
        new_ers = []
        if ers[0] < inter[0]:
            new_ers.append((ers[0], ers[1], inter[0], ers[3]))
        if inter[2] < ers[2]:
            new_ers.append((inter[2], ers[1], ers[2], ers[3]))
        if ers[1] < inter[1]:
            new_ers.append((ers[0], ers[1], ers[2], inter[1]))
        if inter[3] < ers[3]:
            new_ers.append((ers[0], inter[3], ers[2], ers[3]))
        if len(new_ers) == 0:
            return None
        return new_ers

    def GuillotineCut(self, chromosome):
        CloserToBottomLeft = self.CloserToBottomLeft
        Intersection = self.Intersection
        DifElim = self.DifElim
        pieces = self.pieces
        piecesCode = self.piecesCode
        sheets = self.sheets

        # Initialize the ERS lists
        #print("Chromosome: ", chromosome)
        ERS = np.empty(self.sheets_length, dtype=object)
        for i in range(self.sheets_length):
            ERS[i] = [(0, 0, sheets[chromosome[i]][0], sheets[chromosome[i]][1])]
        # print("Initial ERS: ", ERS)
        # Largest ERS and smallers pieces to evaluate the score
        largest_ERS = np.zeros(self.sheets_length, dtype=int)
        small_piece = np.full(self.sheets_length, np.inf)
        placed = np.empty(self.chromosomes_length, dtype=object) # array to determine whether Pj is placed or not
        for i in range(self.sheets_length):
            # Si = self.sheets[chromosome[i]]
            for j in range(self.sheets_length, self.chromosomes_length):
                if placed[j] is not None:
                    continue
                Pj = pieces[piecesCode[chromosome[j]]]
                # Find the ERSk
                ERSk_x, ERSk_y = np.inf, np.inf
                for ers in ERS[i]:
                    if ers[0] + Pj[0] <= ers[2] and ers[1] + Pj[1] <= ers[3] and CloserToBottomLeft(ers[0], ers[1], ERSk_x, ERSk_y):
                        ERSk_x, ERSk_y = ers[0], ers[1]
                if ERSk_x == np.inf:
                    continue
                # If there is ERSk then we place Pj in the bottom-left corner of ERSk
                placed[j] = (chromosome[i], ERSk_x, ERSk_y, ERSk_x + Pj[0], ERSk_y + Pj[1])
                # Determine the smallest piece. Use i instead of chromosome[i] because it's just
                # a permutation anyway
                if Pj[0] * Pj[1] < small_piece[i]:
                    small_piece[i] = Pj[0] * Pj[1]

                removal_cnt = 0
                to_remove = np.zeros(len(ERS[i]), dtype=bool)
                new_ers = []
                for ide, ers in enumerate(ERS[i]):
                    # Create new ERSs
                    inter = Intersection(ERSk_x, ERSk_y, ERSk_x + Pj[0], ERSk_y + Pj[1], ers[0], ers[1], ers[2], ers[3])
                    if inter is None:
                        continue
                    to_remove[ide] = True
                    removal_cnt += 1
                    temp_ers = DifElim(inter, ers)
                    if temp_ers is not None:
                        new_ers.extend(temp_ers)
                # Cross-checking the new ERSs
                n_ers_to_remove = np.zeros(len(new_ers), dtype=bool)
                for id, n_ers in enumerate(new_ers):
                    if n_ers_to_remove[id]:
                        continue
                    for ide, ers in enumerate(ERS[i]):
                        if to_remove[ide]: # ers is no longer valid so we skip it
                            continue
                        if ers[0] <= n_ers[0] and ers[1] <= n_ers[1] and n_ers[2] <= ers[2] and n_ers[3] <= ers[3]:
                            n_ers_to_remove[id] = True
                            removal_cnt += 1
                            break
                    if not n_ers_to_remove[id]:
                        for id2, n_ers2 in enumerate(new_ers):
                            if n_ers_to_remove[id2] or id == id2:
                                continue
                            if n_ers2[0] <= n_ers[0] and n_ers2[1] <= n_ers[1] and n_ers[2] <= n_ers2[2] and n_ers[3] <= n_ers2[3]:
                                n_ers_to_remove[id] = True
                                removal_cnt += 1
                                break
                # Combine the ERSs
                combined_ers_size = len(ERS[i]) + len(new_ers) - removal_cnt
                combined_ers = np.empty(combined_ers_size, dtype=object)
                idx = 0
                for ide, ers in enumerate(ERS[i]):
                    if not to_remove[ide]:
                        combined_ers[idx] = ers
                        idx += 1
                for idn, n_ers in enumerate(new_ers):
                    if not n_ers_to_remove[idn]:
                        combined_ers[idx] = n_ers
                        idx += 1
                # Set the new ERSs
                ERS[i] = combined_ers

            # Determine the largest ERS
            for ers in ERS[i]:
                area = (ers[2] - ers[0]) * (ers[3] - ers[1])
                if area > largest_ERS[i]:
                    largest_ERS[i] = area

        #print(placed[self.sheets_length:])
        return placed[self.sheets_length:], largest_ERS, small_piece
    
    # Score function
    def Score(self, chromosome):
        # The score function is a function of the total area of the pieces placed
        placed, largest_ERS, small_piece = self.GuillotineCut(chromosome)
        score, minus, sheet_area, sheets = 0, 0, 0, self.sheets
        sheet_mark = self.sheet_mark
        sheet_mark.fill(False)

        # Each p in placed has the form sheet_code + (x1, y1, x2, y2)
        for p in placed:
            if p is not None:
                score += (p[3] - p[1]) * (p[4] - p[2])
                if not sheet_mark[p[0]]:
                    sheet_area += sheets[p[0]][0] * sheets[p[0]][1]
                    sheet_mark[p[0]] = True

        for i in range(self.sheets_length):
            if small_piece[i] != np.inf:
                minus += (small_piece[i]/sheet_area) * (largest_ERS[i]/sheet_area)

        return 1 - score/sheet_area - 0.02*minus

    def Breed(self, parent1, parent2):
        # Creates a child from its parents' genes
        newchild = np.full(len(parent1), -1, dtype=int)
        chosen = np.zeros(len(parent1), dtype=bool) # check to see if gene's already been chosen
        for i in range(len(parent1)):
            gene1 = parent1[i]
            gene2 = parent2[i]
            if np.random.rand() < 0.5:
                if not chosen[gene1]:
                    newchild[i] = gene1
                    chosen[gene1] = True
                elif not chosen[gene2]:
                    newchild[i] = gene2
                    chosen[gene2] = True
            else:
                if not chosen[gene2]:
                    newchild[i] = gene2
                    chosen[gene2] = True
                elif not chosen[gene1]:
                    newchild[i] = gene1
                    chosen[gene1] = True
        queue = []
        for gene in parent1:
            if not chosen[gene]:
                queue.append(gene)
        for i in range(len(parent1)):
            if newchild[i] == -1:
                newchild[i] = queue.pop(0)
        return newchild

    # E. Reproduction, F. Crossover and G. Mutation
    def NextGeneration(self):
        chromosomes = self.chromosomes
        Breed = self.Breed
        Score = self.Score
        sheetsCode = self.sheetsCode
        sheets_length = self.sheets_length
        pieces_length = self.pieces_length
        # Calculate the score of each chromosome
        scores = np.empty(len(chromosomes))
        for i in range(len(chromosomes)):
            scores[i] = Score(chromosomes[i])
        # Sort the chromosomes by their increasing scores
        sorted_idx = np.argsort(scores, kind='mergesort')
        chromosomes = chromosomes[sorted_idx]
        # Reproduction. We have Mutation rate = 0.1
        do_mutation = np.random.rand() < 0.1
        # Select the best 20% of the chromosomes for the new generation
        reproduct_range = int(0.2 * self.population_size)
        new_generation = np.empty((self.population_size, self.chromosomes_length), dtype=int)
        new_generation[:reproduct_range] = chromosomes[:reproduct_range]
        limit = 0.8 * self.population_size if do_mutation else self.population_size
        index_newgen = reproduct_range
        while index_newgen < limit:
            # parent1 is chosen from the elitists while parent2 is chosen randomly
            parent1 = new_generation[np.random.randint(reproduct_range)]
            parent2 = chromosomes[np.random.randint(len(chromosomes))]
            newchild_sheets = Breed(parent1[:sheets_length], parent2[:sheets_length])
            newchild_pieces = Breed(parent1[sheets_length:], parent2[sheets_length:])
            new_generation[index_newgen] = np.concatenate((newchild_sheets, newchild_pieces))
            index_newgen += 1
        # If mutation happens, the worst 15% of the chromosomes are regenerated
        if do_mutation:
            for i in range(index_newgen, self.population_size):
                sheets = np.random.permutation(sheetsCode)
                pieces = np.random.permutation(pieces_length)
                new_generation[i] = np.concatenate((sheets, pieces))
        self.chromosomes = new_generation

    def Iteration(self, num_iter):
        best_chromosome = None
        best_cut = None
        best_score = 1
        for _ in range(1, num_iter + 1):
            # old_best = best_score
            # print("Generation: ", _)
            self.NextGeneration()
            best_chromosome = self.chromosomes[0]
            best_cut, _, _ = self.GuillotineCut(best_chromosome)
            best_score = self.Score(best_chromosome)
            # print("Best score: ", best_score, "Previous: ", old_best)
        # print("Iteration done")
        return best_chromosome, best_cut, best_score

    def MPGA(self, num_phase = 7, num_iter = 20):
        # print("Phase: 0 Initialization")
        self.Encoding()
        self.InitPopulation()
        phases_not_improved = 0
        for _ in range(1, num_phase + 1):
            # print("Phase: ", phase)
            best_chromosome, best_cut, best_score = self.Iteration(num_iter)
            if best_score < self.best_score:
                self.best_chromosome = best_chromosome
                self.best_cut = best_cut
                if self.best_score - best_score < 1e-4:
                    phases_not_improved += 1
                else:
                    phases_not_improved = 0
                self.best_score = best_score
            
            # Early stopping to save resources
            if phases_not_improved == 2:
                break
            # print("Best score of this phase: ", best_score)
            # print("Best score so far: ", self.best_score)
            # If the best score has not been updated for 2 consecutive phases, we break
            # Save the top 15% of the chromosomes for the next phase
            top15_chromosomes = self.chromosomes[:int(0.15 * self.population_size)]
            self.InitPopulation()
            # Insert the top 15% of the chromosomes into the new population by replacing part of the new population
            self.chromosomes[:len(top15_chromosomes)] = top15_chromosomes

    def get_action(self, observation, info):
        if self.best_chromosome is None:
            # remember that each sheet/piece has the form {w, h, ci}
            list_stocks = observation["stocks"]
            list_prods = observation["products"]
            self.sheets = []
            self.pieces = []
            self.sheetsArea = 0             # the total area of the sheets
            self.sheet_mark = np.zeros(len(list_stocks), dtype=bool)

            for stock in list_stocks:
                stock_w, stock_h = self._get_stock_size_(stock)
                self.sheetsArea += stock_w * stock_h
                self.sheets.append([stock_w, stock_h])
            for prod in list_prods:
                self.pieces.append([prod["size"][0], prod["size"][1], prod["quantity"]])

            self.MPGA()
        
        while self.idx < len(self.best_cut):
            if self.best_cut[self.idx] is None:
                self.idx += 1
                continue
            placement_info = self.best_cut[self.idx]
            stock_idx = placement_info[0]
            size = (placement_info[3] - placement_info[1], placement_info[4] - placement_info[2])
            position = (placement_info[1], placement_info[2])
            self.idx += 1
            return {"stock_idx": stock_idx, "size": size, "position": position}
        
        # reset for next episode
        self.best_chromosome = None     # the best chromosome
        self.best_cut = None            # the guillotine cut of the best chromosome
        self.best_score = 1             # the score of the best guillotine cut
        self.idx = 0
        return self.get_action(observation, info)