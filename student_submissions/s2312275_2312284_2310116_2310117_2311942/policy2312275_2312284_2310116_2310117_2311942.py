from policy import Policy
import numpy as np
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass

class Policy2312275_2312284_2310116_2310117_2311942(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be in [1, 2]"

        if policy_id == 1:
            self.policy = TreeBasedHeuristic()
        elif policy_id == 2:
            self.policy = GeneticAlgorithm()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
    
class CuttingPattern:
    def __init__(self, stock_size: Tuple[int, int], items: List[Tuple[int, int, int]]):
        self.stock_width, self.stock_height = stock_size
        self.items = items  # List of (width, height, quantity)
        self.placement = []  # List of (x, y, width, height, item_idx)
        self.fitness = 0.0

    def calculate_fitness(self):
        """Calculate fitness based on material utilization and overlap penalty"""
        used_area = sum(item[2] * item[3] for item in self.placement)
        total_area = self.stock_width * self.stock_height
        overlap_penalty = self._calculate_overlap()
        self.fitness = (used_area / total_area) - (overlap_penalty * 0.5)
        return self.fitness

    def _calculate_overlap(self):
        """Calculate overlap between placed items"""
        overlap_area = 0
        for i in range(len(self.placement)):
            for j in range(i + 1, len(self.placement)):
                x1, y1, w1, h1, _ = self.placement[i]
                x2, y2, w2, h2, _ = self.placement[j]
                
                overlap_width = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_height = min(y1 + h1, y2 + h2) - max(y1, y2)
                
                if overlap_width > 0 and overlap_height > 0:
                    overlap_area += overlap_width * overlap_height
        return overlap_area

class GeneticAlgorithm(Policy):
    def __init__(self, 
                 population_size=10,
                 generations=10,
                 mutation_rate=0.2,  # Increased mutation rate
                 elite_size=2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_solution = None
        self.used_positions = {}  # Track used positions per stock
        self.current_stock = 0    # Track current stock

    def reset(self):
        """Reset the algorithm state"""
        self.used_positions.clear()
        self.current_stock = 0
        self.best_solution = None

    def _initialize_population(self, stock_size, items):
        """Initialize population with diverse placements"""
        population = []
        for _ in range(self.population_size):
            pattern = CuttingPattern(stock_size, items)
            self._diverse_placement(pattern)
            population.append(pattern)
        return population

    def _diverse_placement(self, pattern):
        """Generate diverse placement for items"""
        for item_idx, (width, height, qty) in enumerate(pattern.items):
            for _ in range(qty):
                # Try multiple positions with different strategies
                positions = []
                
                # Strategy 1: Bottom-left
                positions.append((0, 0))
                
                # Strategy 2: Top-right
                positions.append((pattern.stock_width - width, pattern.stock_height - height))
                
                # Strategy 3: Center
                positions.append(((pattern.stock_width - width)//2, (pattern.stock_height - height)//2))
                
                # Strategy 4: Random positions
                for _ in range(5):
                    x = random.randint(0, pattern.stock_width - width)
                    y = random.randint(0, pattern.stock_height - height)
                    positions.append((x, y))

                # Try each position
                for x, y in positions:
                    if self._is_valid_position(pattern, x, y, width, height):
                        pattern.placement.append((x, y, width, height, item_idx))
                        break

    def _is_valid_position(self, pattern, x, y, width, height):
        """Check if position is valid and not overlapping"""
        if x + width > pattern.stock_width or y + height > pattern.stock_height:
            return False
            
        for px, py, pw, ph, _ in pattern.placement:
            if (x < px + pw and x + width > px and
                y < py + ph and y + height > py):
                return False
        return True

    def get_action(self, observation, info):
        """Implement the genetic algorithm for cutting stock"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Initialize tracking for new game
        if not hasattr(self, 'previous_products'):
            self.previous_products = None
            self.reset()

        # Check if it's a new game
        current_products = str([(p["size"][0], p["size"][1], p["quantity"]) 
                              for p in products if p["quantity"] > 0])
        if current_products != self.previous_products:
            self.reset()
            self.previous_products = current_products

        # Convert products to required format
        items = [(p["size"][0], p["size"][1], p["quantity"]) 
                for p in products if p["quantity"] > 0]
        
        if not items:
            return {
                "stock_idx": -1,
                "size": np.array([0, 0]),
                "position": np.array([0, 0])
            }

        # Try each stock starting from current_stock
        for stock_idx in range(self.current_stock, len(stocks)):
            stock = stocks[stock_idx]
            stock_size = self._get_stock_size_(stock)
            
            # Initialize tracking for this stock if needed
            if stock_idx not in self.used_positions:
                self.used_positions[stock_idx] = set()

            # Run genetic algorithm
            population = self._initialize_population(stock_size, items)
            best_action = None
            
            for gen in range(self.generations):
                # Evaluate fitness
                for pattern in population:
                    pattern.calculate_fitness()
                
                # Sort by fitness
                population.sort(key=lambda x: x.fitness, reverse=True)
                
                # Check each placement in best pattern
                for placement in population[0].placement:
                    x, y, w, h, _ = placement
                    pos_key = (x, y, w, h)
                    
                    # Skip if position already used
                    if pos_key in self.used_positions[stock_idx]:
                        continue
                        
                    # Validate placement
                    if self._can_place_(stock, (x, y), (w, h)):
                        self.used_positions[stock_idx].add(pos_key)
                        best_action = {
                            "stock_idx": stock_idx,
                            "size": np.array([w, h]),
                            "position": np.array([x, y])
                        }
                        break
                
                if best_action:
                    break
                    
                # Create new population
                new_population = population[:self.elite_size]
                selected = self._selection(population)
                
                while len(new_population) < self.population_size:
                    if len(selected) < 2:
                        break
                    parent1, parent2 = random.sample(selected, 2)
                    child = self._crossover(parent1, parent2)
                    self._mutation(child)
                    new_population.append(child)
                
                population = new_population

            if best_action:
                return best_action

            # Move to next stock if current one is full
            self.current_stock += 1

        # Reset to first stock if we've tried them all
        self.current_stock = 0
        self.used_positions.clear()
        
        return {
            "stock_idx": -1,
            "size": np.array([0, 0]),
            "position": np.array([0, 0])
        }
    
    def _selection(self, population):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def _crossover(self, parent1, parent2):
        """Order crossover for placement sequences"""
        child = CuttingPattern((parent1.stock_width, parent1.stock_height), parent1.items)
        crossover_point = random.randint(0, len(parent1.placement))
        
        child.placement = parent1.placement[:crossover_point]
        remaining = [p for p in parent2.placement if p not in child.placement]
        child.placement.extend(remaining)
        
        return child

    def _mutation(self, pattern):
        """Random repositioning mutation"""
        if random.random() < self.mutation_rate:
            if pattern.placement:
                idx = random.randint(0, len(pattern.placement) - 1)
                x, y, w, h, item_idx = pattern.placement[idx]
                new_x = random.randint(0, pattern.stock_width - w)
                new_y = random.randint(0, pattern.stock_height - h)
                pattern.placement[idx] = (new_x, new_y, w, h, item_idx)

    def _calculate_waste(self, stock, x, y, w, h):
        """Calculate waste score for placing item"""
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Area utilization
        area_ratio = (w * h) / (stock_w * stock_h)
        
        # Edge alignment bonus
        edge_bonus = 0
        if x == 0 or x + w == stock_w:
            edge_bonus += 0.1
        if y == 0 or y + h == stock_h:
            edge_bonus += 0.1

        # Calculate waste score (lower is better)
        waste = (1 - area_ratio) - edge_bonus
        
        return waste

@dataclass
class Node:
    x: int  # x coordinate
    y: int  # y coordinate
    width: int  # width of space
    height: int  # height of space
    used: bool = False
    right: Optional['Node'] = None
    bottom: Optional['Node'] = None

class TreeBasedHeuristic(Policy):
    def __init__(self):
        self.trees = {}  # Dictionary to store trees for each stock
        self.current_stock = 0
        self.min_waste_threshold = 0.1

    def _create_node(self, x: int, y: int, width: int, height: int) -> Node:
        """Create a new tree node representing a rectangular space"""
        return Node(x=x, y=y, width=width, height=height)

    def _find_node(self, root: Node, width: int, height: int) -> Optional[Node]:
        """Find suitable node that fits the product dimensions using best-fit strategy"""
        if root.used:
            # Try right branch
            right_node = self._find_node(root.right, width, height) if root.right else None
            if right_node:
                return right_node
                
            # Try bottom branch
            return self._find_node(root.bottom, width, height) if root.bottom else None

        # Check if product fits in current node
        if width <= root.width and height <= root.height:
            # Check if node is perfect fit
            if width == root.width and height == root.height:
                return root
                
            # Return current node as it can accommodate the product
            return root
            
        return None

    def _split_node(self, node: Node, width: int, height: int) -> bool:
        """Split node into two parts after placing a product"""
        if node.used:
            return False

        # Calculate remaining space
        remaining_right = node.width - width
        remaining_bottom = node.height - height

        # Split horizontally if wider, vertically if taller
        if remaining_right > remaining_bottom:
            # Create right node
            node.right = self._create_node(
                node.x + width,
                node.y,
                remaining_right,
                height
            )
            # Create bottom node
            node.bottom = self._create_node(
                node.x,
                node.y + height,
                node.width,
                remaining_bottom
            )
        else:
            # Create bottom node
            node.bottom = self._create_node(
                node.x,
                node.y + height,
                width,
                remaining_bottom
            )
            # Create right node
            node.right = self._create_node(
                node.x + width,
                node.y,
                remaining_right,
                node.height
            )

        node.used = True
        return True

    def _calculate_utilization(self, node: Node, product_area: int) -> float:
        """Calculate space utilization for placing product in node"""
        if not node:
            return 0.0
            
        node_area = node.width * node.height
        if node_area == 0:
            return 0.0
            
        return product_area / node_area

    def get_action(self, observation, info):
        """Get next action using tree-based placement strategy"""
        stocks = observation["stocks"]
        products = [p for p in observation["products"] if p["quantity"] > 0]
        
        if not products:
            return {
                "stock_idx": -1,
                "size": np.array([0, 0]),
                "position": np.array([0, 0])
            }

        # Sort products by area (largest first)
        products.sort(key=lambda x: (
            -(x['size'][0] * x['size'][1]),  # Area
            -max(x['size']),  # Longest side
            -min(x['size'])   # Shortest side
        ))

        best_action = None
        best_utilization = -1

        # Try each product
        for prod in products:
            if prod["quantity"] <= 0:
                continue

            prod_w, prod_h = prod["size"]
            prod_area = prod_w * prod_h

            # Try each stock
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Initialize tree for stock if not exists
                if stock_idx not in self.trees:
                    self.trees[stock_idx] = self._create_node(0, 0, stock_w, stock_h)

                # Try both orientations
                for w, h in [(prod_w, prod_h), (prod_h, prod_w)]:
                    if w > stock_w or h > stock_h:
                        continue

                    # Find suitable node
                    node = self._find_node(self.trees[stock_idx], w, h)
                    if node:
                        # Calculate utilization
                        utilization = self._calculate_utilization(node, prod_area)
                        
                        if utilization > best_utilization:
                            best_utilization = utilization
                            best_action = {
                                "stock_idx": stock_idx,
                                "size": np.array([w, h]),
                                "position": np.array([node.x, node.y])
                            }

                            # Split node if utilization is good enough
                            if utilization > self.min_waste_threshold:
                                self._split_node(node, w, h)
                                return best_action

        if best_action:
            # Split the chosen node
            node = self._find_node(
                self.trees[best_action["stock_idx"]], 
                best_action["size"][0],
                best_action["size"][1]
            )
            self._split_node(
                node,
                best_action["size"][0],
                best_action["size"][1]
            )
            return best_action

        return {
            "stock_idx": -1,
            "size": np.array([0, 0]),
            "position": np.array([0, 0])
        }

    def _validate_placement(self, stock, position, size) -> bool:
        """Validate if placement is possible"""
        x, y = position
        w, h = size
        
        # Check boundaries
        stock_w, stock_h = self._get_stock_size_(stock)
        if x + w > stock_w or y + h > stock_h:
            return False
            
        # Check if space is empty
        return self._can_place_(stock, position, size)
