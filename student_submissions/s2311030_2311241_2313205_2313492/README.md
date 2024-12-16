# Group 125's Submission - Math Modeling Assignment - Semester 241

## Group Members
- Phạm Trương Anh Huy - 2311241
- Nguyễn Thanh Toàn - 2313492
- Nguyễn Mạnh Thi - 2313205
- Cao Hữu Thiên Hoàng - 2311030

## Assignment: Algorithm for Cutting Stock Problem

Warning: This reinforcement learning algorithm using an self-define environment and masking mechanism in the agent. We add a reset method in our policy. When you test it in more than 1 episode, please add the reset method when the environment return terminated | truncated. Otherwise, this reinforcement learning algorithm can not achieve expected result. Thank you for your attention.

### Iterative Sequential Heuristic Procedure
![ISHP Algorithm](student_submissions/s2210xxx/ishp.gif)
The Iterative Sequential Heuristic Procedure (ISHP) is a heuristic algorithm designed to solve the two-dimensional cutting stock problem (2D CSP). Its primary goal is to maximize material utilization by iteratively generating and evaluating feasible cutting patterns while minimizing waste.
Key features of the ISHP algorithm include:
#### Pattern Generation
- Iterates at each step generating patterns for chosen.
- Uses a 3-Step Greedy Packing (3SGP) strategy to construct cutting patterns.
- Incorporates product rotation and recursive partitioning to improve flexibility and compactness.
#### Pattern Evaluation
- Scores patterns based on material utilization and penalizes frequently reused patterns to encourage diversity.
- Select pattern that has best score.
#### Dynamic Correction Values
- Uses simple correction values methods
- Updates correction values for products based on past placements, allowing the algorithm to "learn" and improve over iterations.
#### Actions queue
- Converts the best-selected pattern into a sequence of actions, ensuring compatibility with environment constraints.

### Branch and Bound
![Branch and Bound Algorithm](student_submissions/s2210xxx/B&B.gif)
Algorithm places products into available stocks to optimize space utilization and minimize waste. It uses a heuristic approach to guide decisions without backtracking.
Key features of the B&B algorithm include:
#### Product Placement
- Checks if a product can fit in a stock at a specific position, considering boundaries and availability.
- Tries both original and rotated orientations to find the best placement.
#### Sorting and Prioritization
- Products are sorted by size (largest first) to maximize space usage.
- Stocks are prioritized based on how well they fit the product, considering available space and dimension matching.
#### Bounds Calculation
- Estimates the minimum number of stocks needed based on the total product area and available stock area.
- Ensures efficient use of resources while avoiding unnecessary stock usage.
#### Branch Exploration
- Attempts to place each product in the best possible stock based on the current state.
- Reverts placements after evaluation to explore other options but does not revisit earlier branches systematically.
#### Action Selection
- Chooses the best action from the evaluated branches and applies it to the environment.


## Reinforcement Learning
The reinforcement learning (RL) algorithm designed for the 2D Cutting Stock Problem (2D CSP) leverages a Proximal Policy Optimization (PPO) framework to train an agent capable of optimizing stock utilization while minimizing trim-loss. The architecture incorporates key components such as encoders, an environment simulator, a masking mechanism, and neural networks for decision-making.
![Reinforcement Learning](student_submissions/s2210xxx/RL.gif)
Core Features:
### Encoders
- Product Encoder: Extracts features from product data using a 1D Convolutional Layer.
- Stock Encoder: Processes stock data using a 2D Convolutional Layer for spatial feature extraction.
### Environment Simulator
Simulates the real environment to enable efficient action queries during training.
### Masking Mechanism
Prevents the agent from repeating invalid actions.
Prioritizes already-used stocks to minimize the number of stocks utilized.
### Neural Networks
- Actor (Policy Network): Outputs a probability distribution over the action space.
- Critic (Value Network): Estimates the expected return from the current state.
### PPO Framework
Ensures stable policy updates through clipping mechanisms and advantage estimation using Generalized Advantage Estimation (GAE).
### Results
While the algorithm reduced trim-loss through heuristic-based masking, it failed to converge to an optimal policy due to:
- Limitations in architecture: Lack of decoding mechanisms and overly simplistic encoders.
- Inefficient reward signals: Inadequate penalization of wasteful actions and insufficient incentivization of compact placements.
- Restricted exploration: The masking mechanism overly constrained the agent, limiting its ability to learn and generalize.

Despite these challenges, the experiment laid a foundation for improvement, highlighting areas such as integrating Transformer-based architectures and enhancing reward structures to guide the learning process effectively.
