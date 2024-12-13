# Mathematical Modelling - Fall 2024

## Assignment: Cutting Stock Problem
<!-- Describe cutting stock problem -->
Cuttin Stock Problem is a combinatorial optimization problem that arises in many industrial applications. The problem consists of cutting stocks of material into smaller pieces in order to meet the demand for smaller pieces. The objective is to minimize the number of stocks used to meet the demand for smaller pieces. The problem is NP-hard and can be solved using integer programming techniques.

Below is a demonstration of greedy algorithm for cutting stock problem.
<!-- Show gif file named demo/greedy.gif -->
![Greedy Algorithm](demo/greedy.gif)

## Installation
<!-- Describe how to install the project -->
To install the project, you need to have Python installed on your machine. You can install Python from the official website. Once you have Python installed, you can clone the repository and run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
<!-- Describe how to use the project -->
To use the project, you need to run the following command:
```bash
python main.py
```

## How to implement your own policy
<!-- Describe how to implement your own policy -->
To implement your own policy, you need to create a new class that inherits from the `Policy` class and implement the `get_action` method. The `get_action` method should take a list of demands and a list of stock as input and return a dictionary that contains action information. The action information should include the size of demand, stock index, and position to cut the stock. You should also implement the `__init__` method to initialize the policy with any required parameters. Please refer to the `RandomPolicy` class in the `policy.py` file for an example implementation.

You can start by creating a new file in the `student_submissions` directory and implementing your policy in `s22110xxx` folder. Your code should be named `policy2210xxx.py` where `2210xxx` is your student ID. The policy class should be named `Policy2210xxx` where `2210xxx` is your student ID and inherit from the `Policy` class. You can have some support files in `s22110xxx` folder. If you are in honor class, you must implement reinforcement learning policy. Once you have implemented your policy, you can run uncomment the line in the `main.py` file that imports your policy and run the project to test your policy. You can only use basic python libraries such as numpy, pandas, torch, tensorflow, scikit-learn and scipy to implement your policy. Please put the new library in the `requirements.txt` file.  

After you complete your policy, you need to submit your code as a pull request to the main repository. The pull request should include the following information:
- Title: Your student ID
- A brief description of your policy
- The implementation of your policy
- The results of your policy on the test data
- Any additional information you would like to include

Any solutions that are not submitted as a pull request or do not follow the above guidelines will not be accepted. If you have any questions or need help implementing your policy, please post a message in the discussion forum.

## Contributing
<!-- Describe how to contribute to the project -->
To contribute to the project, you need to fork the repository and create a new branch. Once you have made your changes, you can create a pull request to merge your changes into the main branch.

## License
<!-- Describe the license under which the project is distributed -->
This project is distributed under the MIT License. See `LICENSE` for more information.
