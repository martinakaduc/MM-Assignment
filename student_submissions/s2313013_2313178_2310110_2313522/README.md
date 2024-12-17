## How to use this project?
First, you need to check the requirements.txt - the one in s2313013_2313178_2310110_2313522 folder, if you already have those package, you don't need to install anymore. If you don't, please run the command below to install all necessary package.
```bash
pip install -r requirements.txt
```

Then, you have to pass some parameters to the init function of the Policy, First is the policy_id, if you want to use Column Generation method, please pass 1 into the init function, if you want to use REINFORCE method, pass 2. Then, is the is_training, this parameters is made for REINFORCE method, if you choose REINFORCE method, you have to pass this parameter, too. If you want the model in training mode, pass True, if you want it in predict mode, pass False.

After that, you can run this project by the code command below:
```bash
python main.py
```