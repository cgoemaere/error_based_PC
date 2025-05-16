## Installation

Install the required packages:

    pip install -r requirements.txt

## Running with WANDB Sweeps

1. **Set up WANDB:**  
    Log in to WANDB with:
    ```
    wandb login
    ```

Create a sweep using a configuration file. All configuration files for the hyperparameter sweep can be found as `configs_sweeps/`. All the configuration files for the models with optimised parameters can be found at `configs_results/`. To start running models you can:

2. **Initialize a Sweep:**  
    ```
    wandb sweep sweep_config.yaml
    ```
    This command will return a Sweep ID.

3. **Run the Sweep Agent:**  
    Start one or more agents to run the experiments:
    ```
    wandb agent <User_Name>/<Project_Name>/<Sweep_ID>
    ```
    From the config files, <Project_Name> is errorPC for sweeps and errorpc_results for running tuned models.

Replace `<Sweep_ID>` with the Sweep ID received from the previous step.
This will start the hyperparameter tuning or run a model for 5 seeds.


**Example:**
    
    wandb login

    wandb sweep configs_results/Error_CIFAR10_VGG5_mse.yaml

    wandb agent <User_Name>/errorpc_results/<Sweep_ID>
    
