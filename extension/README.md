This folder contains the code for backdoored samples detection in the subsequent extended version of the ISSTA paper `attack as defense`.

### Preliminary

The backdoor attack code uses the module [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox). We did not add it as a submodule here, please make sure you have loaded the repository in this directory, and checked out the code run successfully.

### Usage
The main algorithm is in `main_backdoor_detector.py`, you can execute the file with `--help` to check the input arguments.

The path of the datasets and models in the code need to be modified according to your file location. To facilitate the implementation, we have provided several backdoor models in the `/models` folder.


