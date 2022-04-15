# A-W-Duality-Project
Demo code for paper "The activity-weight duality in feed forward neural networks: The geometric determinants of generalization" 
https://arxiv.org/abs/2203.10736

## Usage
Requirments: Anoconda; Python 3.6; Matlab 2021;

Here is the instruction to install Anocoda in PC: https://www.anaconda.com/
* Clone, Setup project on your PC
  ```
  $ git clone https://github.com/YuFengDuke/A-W-Duality-Project.git
  ```
  ```
  $ bash requirements.sh
  ```
  It should take less than 5 minites to finish all installation. 
 
* Run Demo in a small dataset
  The default demo code will show a example of A-W Duality analysis on a subset of MNIST when changing batch size.
  Demo code could be tested on 'normal' PC with CPU. This demo takes ~30 minites. To start demo, one can run:
  ```
  $ bash run.sh
  ```
  To get the results without figure, simply type:
  ```
  $ python3 main.py
  ```
  To get results with other hyperparameters, one can change "para_name" in main.py. Please see model_config.py for more details.
* What to expect:
  * In the running program, one can expect the program print the generalization loss gap and the loss gap estimated by AW-Duality
  * A result file contains the flatness measure and distance measure in all eigen-directions of hessian matrix
  * Figures that visualize the flatness measure and distance measure in each direction
  * Figures that visualize the loss gap contributed from flat and sharp direction
