# Benchmark simulating biological Neuron Model:
Multicore cpu and gpu Benchmark. But in fact it is simulation hodgkin huxley model neuron. Where for computing Ordinary Differential Equations i used Differentiable ODE solver method.
## Script
Benchmark in script version for Linux:
#### Installation
- In Directly benchmark_ytLip/
```bash
sudo apt install python3-venv
```
``` bash
python3 -m venv benchmark_script
```
``` bash
source benchmark_script/bin/activate 
```
Install package:
``` bash 
pip3 -r package.txt
```
``` bash
 sudo apt install python3-tk 
```
Install torchdiffeq
``` bash
pip3 install git+https://github.com/rtqichen/torchdiffeq
```
Benchmark in script version for Windows:
#### Installation
(IN PROGRESS)

## App verssion
Maybe in future ^ ^

### Run:
In directly folder benchmark_script
- [--h] Num Hiden Neuron; [--i] Num input Size; [--d] Device choice; [--t] Test Size
- If Dont put Neuron Hidden num then Hidden_size is equal => 2 * sqrt(Input_size)
```bash
python3 bench.py --i 2048 --d cpu --t 10
## Plan
- Notebook cpu version (DONE)
- Support gpu (DONE)
- Shell Script (Done)
- App Linux (In progress)
- App Windows
- ...
