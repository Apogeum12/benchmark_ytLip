from __future__ import print_function
import torch, argparse, time, matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.matplotlib_fname()

from Neuron_Model import SNN

parser = argparse.ArgumentParser(description='Run benchmark')
parser.add_argument("--h", default=None, type=int, help='Hidden Size') # Hidden_Size 1-Layer Neuron --- My_propose: 1024 ---
parser.add_argument("--i", default=784, type=int, help='Input Size')
parser.add_argument("--d",
                    choices=["cpu", "cuda"], required=True,
                    type=str, help='Device choice')
parser.add_argument("--t", default=10, type=int, help='Test Size')

args = parser.parse_args()

# "cpu" -- Multicore + gpu  | "cuda" --- singlecore + gpu
if args.h is None:
    hidden_size = 2 * int(np.sqrt(args.i))
    input_size=args.i
    device = args.d 
else:
    hidden_size = args.h
    input_size=args.i
    device = args.d 
x = torch.randn(input_size).to(device)
model = SNN(device, input_size, hidden_size).to(device)


def test(test_size, model, x):
    test_score = list()
    for i in range(test_size):
        start_time = time.time()
        X = model(x) #dopri5, euler, adams The best dopri5
        _time = time.time() - start_time
        test_score.append(_time)
    return test_score, X # Get Last X

def point(test_score):
    print("--- Max: {:.4f} s --- Min: {:.4f} s --- Mean: {:.4f} s --- Error: {:.4f} s---".format(np.max(test_score)
                                                                                            ,np.min(test_score), np.mean(test_score),
                                                                                            np.std(test_score)))

def main_plot(input, time_step, dt):
    t = torch.arange(0.0, time_step, dt)
    plt.subplots(figsize=(30, 7)) 
    plt.title('Hodgkin-Huxley Neuron') 
    plt.plot(t.cpu(), X[0].cpu(), 'c')
    plt.ylabel('V (mV)')
    plt.xlabel('s (1e-1 s)')
    plt.savefig('Membrane_Potencial.png')

if __name__ == "__main__":
    test_score, X = test(args.t, model, x)
    point(test_score)
    main_plot(X, time_step=60, dt=0.1)



'''
plt.subplots(figsize=(30, 7)) 
plt.title('Hodgkin-Huxley Neuron') 
plt.plot(t.cpu(), X[1].cpu().detach().numpy(), 'k')
plt.plot(t.cpu(), X[2].cpu().detach().numpy(), 'c')
plt.plot(t.cpu(), X[3].cpu().detach().numpy(), 'b')
plt.ylabel('V (mV)')
plt.xlabel('s (1e-1 s)')
plt.show()
'''