import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from scipy.interpolate import interp1d

# Load the raw data
data = io.loadmat('s1_data_raw.mat')  # a matlab file!

spike_times = [st[:, 0] for st in data['spike_times'].ravel()]  # a list of arrays of spike times in seconds, one for each neuron, spike times in seconds
vel = data['vels']  # velocity data shape (num_time_points, 2) for (x, y) coordinates
vel_times = data['vel_times'].squeeze()  # times the velocities were recorded

# Whiten the recorded velocities
vel_mean = np.mean(vel, axis=0)
vel_std = np.std(vel, axis=0)
vel_whitened = (vel - vel_mean) / vel_std

# Define constants
ms = 1e-3  # use this constant so you can write e.g. 1*ms for a time

# Define time step for discretization
dt = 1e-3  # 1 ms time bins

# Define testing range
test_start_time = 1000  # Start time in seconds
test_end_time = 1010    # End time in seconds
test_range = (test_start_time, test_end_time)

# Define the SNNLayer class
class SNNLayer(nn.Module):
    def __init__(self, n_in, n_out, spiking=True, tau=50, v_th=1.0):
        super(SNNLayer, self).__init__()
        self.n_in = n_in  # Number of input neurons
        self.n_out = n_out  # Number of output neurons
        self.spiking = spiking  # Whether the layer produces spiking outputs
        self.tau = tau  # Time constant of the neurons
        self.v_th = v_th  # Threshold voltage for spiking neurons

        # Initialize weights with normal distribution
        self.w = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.normal_(self.w, mean=0.0, std=0.1)

    def forward(self, x):
        batch_size, _, num_time_points = x.shape  # Get the shape of the input tensor
        device = x.device  # Get the device (CPU/GPU) of the input tensor

        # Initialize membrane potential and output tensor
        v = torch.zeros(batch_size, self.n_out, device=device)  # Membrane potential
        y = torch.zeros(batch_size, self.n_out, num_time_points, device=device)  # Output tensor

        for t in range(num_time_points):
            # Compute input current
            i_t = torch.matmul(x[:, :, t], self.w)  # Multiply input spikes with weights

            # Update membrane potential
            dv = (i_t - v) * (self.dt / self.tau)  # Leaky integrate-and-fire equation
            v = v + dv  # Update membrane potential

            if self.spiking:
                # Generate spikes
                spikes = (v >= self.v_th).float()  # Spikes occur where membrane potential exceeds threshold
                # Store spikes in output tensor
                y[:, :, t] = spikes
                # Reset membrane potential where spikes occur
                v = v * (1 - spikes)  # Reset membrane potential after spiking
            else:
                # Store membrane potential in output tensor
                y[:, :, t] = v  # Store membrane potential in the output tensor

        return y  # Return the output tensor

import pickle

import numpy as np
from scipy import io
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as tnnf
mse = nn.MSELoss()

from tqdm.notebook import tqdm

ms = 1e-3 # use this constant so you can write e.g. 1*ms for a time

import urllib.request, zipfile, os
filename = 's1_data_raw.mat'
if not os.path.exists(filename):
    urllib.request.urlretrieve('https://www.dropbox.com/sh/n4924ipcfjqc0t6/AACPWjxDKPEzQiXKUUFriFkJa?dl=1', 'data.zip')
    with zipfile.ZipFile('data.zip') as z:
        z.extract(filename)


# Load the raw data
data = io.loadmat('s1_data_raw.mat') # a matlab file!

spike_times = [st[:, 0] for st in data['spike_times'].ravel()] # a list of arrays of spike times in seconds, one for each neuron, spike times in seconds
vel = data['vels'] # velocity data shape (num_time_points, 2) for (x, y) coordinates
vel_times = data['vel_times'].squeeze() # times the velocities were recorded


t_start = vel_times[0]
t_end = vel_times[-1]

# Whiten the recorded velocities
vel_mean = np.mean(vel, axis=0)
vel_std = np.std(vel, axis=0)
vel_whitened = (vel - vel_mean) / vel_std

print("Mean of whitened velocities:", np.mean(vel_whitened, axis=0))
print("Standard deviation of whitened velocities:", np.std(vel_whitened, axis=0))

# Compute and print out the number of neurons and the number of spikes recorded
num_neurons = len(spike_times)
num_spikes = sum(len(spike) for spike in spike_times)

print("Expected number of Neurons: 52")
print("2.   Number of neurons in data:", num_neurons)
print("     Expected number of spikes: 51 mins * 60 * 9.3 * 52 = 1479816")
# Discrepancy can be due to equipment used or rounding
print("     Number of spikes recorded:", f"{num_spikes:.3f} (3 sf)")

# Compute and print out the duration of the experiment in seconds and minutes
experiment_duration = t_end - t_start
experiment_duration_minutes = experiment_duration / 60

print("3.   Duration of experiment in seconds:", f"{experiment_duration:.3f} (3 sf)")
print("     Duration of experiment in minutes:", f"{experiment_duration_minutes:.3f} (3 sf)")

# Compute and print out the sampling rate at which spikes were recorded
spike_durations = [spike[-1] - spike[0] for spike in spike_times]
average_spike_duration = np.mean(spike_durations)
spike_sampling_rate = num_spikes / average_spike_duration

print("4.   Sampling rate at which spikes were recorded (Hz):", f"{spike_sampling_rate:.3f} (3 sf)")

# Compute and print out the sampling rate at which velocities were recorded
vel_sampling_rate = len(vel_times) / experiment_duration

print("5.   Sampling rate at which velocities were recorded (Hz):", f"{vel_sampling_rate:.3f} (3 sf)")


plt.figure(figsize=(12, 5))

# Plot all spikes
ax = plt.subplot(231)
for i, spikes in enumerate(spike_times):
    plt.plot(spikes, [i] * len(spikes), '|')
plt.ylabel('Neuron index')

# Plot all x- and y- components of the velocities
plt.subplot(234, sharex=ax)
plt.plot(vel_times, vel[:, 0], label='X velocity')
plt.plot(vel_times, vel[:, 1], label='Y velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend(loc='best')

# Plot spikes at times t=1000 to t=1010
ax = plt.subplot(232)
time_window = (vel_times >= 1000) & (vel_times <= 1010)
for i, spikes in enumerate(spike_times):
    spikes_in_window = spikes[(spikes >= 1000) & (spikes <= 1010)]
    plt.plot(spikes_in_window, [i] * len(spikes_in_window), '|')
plt.ylabel('Neuron index')

# Plot velocities at times t=1000 to t=1010
plt.subplot(235, sharex=ax)
plt.plot(vel_times[time_window], vel[time_window, 0], label='X velocity')
plt.plot(vel_times[time_window], vel[time_window, 1], label='Y velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend(loc='best')

# Compute firing rates for each neuron and plot as a histogram
plt.subplot(233)
firing_rates = [len(spikes) / (t_end - t_start) for spikes in spike_times]
plt.barh(range(len(spike_times)), firing_rates, height=1)
plt.xlabel('Firing rate (sp/s)')
plt.ylabel('Neuron index')

# Plot all velocities as points in (x, y)-plane as a continuous curve
# Emphasise the region from t=1000 to t=1010 with a different colour
plt.subplot(236)
plt.plot(vel[:, 0], vel[:, 1], label='All velocities')
plt.plot(vel[time_window, 0], vel[time_window, 1], label='Velocities (1000-1010s)', color='red')
plt.xlabel('X velocity')
plt.ylabel('Y velocity')
plt.legend(loc='best')

plt.tight_layout()
plt.show()



from scipy.interpolate import interp1d

# Set up training / validation / testing ranges
total_duration = vel_times[-1] - vel_times[0]
train_frac = 0.8
val_frac = 0.1
test_frac = 0.1

train_end = int(total_duration * train_frac)
val_end = train_end + int(total_duration * val_frac)

train_range = (0, train_end)
val_range = (train_end, val_end)
test_range = (val_end, int(total_duration))


import numpy as np
import torch
from scipy.interpolate import interp1d
import random

def prepare_batch(batch_segment, spike_times, vel_times, vel, dt, length, num_time_points):
    start_time = batch_segment
    time_points = np.linspace(start_time, start_time + length, num_time_points)  # 1 second long segment

    # Spike data preparation
    x_batch = torch.zeros((len(spike_times), num_time_points))
    for neuron_index, neuron_spikes in enumerate(spike_times):
        spike_times_in_batch = neuron_spikes[(neuron_spikes >= start_time) & (neuron_spikes < start_time + 1)]
        indices = np.floor((spike_times_in_batch - start_time) / dt).astype(int)
        x_batch[neuron_index, indices] = 1

    # Velocity interpolation
    interp_func = interp1d(vel_times, vel.T, kind='linear', bounds_error=False, fill_value="extrapolate")
    y_batch = torch.tensor(interp_func(time_points))

    return x_batch, y_batch

def batched_data(range_to_use, spike_times, vel, vel_times, dt=ms, length=1, batch_size=64, max_num_batches=40):
    start_time, end_time = range_to_use

    # Create segments each length seconds long
    segments = np.arange(start_time, end_time, length)
    random.shuffle(segments)  # Shuffle segments to randomize batch contents

    num_segments = len(segments)
    num_batches = min(num_segments // batch_size, max_num_batches)

    for batch_idx in range(num_batches):
        x = torch.zeros((batch_size, len(spike_times), int(length / dt)))
        y = torch.zeros((batch_size, 2, int(length / dt)))

        for b in range(batch_size):
            segment_index = batch_idx * batch_size + b
            if segment_index < num_segments:
                x_batch, y_batch = prepare_batch(segments[segment_index], spike_times, vel_times, vel, dt, length, int(length / dt))
                x[b] = x_batch
                y[b] = y_batch

        yield x, y

# Plot a sample of data

# Plot a sample of data
x, y = next(batched_data(train_range, spike_times, vel, vel_times, dt=ms, length=1, batch_size=4))
print(f"X shape: {x.shape}")
print(f"Y shape: {y.shape}")
plt.figure(figsize=(12, 5))
for b in range(4):
  # Plot spikes for this batch index
  ax = plt.subplot(2, 4, b+1)
  for neuron_idx in range(x.shape[1]):  # x.shape[1] will give num neurons in the tensor
      spike_times_batch = np.where(x[b, neuron_idx].numpy() == 1)[0] * ms  # dt=ms
      plt.plot(spike_times_batch, [neuron_idx]*len(spike_times_batch), '|')
  plt.ylabel('Neuron index')
  plt.title(f'Batch index {b}')
  # Plot velocities for this batch index
  plt.subplot(2, 4, b+5, sharex=ax)
  time_axis = np.arange(0, x.shape[2]*1e-3, 1e-3)  
  plt.plot(time_axis, y[b, 0].numpy(), label='X velocity')
  plt.plot(time_axis, y[b, 1].numpy(), label='Y velocity')
  plt.xlabel('Time index')
  plt.ylabel('velocity')
  if b==0:
    plt.legend(loc='best')
  plt.tight_layout();





class SurrogateHeaviside(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrogateHeaviside.scale*torch.abs(input)+1.0)**2
        return grad

# here we overwrite our naive spike function by the "SurrogateHeaviside" nonlinearity which implements a surrogate gradient
surrogate_heaviside  = SurrogateHeaviside.apply


import torch
import torch.nn as nn

class SNNLayer(nn.Module):
    def __init__(self, n_in, n_out, spiking=True, tau=50, v_th=1.0):
        super(SNNLayer, self).__init__()
        self.n_in = n_in  # Number of input neurons
        self.n_out = n_out  # Number of output neurons
        self.spiking = spiking  # Whether the layer produces spiking outputs
        self.tau = tau  # Time constant of the neurons
        self.v_th = v_th  # Threshold voltage for spiking neurons

        # Initialize weights with normal distribution (4)
        self.w = nn.Parameter(torch.empty(n_in, n_out))
        nn.init.normal_(self.w, mean=0.0, std=8000)

    def forward(self, x):
        batch_size, _, num_time_points = x.shape  # Get the shape of the input tensor (1)
        device = x.device  # Get the device (CPU/GPU) of the input tensor

        # Initialize membrane potential and output tensor (3)
        v = torch.zeros(batch_size, self.n_out, device=device)  # Membrane potential
        y = torch.zeros(batch_size, self.n_out, num_time_points, device=device)  # Output tensor

        for t in range(num_time_points):
            # Compute input current (5)
            i_t = torch.matmul(x[:, :, t], self.w)  # Multiply input spikes with weights

            # Update membrane potential (5)
            dv = (i_t - v) * (ms / self.tau)  # Leaky integrate-and-fire equation
            v = v + dv  # Update membrane potential

            if self.spiking:  # (2)
                # Generate spikes (5)
                spikes = (v >= self.v_th).float()  # Spikes occur where membrane potential exceeds threshold
                # Reset membrane potential where spikes occur (5)
                """
                In this line, spikes is a tensor with values of 1 where the membrane potential v exceeds the threshold self.v_th, and 0 elsewhere. The expression 1 - spikes results in a tensor with values of 0 where spikes occurred and 1 elsewhere. Multiplying v by 1 - spikes effectively resets the membrane potential to zero where spikes occurred.
                """
                # Store spikes in output tensor (5)
                y[:, :, t] = spikes 
                
                v = v * (1 - spikes)
            else:  # (2)
                # Store membrane potential in output tensor (5)
                y[:, :, t] = v

        return y  # (5)
    
# Define the Network class
class myNetwork(nn.Module):
    def __init__(self):
        super(myNetwork, self).__init__()
        self.hidden = SNNLayer(n_in=52, n_out=100, spiking=True)
        self.output = SNNLayer(n_in=100, n_out=2, spiking=False, tau=500)
    
    def forward(self, x):
        hidden_spikes = self.hidden(x)
        output = self.output(hidden_spikes)
        return output, hidden_spikes

# Function to evaluate the network on test data
def evaluate_network(network, data_range, spike_times, vel, vel_times, dt=1e-3, length=1, batch_size=64):
    network.eval()
    total_loss = 0
    total_null_loss = 0
    criterion = nn.MSELoss()
    num_batches = 0

    with torch.no_grad():
        for x, y in batched_data(data_range, spike_times, vel, vel_times, dt=dt, length=length, batch_size=batch_size):
            output, _ = network(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            null_output = torch.zeros_like(output)
            null_loss = criterion(null_output, y)
            total_null_loss += null_loss.item()
            num_batches += 1

    mean_loss = total_loss / num_batches
    mean_null_loss = total_null_loss / num_batches
    print(f"Mean Loss: {mean_loss:.4f}")
    print(f"Mean Null Loss: {mean_null_loss:.4f}")
    return mean_loss, mean_null_loss

# Function to plot internal outputs of the network
def plot_internal_outputs(network, input_spikes, target_velocities):
    network.eval()
    with torch.no_grad():
        output, hidden_spikes = network(input_spikes)

    # Plot input spikes
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.imshow(input_spikes[0].cpu().numpy(), aspect='auto', cmap='gray_r')
    plt.title('Input Spikes')
    plt.ylabel('Neuron Index')

    # Plot hidden layer spikes
    plt.subplot(4, 1, 2)
    plt.imshow(hidden_spikes[0].cpu().numpy(), aspect='auto', cmap='gray_r')
    plt.title('Hidden Layer Spikes')
    plt.ylabel('Neuron Index')

    # Plot output velocities
    plt.subplot(4, 1, 3)
    plt.plot(output[0].cpu().numpy().T)
    plt.title('Output Velocities')
    plt.legend(['X Velocity', 'Y Velocity'])
    plt.ylabel('Velocity')

    # Plot target velocities
    plt.subplot(4, 1, 4)
    plt.plot(target_velocities[0].cpu().numpy().T)
    plt.title('Target Velocities')
    plt.legend(['X Velocity', 'Y Velocity'])
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.tight_layout()
    plt.show()

    # Compute and plot firing rates of hidden layer neurons
    firing_rates = hidden_spikes.sum(dim=2).cpu().numpy().flatten()
    plt.figure()
    plt.hist(firing_rates, bins=20)
    plt.title('Hidden Layer Neuron Firing Rates')
    plt.xlabel('Total Spikes')
    plt.ylabel('Number of Neurons')
    plt.show()

# Function to compute firing rates
def compute_firing_rates(hidden_spikes):
    # Sum over all time points to get the total number of spikes for each neuron
    total_spikes_per_neuron = hidden_spikes.sum(dim=2)
    # Sum over batches to integrate spikes from all test examples
    total_spikes_per_neuron = total_spikes_per_neuron.sum(dim=0)
    # Compute the total time duration covered by the spikes
    total_time_seconds = hidden_spikes.size(2) * ms
    # Compute firing rates: spikes per second (Hz) per neuron
    firing_rates = total_spikes_per_neuron / total_time_seconds

    print(f"Mean firing rate: {firing_rates.mean().item():.2f} Hz")
    return firing_rates

# Function to plot firing rates
def plot_firing_rates(firing_rates):
    plt.figure(figsize=(10, 6))
    plt.hist(firing_rates.numpy(), bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Firing Rates in the Hidden Layer')
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Number of Neurons')
    plt.grid(True)
    plt.show()

# Function to load test data
def load_test_data(test_range, spike_times, vel, vel_times, dt=ms, length=1, batch_size=4):
    test_data_generator = batched_data(test_range, spike_times, vel, vel_times, dt, length, batch_size)
    return next(test_data_generator)  # For example, just take the first batch for now

# Initialize the network
network = myNetwork()

# Using the function to get test spikes and velocities
test_spikes, test_velocities = load_test_data(test_range, spike_times, vel, vel_times)

# Evaluate the network
mean_loss, mean_null_loss = evaluate_network(network, test_range, spike_times, vel, vel_times)

with torch.no_grad():
    hidden_spikes = network.hidden(test_spikes)
    outputs, _ = network(test_spikes)

# Calculate firing rates
firing_rates = compute_firing_rates(hidden_spikes)  # Ensure dt is defined correctly

# Plot internal outputs
plot_internal_outputs(network, test_spikes, test_velocities)

# Plot firing rates
plot_firing_rates(firing_rates)