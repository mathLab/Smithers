import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from smithers.signal import HigherOrderSpectrum as HOS

signal = np.load(
    'tests/test_datasets/duffing_10000.0_10000000.0_5000000000.0.signal.npy')
time = np.load(
    'tests/test_datasets/duffing_10000.0_10000000.0_5000000000.0.time.npy')


def test_init():
    hos = HOS(time, signal, window_size=512)

def test_spectrum():
    hos = HOS(time, signal, window_size=512, n_averages=20)
    print(hos.spectrum)

def test_plot_spectrum():
    for i in range(1, 20, 2):
        hos = HOS(time, signal, window_size=512, n_averages=i)
        hos.plot_spectrum()

def test_plot_bicoherence():
    for i in range(2, 20, 2):
        plt.figure()
        hos = HOS(time, signal, window_size=2048, n_averages=i)
        hos.plot_bicoherence(hos.bicoherence(1/16))
