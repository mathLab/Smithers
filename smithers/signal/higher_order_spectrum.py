from scipy.signal import blackman
from scipy.fftpack import fft

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class HigherOrderSpectrum(object):
    """
    Higher order spectrum analysis
    """
    def __init__(self, time, signal, window_size=None, n_averages=1):

        if len(time) != len(signal):
            raise ValueError

        self._signal = signal
        self._time = time
        self.window_size = window_size
        self.n_averages = n_averages

    @property
    def freq(self):
        """
        """
        return np.fft.fftfreq(self.window_size, d=self.dt)

    @property
    def dt(self):
        """
        """
        return self.time[1] - self.time[0]

    @property
    def time(self):
        """
        """
        rem = int(len(self._time) % self.window_size)
        return self._time[:-rem]

    @property
    def signal(self):
        rem = int(len(self._signal) % self.window_size)
        return self._signal[:-rem]

    @property
    def window_size(self):
        """
        The length of the time window.

        :rtype: int
        """
        return self._window_size

    @window_size.setter
    def window_size(self, size):

        if size is None:
            self._window_size = self._signal.shape[0]
        else:
            self._window_size = size

        self.blackman_window = blackman(self.window_size)
        self.windowed_time = np.array(np.split(
            self.time, np.floor(len(self.time)/(self.window_size))))
        self.windowed_sign = np.array(np.split(
            self.signal, np.floor(len(self.signal)/(self.window_size))))

        self._windowed_fft = None

    @property
    def windowed_fft(self):
        """
        The FFT for all the time windows.
        """
        if self._windowed_fft is None:
            self._windowed_fft = fft( (self.windowed_sign.T -
                                       np.average(self.windowed_sign,
                                                  axis=1)).T *
                                     self.blackman_window, axis=1)
        return self._windowed_fft


    @property
    def n_averages(self):
        """
        """
        return self._n_averages


    @n_averages.setter
    def n_averages(self, n):
        try:
            self._n_averages = int(n)
        except:
            raise TypeError


    @property
    def spectrum(self):
        """
        """
        return np.mean(np.abs(self.windowed_fft[:self.n_averages, :]), axis=0)


    def ncoherence(self, window_fraction=1/16, n=2):

        reduced_window_size = int(self.window_size * window_fraction)
        ranges = [reduced_window_size] * n

        term = np.zeros(shape=self.n_averages, dtype=np.complex)
        coherence = np.zeros(shape=ranges)

        Y = self.windowed_fft[:self.n_averages]
        for i in np.ndindex(*ranges):
            if all(i) is False: continue
            term[:] = np.prod(Y[:, i], axis=1) * np.conj(Y[:, sum(i)])
            coherence[i] = np.abs(np.mean(term))/(np.mean(np.abs(term)))

        print(coherence)

        return coherence

    def bicoherence(self, window_fraction=1/16):
        return self.ncoherence(window_fraction, n=2)

    def tricoherence(self, window_fraction=1/16):
        return self.ncoherence(window_fraction, n=3).transpose((1, 0, 2))

    def plot_spectrum(self):
        """
        """
        plt.plot(self.freq, self.spectrum)
        plt.title('Bicoherence')
        plt.ylabel('$f_2$')
        plt.xlabel('$f_1$')


    def plot_bicoherence(self, bicoherence):
        """
        """

        f1, f2 = np.meshgrid(self.freq[:bicoherence.shape[0]], self.freq[:bicoherence.shape[0]])

        #surf = ax.plot_surface(f1, f2, bicoherence, rstride=1, cstride=1,cmap=cm.coolwarm)
        plot = plt.pcolor(f1, f2, bicoherence, cmap=cm.coolwarm)
        plt.colorbar(plot)
        plt.title('Bicoherence')
        plt.ylabel('$f_2$')
        plt.xlabel('$f_1$')

    def plot_tricoherence(self, tricoherence):
        """
        """
        f1, f2, f3 = np.meshgrid(
            self.freq[:tricoherence.shape[0]],
            self.freq[:tricoherence.shape[0]],
            self.freq[:tricoherence.shape[0]])
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(f1.ravel(), f2.ravel(), f3.ravel(), marker='o', alpha=.5, s=np.exp(tricoherence.ravel()))
