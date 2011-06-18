#!/usr/bin/python
#
#  note-plots.py
#
#  Produces some illustrative plots showing musical data in the
#  time and frequency domains.
#
#  This program is distributed under the of the GNU Lesser Public License.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>

import numpy as np;
import matplotlib.pyplot as plt;
import audio;

c4 = audio.AudioFile("notes/c4.wav",44100);
plt.figure(1);

plt.subplot(211);
plt.title("Middle C in the Time Domain");
plt.xlabel("Time (s)");
plt.plot(np.multiply(range(len(c4.wav)), 1.0/44100), c4.wav);

plt.subplot(212);
plt.title("Middle C in the Frequency Domain");
plt.xlabel("Frequency (Hz)");
plt.plot(c4.fftFreqs, np.absolute(c4.fftFrames[0]));

cmaj = audio.AudioFile("majors/c4.wav", 44100);
plt.figure(2);

plt.subplot(211);
plt.title("C Major in the Time Domain");
plt.xlabel("Time (s)");
plt.plot(np.multiply(range(len(cmaj.wav)), 1.0/44100), cmaj.wav);

plt.subplot(212);
plt.title("C Major in the Frequency Domain");
plt.xlabel("Frequency (Hz)");
plt.plot(cmaj.fftFreqs, np.absolute(cmaj.fftFrames[0]));

c4 = audio.AudioFile("notes/c4.wav",4096);
e4 = audio.AudioFile("notes/e4.wav",4096);
g4 = audio.AudioFile("notes/g4.wav",4096);

plt.figure(3);
plt.title("Spectra of each C Major Triad Note");
plt.xlabel("Frequency (Hz)");
plt.semilogy(c4.fftFreqs, np.absolute(c4.fftFrames[6]));
plt.semilogy(e4.fftFreqs, np.absolute(e4.fftFrames[6]));
plt.semilogy(g4.fftFreqs, np.absolute(g4.fftFrames[6]));

plt.show();

