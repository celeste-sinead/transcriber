#!/usr/bin/python
#
#  feature-plots.py
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

import matplotlib.pyplot as plt
import numpy as np
from audio import *
from notes import *
from testsets import c3freq;

c4 = AudioFile("notes/c4.wav",44100/8);
n = NoteSet(c4, c3freq, 36);

plt.title("Feature values accross spectrum of a single note");
plt.xlabel("Frequency (Hz)");
plt.plot(c4.fftFreqs[1:150], np.divide(np.absolute(c4.fftFrames[4,1:150]), 25000));
plt.plot(n.noteFreqs, n.deviations(4));
plt.plot(n.noteFreqs, n.peakinesses(4));
plt.plot(n.noteFreqs, n.relFundamentals(4));

plt.show();

