#!/usr/bin/python
#
#  audio.py
#
#  Provides functionality for reading and basic frequency-domain
#  analysis of audio files.
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
from numpy.fft import fft;
import wave;
import math;

class AudioFile:
    """
    Reads a wav file, splits it into a sequence of frames, and computes the
    FFT of each frame, providing the time-varying frequency spectrum of the
    audio file.
    """

    def __init__(self, filename, framesize):
        # Actually read the file
        self._readWav(filename);

        # Split into FFT frames and run FFTs
        self.framesize = framesize;
        self._fft(framesize);

    def _readWav(self, filename):
        """Reads the right channel of a wav file into a numpy array"""
        inFile = wave.open(filename);

        # Assuming 16bit dual-channel, there are 4 bytes per sample:
        self.wav = np.empty(inFile.getnframes());

        # Read file into array
        for i in range(inFile.getnframes()):
            frame = bytearray(inFile.readframes(1));
            if len(frame) == 0:
                raise Exception("Unexpected end of file");

            # Frame is one sample, two channels, 16 bit, little endian.
            # Right channel is first 2 bytes.
            right = 0;
            # Bash the little-endian signed int bytes into a python int
            if (frame[1]&0x80) != 0:
                # Negative, need some silliness to get signedness past python
                right = int( ~0xffff | frame[0] | (frame[1]<<8) );
            else:
                right = frame[0] | (frame[1]<<8);

            self.wav[i] = right;

    def _fft(self, framesize):
        """Calculates FFTs of individual audio frames"""
        self.fftFrames = np.empty([len(self.wav)/framesize, framesize]);

        # Split wav into frames, executing fft
        for i in range(len(self.wav)/framesize):
            self.fftFrames[i] = fft(self.wav[i*framesize : (i+1)*framesize]);

        # Calculate the frequencies of all of the FFTs, in Hz
        # (Note that the index in the fft is also frequency, in cycles/frame)
        self.fftFreqs = np.array(range(framesize));
        self.fftFreqs = np.multiply(self.fftFreqs, 44100.0/framesize);

        # Calculate means and stdevs for all of the frames
        self.fftMeans = np.array( [ np.mean(np.absolute(self.fftFrames[i]))
                for i in range(len(self.fftFrames)) ] );
        self.fftStDevs = np.array( [ np.std(np.absolute(self.fftFrames[i]))
                for i in range(len(self.fftFrames)) ] );

    def numFrames(self):
        """Returns the number of FFT frames the audio has been split into"""
        return len(self.fftFrames);

    def indexOf(self, frequency):
        """
        Determine the index of a given frequency within FFT data.
        Index is floating point, with a fraction returned for frequencies
        between actual FFT data points.
        """
        return float(frequency) * self.framesize / 44100;

    def magnitude(self, frame, inx):
        """
        Get the magnitude of the FFT for a given index.
        If the index is fractional, magnitude is interpolated.
        """
        lower = np.absolute(self.fftFrames[frame, math.floor(inx)]);
        upper = np.absolute(self.fftFrames[frame, math.ceil(inx)]);

        ret = lower + (inx - math.floor(inx))*(upper-lower);

        return ret;

    def deviation(self, frame, inx):
        """
        Return the interpolated deviation from the mean, in standard deviations,
        at a given index.  Result is interpolated if inx is fractional.
        """
        mag = self.magnitude(frame, inx);
        return (mag - self.fftMeans[frame]) / self.fftStDevs[frame];

    def deviations(self, frame):
        """Convenience returningg deviations for all frequencies in FFT"""
        return [ (i - self.fftMeans[frame]) / self.fftStDevs[frame] for
           i in self.fftFrames[frame] ];


