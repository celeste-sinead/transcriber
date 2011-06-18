#!/usr/bin/python
#
#  notes.py
#
#  Provides simple frequency-domain analyses of audio intended to provide
#  insight into whether a given note is present in the audio.
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
import math;
from audio import *;

class NoteSet:
    def __init__(self, audioFile, reference, numNotes):
        """
        Create a set of not fundamental frequencies.  audioFile is the
        AudioFile instance whose data will be analyzed.
        reference is the frequency of the bottommost reference note.
        noteCount is the number of semitones above the reference to
        be available for analysis.
        """
        self.audioFile = audioFile;
        self.numNotes = numNotes;

        self.noteFreqs = np.array(
                [reference*pow(2,float(i)/12) for i in range(numNotes)] );

        # These are cached on first access
        self.memberCache = None;
        self.nonMemberCache = None;

    def notePeak(self, frame, frequency):
        """
        Finds the peak deviation within a one-semitone band of
        the given frequency, in the given FFT frame.
        """
        lowerInx = int(math.ceil(self.audioFile.indexOf(
                    frequency/pow(2.0,1.0/24))));
        upperInx = int(math.ceil(self.audioFile.indexOf(
                    frequency*pow(2.0,1.0/24))));

        if(lowerInx == upperInx):
            raise Exception("Insufficient frequency resolution");

        return np.max([self.audioFile.deviation(frame,i)
                for i in range(lowerInx, upperInx)]);

    def notePeakInx(self, frame, frequency):
        """
        Finds the index of the peak deviation within a one-semitone band
        of the given frequency, in the given FFT frame
        """
        lowerInx = int(math.ceil(self.audioFile.indexOf(
                    frequency/pow(2.0,1.0/24))));
        upperInx = int(math.ceil(self.audioFile.indexOf(
                    frequency*pow(2.0,1.0/24))));

        if(lowerInx == upperInx):
            raise Exception("Insufficient frequency resolution");

        maxInx = lowerInx;
        for i in range(lowerInx+1,upperInx):
            if(self.audioFile.deviation(frame,i) >
                    self.audioFile.deviation(frame, maxInx)):
                maxInx = i;

        return maxInx;


    def getNumFrames(self):
        """Convenience providing access to the AudioFile frame count"""
        return self.audioFile.numFrames();

    def getNumNotes(self):
        """Get the number of notes available for analysis."""
        return self.numNotes;

    def deviation(self, frame, note):
        """The deviation from the mean for a given note"""
        return self.notePeak(frame, self.noteFreqs[note]);

    def deviations(self, frame):
        """Convenience returning deviations for all notes"""
        return np.array(
                [self.notePeak(frame, i) for i in self.noteFreqs] );

    def peakiness(self, frame, note):
        """
        Measure of how much of a peak is located within a given note band.
        This is just the second derivative at the peak
        """
        peakInx = self.notePeakInx(frame, self.noteFreqs[note]);
        left = self.audioFile.deviation(frame, peakInx-1);
        center = self.audioFile.deviation(frame, peakInx);
        right = self.audioFile.deviation(frame, peakInx+1);
        return center - 0.5*(right + left);

    def peakinesses(self, frame):
        """Convenience returning peakinesses for all notes"""
        return np.array(
                [self.peakiness(frame,i) for i in range(self.numNotes)]);

    def relFundamental(self, frame, note):
        """
        How large is the note, relative to its fundamentals?
        """

        numFuns = 3;
        funMags = np.empty(numFuns);
        for i in range(numFuns):
            try:
                funMags[i] = self.notePeak(frame, self.noteFreqs[note]/(i+2));
            except Exception: # "Insufficient Frequency Resolution"
                # We'll be less fussy about frequency resolution here and
                # just go with it if we can't resolve this note's fundamental
                # from the next
                funMags[i] = self.audioFile.deviation(frame,
                        self.audioFile.indexOf(self.noteFreqs[note]/(i+2)));

        noteMag = self.notePeak(frame, self.noteFreqs[note]);

        return noteMag - np.max(funMags);

    def relFundamentals(self, frame):
        """Convenience returning fundamentalRels for all notes"""
        return np.array(
                [self.relFundamental(frame,i) for i in range(self.numNotes)]);

    def feature(self, frame, note):
        """Returns a feature vector for a given note in a given frame"""
        return np.array([ self.deviation(frame, note),
                self.peakiness(frame, note), self.relFundamental(frame, note)]);

    def featureLen(self):
        """Returns the length of feature vectors given by feature()"""
        return 3;

    def memberNotes(self, frame):
        """
        Returns a list of notes which are known to be sounding in a given frame.
        This is intended to be overridden.
        """
        return np.empty(0);

    def nonMemberNotes(self, frame):
        """
        Returns a list of notes which are known not to be sounding
        in a given frame. This is intended to be overridden.
        """
        return np.empty(0);

    def memberFeatures(self):
        """
        Returns an array of all feature vectors which correspond to member notes.
        """
        if not self.memberCache is None:
            return self.memberCache;

        members = np.empty([0,self.featureLen()]);
        for i in range(self.getNumFrames()):
            if(len(self.memberNotes(i)) > 0):
                members = np.concatenate( (members,
                            [self.feature(i,j) for j in self.memberNotes(i)]) );

        self.memberCache = members;
        return members;

    def nonMemberFeatures(self):
        """
        Returns an array of all feature vectors which do not correspond to
        member notes.
        """
        if not self.nonMemberCache is None:
            return self.nonMemberCache;

        nonmem = np.empty([0,self.featureLen()]);
        for i in range(self.getNumFrames()):
            if(len(self.nonMemberNotes(i)) > 0):
                nonmem = np.concatenate( (nonmem,
                            [self.feature(i,j) for j in self.nonMemberNotes(i)]) );

        self.nonMemberCache = nonmem;
        return nonmem;

