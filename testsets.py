#!/usr/bin/python
#
#  notesets.py
#
#  Provides NoteSet subclasses which can read their data from test .wav files
#  and which provide known classification data.
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


from sys import stdout;
from notes import NoteSet;
from audio import AudioFile;

# Reference frequency, for c3
c3freq = 134.0;
# Number of notes to test
testNoteCount = 36;
# Names of notes, starting from c3
noteNames = [
    "c3",
    "cs3",
    "d3",
    "ds3",
    "e3",
    "f3",
    "fs3",
    "g3",
    "gs3",
    "a3",
    "as3",
    "b3",
    "c4",
    "cs4",
    "d4",
    "ds4",
    "e4",
    "f4",
    "fs4",
    "g4",
    "gs4",
    "a4",
    "as4",
    "b4",
    "c5" ];

class SingleNote(NoteSet):
    def __init__(self, note, framesize, reference=c3freq, numNotes= testNoteCount):
        NoteSet.__init__(self,
                AudioFile("notes/"+noteNames[note]+".wav", framesize),
                reference,
                numNotes);

        self.note = note;

    def memberNotes(self, frame):
        return [self.note];

    def nonMemberNotes(self, frame):
        return range(self.note) + range(self.note+1, self.numNotes);


def readNotes(notes, framesize):
    """Reads the specified note indices, providing progress feedback"""
    stdout.write("Reading notes...\n");
    ret = [];
    for i in range(len(notes)):
        ret += [SingleNote(notes[i], framesize)];
        stdout.write("\r{0}/{1}".format(i+1,len(notes)));
        stdout.flush();
    stdout.write("\nDone\n");
    return ret;


class Octave(NoteSet):
    def __init__(self, root, framesize, reference=c3freq, numNotes= testNoteCount):
        NoteSet.__init__(self,
                AudioFile("octaves/"+noteNames[root]+".wav", framesize),
                reference,
                numNotes);

        self.root = root;

    def memberNotes(self, frame):
        return[self.root, self.root+12];

    def nonMemberNotes(self, frame):
        return range(self.root+12) + range(self.root+13,self.numNotes);


def readOctaves(roots, framesize):
    """Reads the specified note indices, providing progress feedback"""
    stdout.write("Reading octaves...\n");
    ret = [];
    for i in range(len(roots)):
        ret += [Octave(roots[i], framesize)];
        stdout.write("\r{0}/{1}".format(i+1,len(roots)));
        stdout.flush();
    stdout.write("\nDone\n");
    return ret;


class Major(NoteSet):
    def __init__(self, root, framesize, reference=c3freq, numNotes= testNoteCount):
        NoteSet.__init__(self,
                AudioFile("majors/"+noteNames[root]+".wav", framesize),
                reference,
                numNotes);

        self.root = root;

    def memberNotes(self, frame):
        return [self.root, self.root+4, self.root+7];

    def nonMemberNotes(self, frame):
        return range(self.root) + \
            range(self.root+1, self.root+4) + \
            range(self.root+5, self.root+7) + \
            range(self.root+8, self.numNotes);


def readMajors(roots, framesize):
    """Reads the specified note indices, providing progress feedback"""
    stdout.write("Reading majors...\n");
    ret = [];
    for i in range(len(roots)):
        ret += [Major(roots[i], framesize)];
        stdout.write("\r{0}/{1}".format(i+1,len(roots)));
        stdout.flush();
    stdout.write("\nDone\n");
    return ret;


class OctMajor(NoteSet):
    def __init__(self, root, framesize, reference=c3freq, numNotes=testNoteCount):
        NoteSet.__init__(self,
                AudioFile("oct-majors/"+noteNames[root]+".wav", framesize),
                reference,
                numNotes);

        self.root = root;

    def memberNotes(self, frame):
        return [self.root, self.root+12, self.root+16, self.root+19];

    def nonMemberNotes(self, frame):
        return range(self.root) + \
            range(self.root+1, self.root+12) + \
            range(self.root+13, self.root+16) + \
            range(self.root+17, self.root+19) + \
            range(self.root+20, self.numNotes);


def readOctMajors(roots, framesize):
    """Reads the specified note indices, providing progress feedback"""
    stdout.write("Reading octave-majors...\n");
    ret = [];
    for i in range(len(roots)):
        ret += [OctMajor(roots[i], framesize)];
        stdout.write("\r{0}/{1}".format(i+1,len(roots)));
        stdout.flush();
    stdout.write("\nDone\n");
    return ret;


class MajorMajor(NoteSet):
    def __init__(self, root, framesize, reference=c3freq, numNotes=testNoteCount):
        NoteSet.__init__(self,
                AudioFile("major-majors/"+noteNames[root]+".wav", framesize),
                reference,
                numNotes);

        self.root = root;

    def memberNotes(self, frame):
        return [self.root, self.root+4, self.root+7,
            self.root+12, self.root+16, self.root+19];

    def nonMemberNotes(self, frame):
        return range(self.root) + \
            range(self.root+1, self.root+4) + \
            range(self.root+5, self.root+7) + \
            range(self.root+8, self.root+12) + \
            range(self.root+13, self.root+16) + \
            range(self.root+17, self.root+19) + \
            range(self.root+20, self.numNotes);


def readMajorMajors(roots, framesize):
    """Reads the specified note indices, providing progress feedback"""
    stdout.write("Reading major-majors...\n");
    ret = [];
    for i in range(len(roots)):
        ret += [MajorMajor(roots[i], framesize)];
        stdout.write("\r{0}/{1}".format(i+1,len(roots)));
        stdout.flush();
    stdout.write("\nDone\n");
    return ret;


