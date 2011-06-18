#!/usr/bin/python
#
#  proj.py
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
from sys import stdout;

from audio import *;
from notes import *;
from testsets import *;
from bayes import *;

framesize = 44100/8;

# Read all data
datasets = [
    readNotes(range(0,25), framesize),
    readMajors(range(0,25), framesize),
    readOctaves(range(0,13), framesize),
    readOctMajors(range(0,13), framesize),
    readMajorMajors(range(0,13), framesize)
    ];

# Construct Naive Bayes classifiers for all datasets
nbs = [NaiveBayes() for i in range(6)];
# The single-dataset NBs:
for i in range(5):
    nbs[i].addLabelledData(datasets[i]);
    nbs[i].learn();
# The 'all' NB:
for data in datasets:
    nbs[5].addLabelledData(datasets[i]);
nbs[5].learn();

# Titles for all datasets:
titles = [
    "Single Notes",
    "Majors",
    "Octaves",
    "Octave-Majors",
    "Double Majors",
    "All"
    ];

# Print a lovely Latex table of results.
table = "";
for i in range(len(nbs)):
    table += titles[i];
    learn = nbs[i].learningAccuracy();
    table += " & {0:.3}\\%".format(100.0*learn[0]/learn[1]);
    table += " & {0:.3}\\%".format(100.0*learn[2]/learn[3]);
    test = nbs[i].testingAccuracy();
    table += " & {0:.3}\\%".format(100.0*test[0]/test[1]);
    table += " & {0:.3}\\%".format(100.0*test[2]/test[3]);
    table += " \\\\ \n\\hline\n";

print(table);

