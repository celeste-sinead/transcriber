#!/usr/bin/python
#
#  bayes.py
#
#  Provides functionality for naive bayes learning and classification.
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

import random;
import math;
import numpy as np;
from sys import stdout;

def gauss(x, u, v):
    """
    Compute the elementwise gaussian probability density of x,
    for means u and variances v.
    """
    exp = np.subtract(x, u);
    exp = np.power(exp, 2);
    exp = np.divide(exp, np.multiply(v, 2));
    exp = np.multiply(exp, -1);
    exp = np.power(math.e, exp);
    den = np.multiply(2*math.pi, v);
    den = np.sqrt(den);
    return np.divide(exp, den);


class NaiveBayes:
    def __init__(self):
        self.learnMembers = None;
        self.learnNonMembers = None;
        self.testMembers = None;
        self.testNonMembers = None;

    def addLearningData(self, noteSets):
        """Add a list of NoteSet instances to the learning dataset."""
        if self.learnMembers is None:
            self.learnMembers = np.empty([0, noteSets[0].featureLen()]);
        if self.learnNonMembers is None:
            self.learnNonMembers = np.empty([0, noteSets[0].featureLen()]);

        for noteSet in noteSets:
            self.learnMembers = np.concatenate((self.learnMembers,
                    noteSet.memberFeatures()));
            self.learnNonMembers = np.concatenate((self.learnNonMembers,
                    notSet.nonMemberFeatures()));

    def addTestingData(self, noteSets):
        """Add a list of NoteSet instances to the training dataset."""
        if self.testMembers is None:
            self.testMembers = np.empty([0, noteSets[0].featureLen()]);
        if self.testNonMembers is None:
            self.testNonMembers = np.empty([0, noteSets[0].featureLen()]);

        for noteSet in noteSets:
            self.testMembers = np.concatenate((self.testMembers,
                    noteSet.memberFeatures()));
            self.testNonMembers = np.concatenate((self.testNonMembers,
                    notSet.nonMemberFeatures()));

    def addLabelledData(self, noteSets):
        """
        Add labelled data to be used for learning and testing. Each item
        of labelled data will be randomly assigned to either the learning
        or the testing data sets.
        noteSets should be a list of NoteSet instances which provide known
        member and nonMember note lists
        """
        featureLen = noteSets[0].featureLen();
        if self.learnMembers is None:
            self.learnMembers = np.empty([0, featureLen]);
        if self.learnNonMembers is None:
            self.learnNonMembers = np.empty([0, featureLen]);
        if self.testMembers is None:
            self.testMembers = np.empty([0, featureLen]);
        if self.testNonMembers is None:
            self.testNonMembers = np.empty([0, featureLen]);

        for noteSet in noteSets:
            for i in range(len(noteSet.memberFeatures())):
                if random.random() < 0.5:
                    self.learnMembers = np.concatenate((
                            self.learnMembers,
                            noteSet.memberFeatures()[i].reshape((1,featureLen))));
                else:
                    self.testMembers = np.concatenate((
                            self.testMembers,
                            noteSet.memberFeatures()[i].reshape((1,featureLen))));

            for i in range(len(noteSet.nonMemberFeatures())):
                if random.random() < 0.5:
                    self.learnNonMembers = np.concatenate((
                            self.learnNonMembers,
                            noteSet.nonMemberFeatures()[i].reshape((1,featureLen))));
                else:
                    self.testNonMembers = np.concatenate((
                            self.testNonMembers,
                            noteSet.nonMemberFeatures()[i].reshape((1,featureLen))));

    def learn(self):
        """Train the naive bayes classifier from available labelled data"""

        if(len(self.learnMembers)==0):
            raise Exception("Learning failed: no positive cases.");
        if(len(self.learnNonMembers)==0):
            raise Exception("Learning failed: no negative cases.");

        # Calculate mean/var for all features and both cases
        self.memberMeans = np.mean(self.learnMembers, 0);
        self.memberVars = np.var(self.learnMembers, 0);
        self.nonMemberMeans = np.mean(self.learnNonMembers, 0);
        self.nonMemberVars = np.var(self.learnNonMembers, 0);

        # Print out learning results:
        print("Learning Results:");
        print("Member:    Mean {0}, Var {1}".format(
                    self.memberMeans, self.memberVars));
        print("NonMember: Mean {0}, Var {1}".format(
                    self.nonMemberMeans, self.nonMemberVars));

    def isMember(self, feature):
        """Classify the given feature vector"""

        # Compute the probabilites of each hypothesis, with the assumption that
        # the hypotheses have equal prior probabilities.
        pMember = np.product(
                gauss(feature, self.memberMeans, self.memberVars));
        pNonMember = np.product(
                gauss(feature, self.nonMemberMeans, self.nonMemberVars));

        # Is the feature a member?
        return pMember >= pNonMember;

    def _accuracy(self, members, nonMembers):
        """Find the accuracy of the classifier when classifying the given dataset"""
        corMembers = 0;
        for i in range(len(members)):
            if self.isMember(members[i]):
                corMembers += 1;

        corNonMembers = 0;
        for i in range(len(nonMembers)):
            if not self.isMember(nonMembers[i]):
                corNonMembers += 1;

        return (corMembers, corNonMembers);

    def learningAccuracy(self):
        """Find the classifier accuracy when classifying learning data"""
        accuracy = self._accuracy(self.learnMembers, self.learnNonMembers);

        print("Learning Accuracy:");
        print("Positive: {0}/{1}".format(accuracy[0], len(self.learnMembers)));
        print("Negative: {0}/{1}".format(accuracy[1], len(self.learnNonMembers)));

        return (accuracy[0], len(self.learnMembers),
                accuracy[1], len(self.learnNonMembers));

    def testingAccuracy(self):
        """Find the classifier accuracy when classifying testing data"""
        accuracy = self._accuracy(self.testMembers, self.testNonMembers);

        print("Testing Accuracy:");
        print("Positive: {0}/{1}".format(accuracy[0], len(self.testMembers)));
        print("Negative: {0}/{1}".format(accuracy[1], len(self.testNonMembers)));

        return (accuracy[0], len(self.testMembers),
                accuracy[1], len(self.testNonMembers));

