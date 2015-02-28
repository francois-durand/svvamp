# -*- coding: utf-8 -*-
"""
Created on oct. 30, 2014, 23:52 
Copyright François Durand 2014, 2015
fradurand@gmail.com

This file is part of SVVAMP.

    SVVAMP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SVVAMP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SVVAMP.  If not, see <http://www.gnu.org/licenses/>.

==============

The function preflib_to_preferences_utilities below is adapted from
PreflibUtils.py by Nicholas Mattei. We reproduce its license here.

	File: 	PrefLibUtilities.py
	Author:	Nicholas Mattei (nicholas.mattei@nicta.com.au)
	Date:	April 4, 2013
			November 6th, 2013

  * Copyright (c) 2014, Nicholas Mattei and NICTA
  * All rights reserved.
  *
  * Developed by: Nicholas Mattei
  *               NICTA
  *               http://www.nickmattei.net
  *               http://www.preflib.org
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *     * Redistributions of source code must retain the above copyright
  *       notice, this list of conditions and the following disclaimer.
  *     * Redistributions in binary form must reproduce the above copyright
  *       notice, this list of conditions and the following disclaimer in the
  *       documentation and/or other materials provided with the distribution.
  *     * Neither the name of NICTA nor the
  *       names of its contributors may be used to endorse or promote products
  *       derived from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY NICTA ''AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL NICTA BE LIABLE FOR ANY
  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pandas as pd

from svvamp.Preferences.Population import \
    preferences_ut_to_preferences_borda_ut
from svvamp.Preferences.Population import Population


class PopulationFromFile(Population):

    _layout_name = 'From file'

    def __init__(self, file_name, relative_noise=0., absolute_noise=0.):
        """Population from a file.

        :param file_name: -- String. The name of the file.
        :param relative_noise: -- Number.
        :param absolute_noise: -- Number.

        :return: A :class:`~svvamp.Population` object.

        If the file name ends with '.t.csv' (t = transposed format): simple
        table of utilities with candidates declared in the first column and
        voters declared in the first row.

        If the file name ends with '.csv' (but not '.t.csv'), candidates
        must be declared in the first row and voters in the first column.

        Otherwise, the file is considered as a PrefLib file. In that case,
        since information is ordinal only,
        ``preferences_ut[v, c]`` is set to the Borda score
        (with no vtb) minus ``(C - 1) / 2``. This way, utilities are between
        ``- (C - 1) / 2`` and ``(C - 1) / 2``.

        To each ``preferences_ut[v, c]``, a random noise is added which
        is drawn independently and uniformly in the interval
        ``[- relative_noise * amplitude, relative_noise * amplitude]``,
        where ``amplitude`` is the difference between the lowest and the
        highest utility.

        Another random noise is added, which is drawn independently and
        uniformly in the interval ``[-absolute_noise, absolute_noise]``.
        """
        if file_name[-4:] == '.csv':
            df = pd.read_csv(filepath_or_buffer=file_name, sep=';',
                             index_col=0)
            if file_name[-6:-4] == '.t':
                preferences_utilities = df.transpose().values
                labels_candidates = df.index.values.astype(np.str)
            else:
                preferences_utilities = df.values
                labels_candidates = df.columns.values.astype(np.str)
            pop_temp = Population(preferences_utilities)
            nb_victories_temp = np.sum(pop_temp.matrix_victories_ut_rel, 1)
            scores_temp = (nb_victories_temp +
                           pop_temp.borda_score_c_ut / pop_temp.C /
                           pop_temp.V)
            candidates_best_to_worst = np.argsort(- scores_temp,
                                                  kind='mergesort')
            preferences_utilities = preferences_utilities[
                :, candidates_best_to_worst]
            labels_candidates = labels_candidates[candidates_best_to_worst]
        else:
            preferences_utilities, labels_candidates = (
                preflib_to_preferences_utilities(file_name))

        preferences_utilities = preferences_utilities.astype(np.float)
        total_noise = absolute_noise
        if relative_noise != 0:
            amplitude = np.max(preferences_utilities) - np.min(
                preferences_utilities)
            total_noise += relative_noise * amplitude
        if total_noise != 0:
            preferences_utilities += total_noise * 2 * (
                0.5 - np.random.rand(*preferences_utilities.shape))

        log_creation = ['From file',
                        'File name', file_name,
                        'Relative noise', relative_noise,
                        'Absolute noise', absolute_noise]
        super().__init__(preferences_ut=preferences_utilities,
                         log_creation=log_creation,
                         labels_candidates=labels_candidates)

    # TODO: iterator and meta-iterator
    @staticmethod
    def iterator(culture_parameters, nb_populations):
        for i in range(nb_populations):
            yield PopulationFromFile(**culture_parameters)

    # @staticmethod
    # def meta_iterator(culture_parameters_list, nb_populations):
    #     for C, V, culture_parameters in itertools.product(
    #             C_list, V_list, culture_parameters_list):
    #         log_csv = ['Euclidean box',
    #                    'Box dimensions', culture_parameters['box_dimensions'],
    #                    'Number of dimensions',
    #                    len(culture_parameters['box_dimensions'])]
    #         log_print = ('Euclidean box, V = ' + str(V) + ', C = ' + str(C) +
    #                      ', box dimensions = ' +
    #                      format(culture_parameters['box_dimensions']))
    #         yield log_csv, log_print, PopulationFromFile.iterator(
    #             C, V, culture_parameters, nb_populations)


def preflib_to_preferences_utilities(file_name):
    # TODO: Is the copyright mention above enough? Ask Nicholas Mattei.
    # PreflibUtils.py
    input_file = open(file_name, 'r')
    # Number of candidates
    l = input_file.readline()
    # Map of candidates
    C = int(l.strip())
    candidates_map = {}
    labels_candidates = []
    for c in range(C):
        bits = input_file.readline().strip().split(",")
        candidates_map[int(bits[0].strip()) - 1] = bits[1].strip()
        labels_candidates.append(bits[1].strip())
    # Now we have V, sum_of_vote_count, num_unique orders
    bits = input_file.readline().strip().split(",")
    V = int(bits[0].strip())
    sum_votes = int(bits[1].strip())
    unique_orders = int(bits[2].strip())
    # Now, the ballots themselves
    preferences_utilities = np.zeros((V, C))
    start_index = 0
    for i in range(unique_orders):
        rec = input_file.readline().strip()
        #need to parse the rec properly..
        count = int(rec[:rec.index(",")])
        bits = rec[rec.index(",")+1:].strip().split(",")
        ballot_temp = np.full(C, - C)
        if rec.find("{") == -1:
            #its strict, just split on ,
            for crank in range(len(bits)):
                ballot_temp[int(bits[crank]) - 1] = - crank
        else:
            crank = 0
            partial = False
            for ccand in bits:
                if ccand.find("{") != -1:
                    partial = True
                    t = ccand.replace("{","")
                    ballot_temp[int(t.strip()) - 1] = - crank
                elif ccand.find("}") != -1:
                    partial = False
                    t = ccand.replace("}","")
                    ballot_temp[int(t.strip()) - 1] = - crank
                    crank += 1
                else:
                    ballot_temp[int(ccand.strip()) - 1] = - crank
                    if partial == False:
                        crank += 1
        preferences_utilities[
            start_index:start_index + count, :] = \
            preferences_ut_to_preferences_borda_ut(
                ballot_temp[np.newaxis, :]) - (C - 1) / 2
        start_index += count
    input_file.close()
    return preferences_utilities, labels_candidates


if __name__ == '__main__':
    # pop = PopulationFromFile('ED-00001-00000001.soi')
    pop = PopulationFromFile('example_ballots.t.csv',
                             absolute_noise=0.5)
    # preferences_ut = pop.preferences_ut[:, 0:3]
    # preferences_ut -= np.mean(preferences_ut, 1)[:, np.newaxis]
    # pop_temp = Population(preferences_ut)
    pop.demo()
    pop.plot3(use_labels=True)
    pop.plot4(use_labels=True)
