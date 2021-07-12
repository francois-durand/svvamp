# -*- coding: utf-8 -*-
"""
Created on oct. 30, 2014, 23:52
Copyright François Durand 2014-2018
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

The function preflib_to_preferences_ut below is adapted from
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
from svvamp.utils.misc import preferences_ut_to_preferences_borda_ut


def preflib_to_preferences_ut(file_name):
    # PreflibUtils.py
    input_file = open(file_name, 'r')
    # Number of candidates
    the_line = input_file.readline()
    # Map of candidates
    n_c = int(the_line.strip())
    candidates_map = {}
    labels_candidates = []
    for c in range(n_c):
        bits = input_file.readline().strip().split(",")
        candidates_map[int(bits[0].strip()) - 1] = bits[1].strip()
        labels_candidates.append(bits[1].strip())
    # Now we have V, sum_of_vote_count, num_unique orders
    bits = input_file.readline().strip().split(",")
    n_v = int(bits[0].strip())
    unique_orders = int(bits[2].strip())
    # Now, the ballots themselves
    preferences_utilities = np.zeros((n_v, n_c))
    start_index = 0
    for i in range(unique_orders):
        rec = input_file.readline().strip()
        # need to parse the rec properly..
        count = int(rec[:rec.index(",")])
        bits = rec[rec.index(",")+1:].strip().split(",")
        ballot_temp = np.full(n_c, - n_c)
        if rec.find("{") == -1:
            # its strict, just split on ,
            for crank in range(len(bits)):
                ballot_temp[int(bits[crank]) - 1] = - crank
        else:
            crank = 0
            partial = False
            for ccand in bits:
                if ccand.find("{") != -1:
                    partial = True
                    t = ccand.replace("{", "")
                    ballot_temp[int(t.strip()) - 1] = - crank
                elif ccand.find("}") != -1:
                    partial = False
                    t = ccand.replace("}", "")
                    ballot_temp[int(t.strip()) - 1] = - crank
                    crank += 1
                else:
                    ballot_temp[int(ccand.strip()) - 1] = - crank
                    if not partial:
                        crank += 1
        preferences_utilities[start_index:start_index + count, :] = (
            preferences_ut_to_preferences_borda_ut(ballot_temp[np.newaxis, :]) - (n_c - 1) / 2)
        start_index += count
    input_file.close()
    return preferences_utilities, labels_candidates
