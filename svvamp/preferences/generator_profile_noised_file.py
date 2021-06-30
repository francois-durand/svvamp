# -*- coding: utf-8 -*-
"""
Created on nov. 29, 2018, 09:56
Copyright François Durand 2018
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
"""
from svvamp.preferences.generator_profile_noise import GeneratorProfileNoise
from svvamp.preferences.profile_from_file import ProfileFromFile


class ProfileGeneratorNoisedFile(GeneratorProfileNoise):
    """Profile generator loading a file, then adding noise

    Parameters
    ----------
    file_name : str
        The file for the initial profile.
    relative_noise : number
        The relative noise.
    absolute_noise : number
        The absolute noise

    Notes
    -----
    This class is just a combination of :class:`ProfileFromFile` and :class:`GeneratorProfileNoise`.
    """

    def __init__(self, file_name, relative_noise=0., absolute_noise=0.):
        self.file_name = file_name
        super().__init__(base_profile=ProfileFromFile(file_name=file_name, sort_voters=False),
                         relative_noise=relative_noise, absolute_noise=absolute_noise)
        self.log_creation = ['Noised File', 'File name', file_name,
                             'Relative noise', relative_noise, 'Absolute noise', absolute_noise]
