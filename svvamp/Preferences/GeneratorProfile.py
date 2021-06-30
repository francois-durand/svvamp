# -*- coding: utf-8 -*-
"""
Created on nov. 28, 2018, 13:42
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
from svvamp.Utils.Cache import cached_property, DeleteCacheMixin


class GeneratorProfile(DeleteCacheMixin):
    """A generator of profiles.

    For examples, cf. :class:`GeneratorProfileEuclideanBox`.
    """

    def __init__(self):
        """Store the parameters of the generator.
        """
        pass

    def __call__(self):
        """Erase the cache (in particular, the last generated profile).

        :return: the generator itself.
        """
        self.delete_cache()
        return self

    @cached_property
    def profile_(self):
        """The profile generated.

        :return: A :class:`~svvamp.Profile` object.
        """
        raise NotImplementedError
