def _cache(f):
    """Auxiliary decorator used by :meth:`cached_property`.

    Parameters
    ----------
    f : callable
        A method with no argument (except ``self``).

    Returns
    -------
    callable
        The same function, but with a `caching` behavior.

    Notes
    -----
    In practice, the values are stored in an attribute of the object that is a dictionary called ``_cached_properties``.
    For example, consider a method ``foo(self)``.

        * If the value is not computed yet, the decorated method will compute the value, store it in
          ``self._cached_properties['foo']`` and return it.
        * If the value is already computed, the decorated method will get it from ``self._cached_properties['foo']``
          and return it.
    """
    name = f.__name__

    # noinspection PyProtectedMember
    def _f(*args):
        try:
            return args[0]._cached_properties[name]
        except (KeyError, AttributeError):
            value = f(*args)
            try:
                # Not stored in cache
                args[0]._cached_properties[name] = value
            except AttributeError:
                # Cache does not even exist
                args[0]._cached_properties = {name: value}
            return value
    _f.__doc__ = f.__doc__
    return _f


def cached_property(f):
    """Decorator used in replacement of ``@property`` to put the value in cache automatically.

    Notes
    -----
    The first time the attribute is used, it is computed on-demand and put in cache. Later accesses to the
    attributes will use the cached value.

    Cf. :class:`DeleteCacheMixin` for an example.
    """
    return property(_cache(f))


class DeleteCacheMixin:
    """Mixin used to delete cached properties.

    Notes
    -----
    Cf. decorator :meth:`cached_property`.

    Examples
    --------
        >>> class Example(DeleteCacheMixin):
        ...     @cached_property
        ...     def x(self):
        ...         print('Big computation...')
        ...         return 6 * 7
        >>> a = Example()
        >>> a.x
        Big computation...
        42
        >>> a.x
        42
        >>> a.delete_cache()
        >>> a.x
        Big computation...
        42
        >>> a.delete_cache(contains='x')
        >>> a.x
        Big computation...
        42
        >>> a.delete_cache(contains='blabla')
        >>> a.x
        42
    """

    # noinspection PyAttributeOutsideInit
    def delete_cache(self, contains='', suffix='') -> None:
        """Delete the cache (possibly only for some selected variables).

        Parameters
        ----------
        contains : str
            If specified, only the cached variables whose name contains this string will be removed from the cache.
        suffix : str
            If specified, only the cached variables whose name finishes with this string will be removed from the cache.
        """
        if not hasattr(self, '_cached_properties'):
            return
        if contains == '' and suffix == '':
            self._cached_properties = dict()
            return
        cached_properties_new = self._cached_properties.copy()
        for p in self._cached_properties:
            if contains in p and p.endswith(suffix):
                del cached_properties_new[p]
        # noinspection PyAttributeOutsideInit
        self._cached_properties = cached_properties_new


def property_deleting_cache(hidden_variable_name, doc=''):
    """Define a property that deletes the cache when set or deleted.

    Parameters
    ----------
    hidden_variable_name : str
        The name of the hidden variable used to store the value of the property.
    doc : str
        The docstring of the property.

    Notes
    -----
    The class must inherit from :class:`DeleteCacheMixin`.

    Examples
    --------
        >>> # noinspection PyUnresolvedReferences
        >>> class MyClass(DeleteCacheMixin):
        ...     def __init__(self, some_parameter):
        ...         self.some_parameter = some_parameter
        ...     some_parameter = property_deleting_cache('_some_parameter')
        ...     @cached_property
        ...     def my_cached_property(self):
        ...         print('Computing my_cached_property...')
        ...         return 'Hello %s!' % self.some_parameter
        >>> my_object = MyClass(some_parameter='World')
        >>> my_object.my_cached_property
        Computing my_cached_property...
        'Hello World!'
        >>> my_object.my_cached_property
        'Hello World!'
        >>> my_object.some_parameter = 'everybody'
        >>> my_object.my_cached_property
        Computing my_cached_property...
        'Hello everybody!'
        >>> del my_object.some_parameter
        >>> my_object.my_cached_property
        Traceback (most recent call last):
        AttributeError: 'MyClass' object has no attribute '_some_parameter'
    """

    def getter(self):
        return getattr(self, hidden_variable_name)

    def setter(self, value):
        self.delete_cache()
        setattr(self, hidden_variable_name, value)

    def deleter(self):
        self.delete_cache()
        delattr(self, hidden_variable_name)

    return property(getter, setter, deleter, doc)
