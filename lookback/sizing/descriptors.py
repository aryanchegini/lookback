"""Reusable validating descriptors: the rule lives on the attribute, not
in each class's __init__, and runs on every assignment."""


class Fraction01:
    """A float attribute constrained to [0, 1]."""

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name)

    def __set__(self, obj, value):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{self._name[1:]} must be in [0, 1], got {value}")
        setattr(obj, self._name, float(value))


class PositiveInt:
    """An int attribute constrained to > 0."""

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name)

    def __set__(self, obj, value):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{self._name[1:]} must be a positive int, got {value!r}")
        setattr(obj, self._name, value)


class PositiveFloat:
    """A float attribute constrained to > 0 (unbounded above)."""

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self._name)

    def __set__(self, obj, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{self._name[1:]} must be a positive number, got {value!r}")
        setattr(obj, self._name, float(value))
