class _Options:
    def __init__(self):
        self.ERROR_WHEN_UNCOVERED_CODE = False
        """bool. If True, throw an AssertionError when an uncovered portion of code is reached."""
        self.PRINT_UNCOVERED_CODE = True
        """bool. If True, print a message when an uncovered portion of code is reached (only the first time for each
        rule class, method and additional message)."""


OPTIONS = _Options()
