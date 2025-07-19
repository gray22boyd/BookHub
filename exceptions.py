class BookNotFoundException(Exception):
    """
    Exception raised when a book is not found.
    """

class BookSourceNotSupported(Exception):
    """
    Exception raised when a book source is not supported.
    """

class BookTextNotFoundException(Exception):
    """
    Exception raised when a book text is not found.
    """

class ApiConnectionError(Exception):
    """
    Exception raised when a connection to an API fails.
    """