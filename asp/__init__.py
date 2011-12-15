# From http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
# Author: James Antill (http://stackoverflow.com/users/10314/james-antill)
__version__ = '0.1.2.3'
__version_info__ = tuple([ int(num) for num in __version__.split('.')])


class SpecializationError(Exception):
    """
    Exception that caused specialization not to occur.

    Attributes:
      msg -- the message/explanation to the user
      phase -- which phase of specialization caused the error
    """

    def __init__(self, msg, phase="Unknown phase"):
        self.msg = msg
        
