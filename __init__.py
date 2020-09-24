import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:

    class NullHandler(logging.Handler):
        def emit(self, record):
            pass


logging.getLogger(__name__).addHandler(NullHandler())


class MyslamException(Exception):
    def __init__(self, *args, **kwargs):
        # Python 3 base exception doesn't have "message" anymore, only args.
        # We restore it here for convenience.
        self.message = args[0] if len(args) >= 1 else ""
        super(MyslamException, self).__init__(*args, **kwargs)
