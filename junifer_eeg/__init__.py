"""JUnifer EEG - EEG extension for the JUelich NeuroImaging FEature extractoR."""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Giovanni Marraffini, Fede Raimondo"
__email__ = "g.marraffini@neurometers.ai"


# Import main modules
from . import markers  # noqa: F401
from . import datagrabber  # noqa: F401
from . import preprocessors  # noqa: F401 