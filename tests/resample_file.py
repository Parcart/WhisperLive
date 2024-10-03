import os

from whisper_live import utils

# file_path = os.path.abspath('test_woman')
utils.resample("test_woman", sr=41000)