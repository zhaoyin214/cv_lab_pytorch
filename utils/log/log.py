from typing import Text
import logging

class Logger(object):

    # debug < info < warning < error < critical
    _level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "crit": logging.CRITICAL
    }

    def __init__(
        self,
        filename: Text,
        level: Text="info",
        fmt: Text="%(asctime)s - %(levelname)s: %(message)s"
    ) -> None:

        # get log file
        self._logger = logging.getLogger(filename)
        self._logger.setLevel(self._level_map.get(level))
        fmt = logging.Formatter(fmt)

        # output to shell
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        # output to file
        th = logging.FileHandler(
            filename=filename, mode="w"
        )
        th.setFormatter(fmt)

        self._logger.addHandler(sh)
        self._logger.addHandler(th)

    def debug(self, value: Text):
        return self._logger.debug(value)

    def info(self, value: Text):
        return self._logger.info(value)

    def warning(self, value: Text):
        return self._logger.warning(value)

    def error(self, value: Text):
        return self._logger.error(value)

    def critical(self, value: Text):
        return self._logger.critical()
