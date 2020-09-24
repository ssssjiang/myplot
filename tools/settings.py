from __future__ import print_function

import json
import logging
import os

from myplot_tools import MyException

logger = logging.getLogger(__name__)

PACKAGE_BASE_PATH = os.path.abspath(__file__ + "/../../")
DEFAULT_PATH = os.path.join(PACKAGE_BASE_PATH, "settings.json")


class SettingsException(MyException):
    pass


class SettingsContainer(dict):
    def __init__(self, data, lock=True):
        super(SettingsContainer, self).__init__()
        for k, v in data.items():
            setattr(self, k, v)
        setattr(self, "__locked__", lock)

    @classmethod
    def from_json_file(cls, settings_path):
        with open(settings_path) as settings_file:
            data = json.load(settings_file)
        return SettingsContainer(data)

    def locked(self):
        if "__locked__" in self:
            return self["__locked__"]

    def __getattr__(self, attr):
        # allow dot access
        if attr not in self:
            raise SettingsException("unknown settings parameter: " + str(attr))
        return self[attr]

    def __setattr__(self, attr, value):
        # allow dot access
        if self.locked() and attr not in self:
            raise SettingsException(
                "write-access locked, can't add new parameter {}".format(attr))
        else:
            self[attr] = value


SETTINGS = SettingsContainer.from_json_file(DEFAULT_PATH)
