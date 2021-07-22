#  Copyright (c) 2021 by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import os
from os import makedirs, scandir


class OSHelper:

    @staticmethod
    def mkdirs(paths):
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                makedirs(path, exist_ok=True)
        else:
            makedirs(paths, exist_ok=True)

    @staticmethod
    def scan_dirs_for_folder(paths) -> [os.DirEntry, ...]:
        entry_list = OSHelper.scan_dirs(paths)
        return [entry for entry in entry_list if entry.is_dir()]

    @staticmethod
    def scan_dirs_for_file(paths, allow_list: [str, ...]):
        entry_list = OSHelper.scan_dirs(paths)

        def _is_allow(en: os.DirEntry):
            return any(en.name.endswith(extension) for extension in allow_list)

        return [entry for entry in entry_list if entry.is_file() and _is_allow(entry)]

    @staticmethod
    def scan_dirs(paths: list or tuple or str):
        if isinstance(paths, str):
            paths = [paths]
        return OSHelper._scan_dirs(paths)

    @staticmethod
    def _scan_dirs(paths: list or tuple) -> [os.DirEntry, ...]:
        re = []
        for path in paths:
            with scandir(path) as it:
                entry: os.DirEntry
                re.extend([entry for entry in it])
        return re

