"""Adds Animation class for simple gifs
"""

import contextlib
import os
import shutil
import tempfile

import imageio
import matplotlib.pyplot as plt


class Animation:
    """Adds simple way to create gif animations

    Usage:

    .. python::
        anim = Animation('out.gif')
        for i in range(10):
           with anim.add_frame():
               plt.plot(...)
        anim.save()
    """
    def __init__(self, filename: str, duration: float = 0.25, size=None):
        self._filename = filename
        self._duration = duration
        self._size = size
        self._frame_idx = 0
        self.__tmp_dir = tempfile.mkdtemp()

    @contextlib.contextmanager
    def add_frame(self):
        if self._size:
            fig = plt.figure(figsize=self._size)
        else:
            fig = plt.figure(figsize=(12, 12))
        plt.cla()
        yield
        plt.savefig(os.path.join(self.__tmp_dir, f"{self._frame_idx}.png"))
        plt.close(fig)
        self._frame_idx += 1

    def save(self):
        with imageio.get_writer(self._filename, mode='I', duration=self._duration) as writer:
            for idx in range(self._frame_idx):
                frame_path = os.path.join(self.__tmp_dir, f"{idx}.png")
                image = imageio.imread(frame_path)
                writer.append_data(image)
        shutil.rmtree(self.__tmp_dir)
