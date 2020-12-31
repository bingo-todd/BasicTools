import numpy as np


class Iterator:
    def __init__(self, x, _n_repeat=1):
        """ x: list or ndarray
        """

        if isinstance(x, list):
            _n_item = len(x)
        elif isinstance(x, np.ndarray):
            _n_item = x.shape[0]

        self.x = x
        self._n_repeat = _n_repeat
        self._n_item = _n_item
        self._item_counter = 0
        self._repeat_counter = 0

    def is_done(self):
        return (self._repeat_counter == self._n_repeat-1
                and self._item_counter == self._n_item)

    def next(self):
        """
        """
        if (self._repeat_counter == self._n_repeat-1
                and self._item_counter == self._n_item):
            item = None
        else:
            item = self.x[self._item_counter]
            self._item_counter = self._item_counter + 1
            if (self._item_counter == self._n_item
                    and self._repeat_counter < self._n_repeat-1):
                self._item_counter = 0
                self._repeat_counter = self._repeat_counter+1
        return item

    def reset(self):
        self._item_counter = 0
        self._repeat_counter = 0
