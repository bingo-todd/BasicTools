import numpy as np


class Iterator:
    def __init__(self, x, n_repeat=1):
        """ x: list or ndarray
        """

        if isinstance(x, list):
            n_item = len(x)
        elif isinstance(x, np.ndarray):
            n_item = x.shape[0]
        else:
            raise Exception(f'unsupported type {type(x)}')

        self.x = x
        self._n_repeat = n_repeat
        self._n_item = n_item
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


if __name__ == '__main__':
    generator = Iterator([1, 2, 3, 4, 5])
    while True:
        value = generator.next()
        print(value)
        if value is None:
            break
