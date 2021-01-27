import numpy as np


class Iterator:
    def __init__(self, x):
        """ x: list or ndarray
        """

        if isinstance(x, list):
            n_item = len(x)
        elif isinstance(x, np.ndarray):
            n_item = x.shape[0]
        else:
            raise Exception(f'unsupported type {type(x)}')

        self.x = x
        self._n_item = n_item  # number of elements in x
        self._item_pointer = 0   # index of element avaliable new
        self._repeat_pointer = 0  # cycle index counter

    def is_done(self, n_keep=0):
        """ whether there are n_keep item left
        """
        return self._item_pointer == self._n_item-1-n_keep

    def next(self):
        """ get next element, None will return if reach the end
        """
        if self.is_done():
            item = None
        else:
            item = self.x[self._item_pointer]
            self._item_pointer = self._item_pointer + 1
        return item

    def go_back(self, n_step):
        """ move _item_pointer backwards
        """
        if n_step >= self._item_pointer:
            raise Exception(f'n_step illegal: {n_step}')
        self._item_pointer -= n_step

    def reset(self):
        self._item_pointer = 0


if __name__ == '__main__':
    generator = Iterator([1, 2, 3, 4, 5])
    while True:
        value = generator.next()

        print(value)
        if value is None:
            generator.go_back(2)
