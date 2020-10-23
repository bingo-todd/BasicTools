import os


def parse_value_str(value_str, dtype):
    items = [[dtype(x) for x in item.strip().split()]
             for item in value_str.strip().split(';')]
    return items


def file2dict(file_path, dtype=None):
    """parse file to dictionary
    the content of file should in the following format
    key: item0; item1, ...
    ....
    Args:
        file_path:
        dtype: if not specified, the value string will not be parsed further
    """
    file_path = os.path.expanduser(file_path)
    dict_obj = {}
    with open(file_path, 'r') as dict_file:
        lines = dict_file.readlines()
        for line_i, line in enumerate(lines):
            try:
                line = line.strip()
                if len(line) < 1:
                    continue
                if line.startswith('#'):
                    continue

                key, value = line.split(':')
                key, value = key.strip(), value.strip()
                if dtype is not None:
                    value = parse_value_str(value, dtype)
                if key in dict_obj.keys():
                    raise Exception('duplicate key')
                dict_obj[key] = value
            except Exception as e:
                print(f'error in {file_path} line:{line_i}')
                raise Exception(e)
    return dict_obj


def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def dict2file(file_path, dict_obj, item_format='', is_sort=True):
    file_path = os.path.expanduser(file_path)

    keys = list(dict_obj.keys())
    if is_sort:
        keys.sort()

    with open(file_path, 'w') as dict_file:
        for key in keys:
            if isinstance(dict_obj[key], str):
                value_str = dict_obj[key]
            elif iterable(dict_obj[key]):
                value_str = '; '.join(
                    map(
                        lambda x: ('{:'+item_format+'}').format(x),
                        dict_obj[key]))
            else:
                value_str = f'{dict_obj[key]}'
            dict_file.write(f'{key}: {value_str}\n')
