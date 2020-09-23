def parse_value_str(value_str, dtype):
    items = [[dtype(x) for x in item.strip().split()] 
            for item in value.strip().split(';') ]


def file2dict(file_path, dtype=None):
    """parse file to dictionary
    the content of file should in the following format
    key: item0; item1, ...
    ....
    Args: 
        file_path:
        dtype: if not specified, the value string will not be parsed further
    """
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
                if dtype is not None:
                    value = parse_value_str(value_str, dtype)
                if key in dict_obj.keys():
                    raise Exception('duplicate key')
                dict_obj[key] = value
            except Exception as e:
                print(f'error in {line_i}')
                raise Exception(e)
    return dict_obj


def dict2file(dict_obj, file_path, is_sort=True):
    keys = list(dict_obj.keys())
    if is_sort:
        keys.sort()
    
    with open(file_path, 'w') as dict_file:
        for key in keys:
            dict_file.write(f'{key}: {dict_obj[key]}\n')
    

