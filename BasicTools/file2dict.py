def file2dict(file_path, value_type=float):
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
                items = [list(map(value_type, item.strip().split())) 
                        for item in value.strip('').split(';') ]
                if key in dict_obj.keys():
                    raise Exception('duplicate key')
                dict_obj[key] = items
            except Exception as e:
                print(f'error in {line_i}')
                raise Exception(e)
    return dict_obj
