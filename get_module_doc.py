import importlib
import inspect
import sys

def load_module(module_name):
    try:
        module_obj = importlib.import_module(module_name)
    except Exception as e:
        print(e)
        return None
    return module_obj

public_attr_filter = lambda x: len(x)>1 and x[0] != '_'

def get_doc(obj):
    doc = dict()
    attr_name_list = filter(public_attr_filter,dir(obj))
    for attr_name in attr_name_list:
        attr = getattr(obj,attr_name)
        if inspect.isfunction(attr):
            doc[attr_name] = attr.__doc__

        if inspect.ismodule(attr):
            doc[attr_name] = [attr.__doc__,get_doc]
    return doc

def print_key_value(key,value,level=0):
    if level > 0:
        print('{} {}: {}'.format('\t'*level,key,value))
    else:
        print('{}: {}'.format(key,value))

def print_dict(dic_obj,level=0):
    for key in dic_obj.keys():
        if isinstance(dic_obj[key],list):
            print_key_value(key,dic_obj[key][0],level)
            print_dict(key,dic_obj[key][1],level+1)
        else:
            print_key_value(key,dic_obj[key],level)

if __name__ == "__main__":
    module_name = sys.argv[1]
    module_obj = load_module(module_name)
    doc = get_doc(module_obj)
    print_dict(doc)
