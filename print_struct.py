"""
Print the structure of a module
"""

import inspect
import sys
import importlib
import os
import clipboard
import re

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(tree_str='') #
def add_branch(obj,is_doc=True,level=1):
    if isinstance(obj,str):
        name = obj
        doc = None
        params = None
    else:
        name = obj.__name__
        doc = obj.__doc__
        if inspect.isfunction(obj):
            arg_info = inspect.getfullargspec(obj)
            params = ['{}'.format(arg_name) for arg_name in arg_info.args]
            if arg_info.defaults is not None:
                for i in range(-len(arg_info.defaults),0):
                    default_value = arg_info.defaults[i]
                    if inspect.isfunction(default_value):
                        default_value = '.'.join([default_value.__module__,default_value.__name__])
                    params[i]='='.join([params[i],str(default_value)])
            params = ', '.join(params)
        else:
            params = None

    if params is not None:
        add_branch.tree_str = ''.join([add_branch.tree_str,
                                       '|{}{}({})\n'.format('-'*level,name,params)])
    else:
        add_branch.tree_str = ''.join([add_branch.tree_str,
                                   '|{}{}\n'.format('-'*level,name)])

    if doc is not None and is_doc:
        for line in doc.splitlines(keepends=False):
            add_branch.tree_str = ''.join([add_branch.tree_str,
                                           '|\t{}\n'.format(line)])
    else:
        add_branch.tree_str = ''.join([add_branch.tree_str,'|\n'])

    # add_branch.tree_str = ''.join([add_branch.tree_str,'|\n'])

def show_struct(file_path,is_doc=True):
    # first load file
    py_dir = os.path.dirname(file_path)
    sys.path.append(py_dir)
    module_name = os.path.basename(file_path)[:-3]
    module_obj = importlib.import_module(module_name)

    # get all attris in this file, may contain modules loaded in file
    attr_name_list = dir(module_obj)
    module_attr_list = [getattr(module_obj,attr_name) for attr_name in attr_name_list
                            if (not attr_name.startswith('__') and
                                inspect.isclass(getattr(module_obj,attr_name)))]

    # find all classes defined in this file
    class_obj_list = [attr for attr in module_attr_list if inspect.isclass(attr)]
    for class_obj in class_obj_list:
        attr_name_list = dir(class_obj)
        func_obj_list = [getattr(class_obj,attr_name) for attr_name in attr_name_list
                            if (not attr_name.startswith('__') and
                                inspect.isfunction(getattr(class_obj,attr_name)))]

    add_branch(class_obj,is_doc,0)
    add_branch('functions',is_doc,1)
    if is_doc:
        for func_obj in func_obj_list:
            add_branch(func_obj,is_doc,2)
    else:
        for func_obj in func_obj_list:
            add_branch(func_obj,is_doc,2)

if __name__ == '__main__':

    txt_path = None
    is_doc = True
    is_tight = False

    args = sys.argv
    if 'no_doc' in args:
        is_doc = False

    if 'tight' in args:
        is_tight = True

    args = sys.argv
    if len(args) < 2:
        raise Exception('py file path needed')
    if len(args) >= 2:
        py_path = args[1]
    # if len(args) >= 3:
    #     txt_path = args[2]


    if not os.path.exists(py_path):
        raise Exception('fail to find {}'.format(py_path))

    show_struct(py_path,is_doc=is_doc)

    if is_tight:
        tree_str = re.sub('\|\s*\n','',add_branch.tree_str)
    else:
        tree_str = add_branch.tree_str

    if txt_path is not None:
        with open(txt_path,'w') as txt_file:
            txt_file.writelines(tree_str)
    else:
        print(tree_str)
        clipboard.copy(tree_str)

    # clear tree_str
    add_branch.tree_str = ''
