import os

def get_fpath(dir,suffix):
    """
    """
    file_filter = lambda fname:(len(fname)>4 and
                                     fname[-4:] == suffix and
                                     fname[0]!='.')
    fpath_all = []
    for root,dirs,files in os.walk(dir):
        for fname in files:
            if file_filter(fname):
                fpath_all.append(os.path.join(root,fname))
    return fpath_all
