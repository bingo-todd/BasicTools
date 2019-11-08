import os
import datetime

def savefig(fig,fig_name=None,fig_dir='./images'):

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # use date as fig_name if fig_name is not defined
    if fig_name is None:
        fig_name = '{0.year}_{0.month}_{0.day}.png'.format(datetime.date.today())

    # check whether fig_name has suffix
    img_suffix_list = ['.bmp','.png']
    default_suffix = '.png'
    if len(fig_name)>4 and fig_name[-4:] not in img_suffix_list:
        fig_path = os.path.join(fig_dir,''.join((fig_name,'.png')))
    else:
        fig_path = os.path.join(fig_dir,fig_name)

    print('{} is saved in {}'.format(fig_name,fig_dir))
    fig.savefig(fig_path)
