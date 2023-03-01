import os.path as osp
from glob import glob
import numpy as np
# import wandb.apis.reports as wr
from jutils import web_utils
np.random.seed(123)


# 
def main():
    report = wr.Report(
        project='vhoi_datset',
        title='VHOI datasets comparison',
        description="That was easy!",
        
    )

    add_hhor('/home/yufeiy2/scratch/result/HOI4D/')
    report.save()                     # Save


def add_hhor(data_dir='/home/yufeiy2/scratch/result/HOI4D/'):
    mapping = [
        '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
        'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
        'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
    ]
    rigid = [
        'Bowl', 'Bottle', 'Mug', 'ToyCar', 'Knife', 'Kettle',
    ]

    name2id = {}
    for i, name in enumerate(mapping):
        name2id[name] = i
    # Bottle: C5
    #     
    file_list = sorted(glob(osp.join(data_dir, '*', 'image.gif')))
    print(file_list, osp.join(data_dir, '*', 'image.gif'))
    # np.random.shuffle(file_list)
    cell_list = []
    for image_file in file_list:
        line = []
        C = int(image_file.split('/')[-2].split('_')[2][1:])
        name = mapping[C]
        if name not in rigid:
            print(name)
            continue
        line.append(image_file)
        line.append(image_file.replace('image.gif', 'overlay.gif'))

        cell_list.append(line)
    web_utils.run(osp.join(data_dir, 'vis'), cell_list)

    
    return 

def add_ho3d_det(data_dir='/home/yufeiy2/scratch/result/ho3d_det/'):
    # Bottle: C5
    #     
    file_list = sorted(glob(osp.join(data_dir, '*', 'image.gif')))
    print(file_list, osp.join(data_dir, '*', 'image.gif'))
    # np.random.shuffle(file_list)
    cell_list = []
    for image_file in file_list:
        line = []
        line.append(image_file.split('/')[-2])
        line.append(image_file)
        line.append(image_file.replace('image.gif', 'overlay.gif'))
        line.append(image_file.replace('image.gif', 'overlay_pred.gif'))
        # line.append(image_file.replace('image.gif', 'overlay_smooth_10.0.gif'))
        line.append(image_file.replace('image.gif', 'overlay_smooth_100.gif'))
        # line.append(image_file.replace('image.gif', 'overlay_smooth_200.0.gif'))

        cell_list.append(line)
    web_utils.run(osp.join(data_dir, 'vis'), cell_list)

        
if __name__ == '__main__':
    # add_hhor()
    add_ho3d_det('/home/yufeiy2/scratch/result/HOI4D/')
    # add_hoi4d()