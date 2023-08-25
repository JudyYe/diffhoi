import cv2
from PIL import Image
import imageio
import argparse
import numpy as np
from glob import glob
import os
import os.path as osp
from jutils import web_utils, image_utils


# degree_list = [0, 60, 90, 180, 270, 300]
data_dir = '/private/home/yufeiy2/scratch/result/org/'
# save_dir = '/private/home/yufeiy2/scratch/result/figs/'
save_dir = '/private/home/yufeiy2/scratch/result/figs_row/'

method2name = {
    'gt': 'GT',
    'ours': 'Ours',
    'homan_gt': 'HOMAN-GT',
    'homan_avg': 'HOMAN-AVG',
    'homan_far': 'HOMAN-FAR',

    'ihoi': 'iHOI',
    'hhor': 'HHOR',
    'obj_prior': 'Category Prior',
    'hand_prior': 'Hand Prior',
    'no_prior': 'No Prior',
    'w_mask': 'wo Mask',
    'w_normal': 'wo Normal',
    'w_depth': 'wo Depth',

    'blend': 'Hard Blend',
    'oTh': 'wo Learned Pose',

    'wild_ours': 'Ours',
    'wild_ihoi': 'iHOI',

}

def overlay(image, inp):
    alpha = image[:, :, 3:4] / 255.0
    image = image.astype(np.float32)
    inp = inp.astype(np.float32)
    image = image[..., 0:3] * alpha + inp * (1 - alpha)
    image = np.concatenate([image, np.ones_like(alpha) * 255], -1)
    return image.astype(np.uint8)

def get_fig_dir(method, index):
    # if method == 'hhor':
        # return osp.join(data_dir, method, index, 'handscanner/vis_clip')
    return osp.join(data_dir, method, index, 'vis_clip')

def get_suf_list(suf):
    if 'triplet' in suf:
        suf_list = ['overlay_hoi', '90_hoi', '90_obj' ] # '90_hoi', '90_obj',]
    elif 'narrow_triplet' in suf:
        suf_list = ['overlay_hoi', '60_hoi', '60_obj' ] # '90_hoi', '90_obj',]
    elif 'teaser' in suf:
        suf_list = ['gt' , 'overlay_hoi', '90_hoi']
    elif 'vid_t' in suf:
        suf_list = ['render_0', 'render_1'] + ['vHoi', 'vObj_t']
    elif 'narrow_two' in suf:
        suf_list = ['overlay_hoi', '60_hoi']
    elif 'vid_obj' in suf:
        suf_list = ['vHoi', 'vObj']
    else:
        suf_list = suf.split(',')
    return suf_list

def get_video_grid(args):
    """"[[m1_suf1 | m1_suf2]  ]"""
    index_list = get_data_list(args.data)
    row_dir = osp.join(save_dir, args.fig)
    suf_list = get_suf_list(args.suf)

    method_list = args.method.split(',')

    cell_list = []
    # input
    # cell_list.append(['input', ] + )
    # add header
    header_image = add_header(method_list)
    fname = osp.join(row_dir, 'header.png')
    imageio.imwrite(fname, header_image)
    cell_list.append([' ', fname])
    titled = False
    for i, index in enumerate(index_list):
        line = []        
        # add input
        suf = 'input'
        method = method_list[0]
        fig_dir = osp.join(data_dir, method_list[0], index, 'vis_video', '{}', '{}.png')        
        t_num_list = sorted(glob(osp.join(fig_dir.format(suf, '*'))))
        t_num_list = [osp.basename(t).replace('.png', '') for t in t_num_list]
        fname = osp.join(row_dir, f'{index}_input.mp4')        
        if args.dry:
            if osp.exists(fname):
                line.append(fname)
            else:
                print('not exist', fname)
        else:
            frame_list = []
            for t in t_num_list:
                # time slice
                image_list = []
                image = imageio.imread(fig_dir.format(suf, t))[..., :3]
                image_list.append(image)
                one_frame = put_one_row(image_list)
                frame_list.append(one_frame)
            image_utils.write_mp4(frame_list, fname.replace('.mp4', ''))
            line.append(fname)

        t_num_list = sorted(glob(osp.join(fig_dir.format(suf_list[0], '*'))))
        t_num_list = [int(osp.basename(t).replace('.png', '')) for t in t_num_list]
        frame_list = []
        method_str = ','.join(method_list)
        fname = osp.join(row_dir, f'{index}_{method_str}.mp4')
        # add output
        if args.dry:
            if osp.exists(fname):
                line.append(fname)
                cell_list.append(line)
            continue
        frame_list = []
        for t in t_num_list:            
            row_list = []
            for m, method in enumerate(method_list):
                fig_dir = osp.join(data_dir, method, index, 'vis_video', '{}', '{}.png')

                # time slice
                image_list = []
                for suf in suf_list:
                    max_t = len(glob(osp.join(fig_dir.format(suf, '*'))))
                    if max_t == 0:
                        continue
                    image = imageio.imread(fig_dir.format(suf, '%03d' % (t%max_t)))[..., :3]
                    image = cv2.resize(image, (200, 200))
                    image_list.append(image)
                if len(image_list) < 4:
                    continue
                image_list[0] = add_text(image_list[0], 't=%3d' % t, 'bl', 'small')
                image_list[1] = add_text(image_list[1], 't=%3d' % t, 'bl', 'small')
                image_list[2] = add_text(image_list[2], 't=%d' % (len(t_num_list)//2), 'bl', 'small')
                text = 'object (t=%3d)' % t if 'ihoi' in method else 'object'
                image_list[3] = add_text(image_list[3], text, 'bl', 'small')

                if len(method_list) > 1:
                    one_frame_method = get_one_frame(image_list)
                else:
                    one_frame_method = put_one_row(image_list)
                # if not titled:
                    # one_frame_method = add_text(one_frame_method, f'{method2name[method]}')
                    # if t == t_num_list[-1] titled = True
                row_list.append(one_frame_method)
            if len(row_list) == 0:
                continue
            # if not titled and len(method_list) > 1: 
            #     row_list = add_title(row_list, [method2name[e] for e in method_list])
            #     if t == t_num_list[-1]:
            #         titled = True
            frame = put_one_row(row_list, )
            frame_list.append(frame)

        image_utils.write_mp4(frame_list, fname.replace('.mp4', ''))
        if not osp.exists(fname):
            continue
        line.append(fname)
        cell_list.append(line)
    print(cell_list, len(cell_list[0]), len(cell_list[1]))
    if len(method_list) > 1:
        width_list = [1, len(method_list)*2]
    else:
        width_list = [1, 4]
    web_run(osp.join(row_dir, 'vis.html'), cell_list, width=100, width_list=width_list,   inplace=True, homo=len(method_list) == 1)
    return cell_list




def add_title(row_list, title_list):
    for i, row in enumerate(row_list):
        row_list[i] = add_text(row, title_list[i])
        # one_frame_method = add_text(one_frame_method, f'{method2name[method]}')
    return row_list

def add_header(method_list):
    # create image
    if len(method_list) > 1:
        width = 200 * 2
    else:
        width = 200 * 4
    banner_list = []
    for method in method_list:
        banner = np.ones((30, width, 3), dtype=np.uint8) * 255
        banner = add_text(banner, method2name[method])
        banner_list.append(banner)
    banner = put_one_row(banner_list)
    return banner


def add_text(frame, text, place='tm', scale='title'):
    # add text in the top middle
    font = cv2.FONT_HERSHEY_SIMPLEX
    if scale == 'title':
        fontScale = 0.75 
        thickness = 1
    elif scale == 'small':
        fontScale = 0.5
        thickness = 1
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    frame = frame.copy()
    if place == 'tm':
        textX = (frame.shape[1] - textsize[0]) // 2
        textY = 20
    elif place == 'bl':
        textX = 10
        textY = frame.shape[0] - 10
    elif place == 'bm':
        textX = (frame.shape[1] - textsize[0]) // 2
        textY = frame.shape[0] - 10

    cv2.putText(frame, text, (textX, textY), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
    return frame
    
def get_one_frame(image_list):
    frame = put_one_col([
        put_one_row(image_list[0:2]),
        put_one_row(image_list[2:4]),
    ])
    # pad black line with m at boundary
    frame = np.concatenate([frame, np.zeros([frame.shape[0], 1, 3], np.uint8)], 1)
    # resize height to 300 keep aspect ratio
    frame = cv2.resize(frame, (int(frame.shape[1] * 300 / frame.shape[0]), 300))
    
    return frame


def merge_fig(args):
    index_list = get_data_list(args.data)
    method_list = args.method.split(',')
    # cp input to index_inp.png
    row_dir = osp.join(save_dir, args.fig)
    
    inp_dict = {}
    for index in index_list:
        method = method_list[0]
        fig_dir = get_fig_dir(method, index)
        img_list = sorted(glob(osp.join(fig_dir, f'*_gt.png')))
        t = min(args.t, len(img_list))
        if len(img_list) == 0:
            print('no image?', osp.join(fig_dir, f'*_gt.png'))
            continue
        row = imageio.imread(img_list[t])
        inp_dict[index] = row

        fname = osp.join(row_dir, f'{index}_input.png')
        os.makedirs(osp.dirname(fname), exist_ok=True)
        imageio.imwrite(fname, row)

    for index in index_list:
        row_list = []
        for method in method_list:
            image_list = []
            fig_dir = get_fig_dir(method, index)
            suf_list = get_suf_list(args.suf)
            for suf in suf_list:
                suf = '_' + suf
                # for degree in args.degree.split(','):
                img_list = sorted(glob(osp.join(fig_dir, f'*{suf}.png')))
                if len(img_list) == 0:
                    print(osp.join(fig_dir, f'*{suf}.png'), 'no image')
                    continue
                t = min(args.t, len(img_list))
                image = imageio.imread(img_list[t])
                # if args.suf == 'triplet':
                #     if 'overlay_hoi' not in suf:
                #         # print('resize overlay')
                #         # image = resize_crop(image, 1)
                #         image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                
                    # image = overlay(image, inp_dict[index])
                image_list.append(image)
            if len(image_list) == 0:
                continue
            row = get_row(image_list, args)
            row_list.append(row)
        if len(row_list) == 0:
            continue

        row = merge_method(row_list)
        name = ','.join(method_list)
        fname = osp.join(row_dir, f'{index}_{name}.png')
        os.makedirs(osp.dirname(fname), exist_ok=True)
        imageio.imwrite(fname, row)

    return


def merge_teaser(args):
    index_list = get_data_list(args.data)
    method_list = args.method.split(',')
    # cp input to index_inp.png
    row_dir = osp.join(save_dir, args.fig)
    
    for index in index_list:
        for method in method_list:
            fig_dir = get_fig_dir(method, index)
            suf_list = get_suf_list(args.suf)
            if ',' not in args.ts:
                t_list = range(int(args.ts))
            else:
                t_list = args.ts.split(',')
            for t in t_list:
                row_list = []
                image_list = []
                for suf in suf_list:
                    suf = '_' + suf
                    img_list = sorted(glob(osp.join(fig_dir, f'*{suf}.png')))
                    if len(img_list) == 0:
                        print(osp.join(fig_dir, f'*{suf}.png'), 'no image')
                        continue
                    # t = min(args.t, len(img_list))
                    print(img_list[t])
                    image = imageio.imread(img_list[t])
                    # image = overlay(image, inp_dict[index])
                    image_list.append(image)
                if len(image_list) == 0:
                    continue
                row = get_row(image_list, args)
                row_list.append(row)
                if len(row_list) == 0:
                    continue
                row = merge_method(row_list)
                name = '%02d' % t
                fname = osp.join(row_dir, f'{index}_{name}.png')
                os.makedirs(osp.dirname(fname), exist_ok=True)
                imageio.imwrite(fname, row)
    return

def merge_method(row_list):
    # pad black line on left and right 
    row_list = [np.pad(row, ((0, 0), (10, 10), (0, 0)), 'constant') for row in row_list]
    return put_one_row(row_list)


def get_row(image_list ,args):
    if args.suf == 'compact_triplet':
        row = put_triplet(image_list, 0.1)
    elif 'triplet' in args.suf:
        row = put_triplet(image_list)
    elif args.suf == 'teaser':
        row = put_one_col_jpg(image_list)
    else:
        if len(image_list) == 2:
            row = put_one_col(image_list)
        elif len(image_list) == 4:
            row = put_to_2x2(image_list)
        else:
            row = put_one_row(image_list)        
    return row


def put_triplet(image_list, m=0):
    img1, img2, img3  = image_list
    img2 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)
    img3 = cv2.resize(img3, (0, 0), fx=0.5, fy=0.5)
    right = put_one_col([img2, img3])
    if m == 0:
        left = img1
    else:
        # crop img 1 width with m 
        W, H = img1.shape[1], img1.shape[0]
        left = img1[:, int(W*m):int(W*(1-m))]
        print('cut')

    row = put_one_row([left, right])
    return row

def resize_crop(image, perc=1.4):
    image = Image.fromarray(image)
    W, H = image.size
    new_W, new_H = int(W*perc), int(H*perc)
    image = image.resize((int(W*perc), int(H*perc)))
    # if perc > 1:
    m_W = (new_W - W) // 2
    m_H = (new_H - H) // 2
    center_box = [m_W, m_H, m_W+W, m_H+H]
    image = image.crop(center_box, )
    return np.array(image)


def put_one_col_jpg(image_list):
    for i, e in enumerate(image_list):
        if e.shape[2] == 3:
            e = np.concatenate([e, np.zeros((e.shape[0], e.shape[1], 1)) +255],axis=2).astype(np.uint8)
            image_list[i] = e
    # image_list = [e[..., :3] for e in image_list]
    image_list[-1] = resize_crop(image_list[-1], 0.8)
    a = np.concatenate(image_list, axis=0)

    return a


def put_one_col(image_list):
    # 2 element, make 2x1 grid
    a = np.concatenate(image_list, axis=0)
    return a

def put_one_row(image_list):
    # 2 element, make 1x2 grid
    a = np.concatenate(image_list, axis=1)
    return a

def put_to_2x2(image_list):
    # 4 element,  make 2x2 grid
    N = len(image_list)
    a = np.concatenate(image_list[0:N//2], axis=0)
    b = np.concatenate(image_list[N//2:], axis=0)
    c = np.concatenate([a,b], axis=1)
    return c


def teaser_web(args):
    index_list = get_data_list(args.data)
    row_dir = osp.join(save_dir, args.fig)
    web_dir = osp.join(row_dir, 'web')
    # empty_list of 100x10
    # cell_list = [['' for _ in range(10)] for _ in range(100)]

    cell_list = []
    cnt = 0
    for i, index in enumerate(index_list):
        line = []
        query = osp.join(row_dir, f'{index}_*.png')
        line = sorted(glob(query))
        if len(line) == 0:
            continue
        cnt += 1
        cell_list.append(line)
        # for j, fname in enumerate(line):
            # cell_list[cnt][j] = fname
        # print(len(line), query)
        
        # cell_list.append(line)
        print(line)
    print(cell_list)
    max_len = max([len(e) for e in cell_list])
    cell_list = [e + ['' for _ in range(max_len - len(e))] for e in cell_list]
    # make cell_list to regular 2D list 

    web_utils.run(web_dir, cell_list, width=200)
    return cell_list
    

def web_run(html_root, cell_list, width=200, hide_text=False, height=None, width_list=None, inplace=False, homo=False):
    """
    cell_list: 2D array, each element could be: filepath of vid/image, str
    """
    if not html_root.endswith('.html'):
        html_file = os.path.join(html_root, 'index.html')
    else:
        html_file = html_root
        html_root = os.path.dirname(html_root)
    os.makedirs(html_root, exist_ok=True)

    ncol = len(cell_list[0])
    # title
    TableCls = web_utils.create_table('TableCls')
    for c in range(ncol):
        label = 'Input' if c == 0 else 'Output'
        TableCls = TableCls.add_column('%d' % c, web_utils.Col(label))

    items = []
    for r, row in enumerate(cell_list):
        line = {}
        for c in range(ncol):
            pref = 'r%02dc%02d' % (r, c)
            # h = height//2 if c == 0 and not homo else height
            w = width_list[c] * width
            out = web_utils.html_add_col_text(row[c], html_root, w, pref, hide_text, height=height, inplace=inplace)
            out = out.split('<br/>')[0]
            line['%d' % c] = out
        items.append(line)
    table = TableCls(items)
    html_str = table.__html__()
    with open(os.path.join(html_file), 'w') as fp:
        # add header
        fp.write('<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>\n')
        fp.write(html_str)
        print('write to %s' % html_file)


def web_merge(args):
    index_list = get_data_list(args.data)
    row_dir = osp.join(save_dir, args.fig)
    web_dir = osp.join(row_dir, 'web')
    # empty_list of 100x10
    # cell_list = [['' for _ in range(10)] for _ in range(100)]
    cell_list = []
    cnt = 0
    for i, index in enumerate(index_list):
        line = []
        query = osp.join(row_dir, f'{index}_*.png')
        line = sorted(glob(query))
        if len(line) == 0:
            continue
        cnt += 1
        cell_list.append(line)
        # for j, fname in enumerate(line):
            # cell_list[cnt][j] = fname
        # print(len(line), query)
        
        # cell_list.append(line)
        print(line)
    print(cell_list)
    web_utils.run(web_dir, cell_list, height=200)
    return cell_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='hoi4d')
    parser.add_argument('--degree', type=str, default='overlay,90')
    parser.add_argument('--suf', type=str, default='hoi,obj')

    parser.add_argument('--t', type=int, default=1)
    parser.add_argument('--ts', type=str, default='10')
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--fig', type=str, default='default')
    parser.add_argument('--dry', action='store_true', default=False)
    args = parser.parse_args()
    return args




def get_data_list(data):
    if data == 'hoi4d':
        data = 'hoi4d'
        cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
        ind_list = [1,2]
        index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]
    if data == 'hoi4d_half':
        data = 'hoi4d'
        cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
        ind_list = [1]
        index_list = [f"{cat}_{ind}" for ind in ind_list for cat in cat_list ]
    elif data == 'wild': 
        index_list = get_data_list('3rd') + get_data_list('1st') + get_data_list('visor')
    elif data == '3rd':
        data = '3rd_nocrop'
        cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
        ind_list = list(range(10))
        index_list = [f"{cat.lower()}{ind}" for ind in ind_list for cat in cat_list ]
    elif data == 'visor':
        data = 'VISOR'
        cat_list = "Kettle,Bowl,Knife,ToyCar".split(',')
        # ind_list = 
        index_list ='Kettle_101,Kettle_102,Bottle_102'.split(',')
    elif data == 'hhor':
        cat_list = 'AirPods,Doraemon'.split(',')
        ind_list = list(range(3))
        index_list = [f"{cat}_{ind}" for cat in cat_list for ind in ind_list  ]       

    elif data == '1st':
        data = '1st_nocrop'
        cat_list = "Mug,Bottle,Kettle,Bowl,Knife,ToyCar".split(',')
        ind_list = list(range(10))
        index_list = [f"{cat.lower()}_{ind}" for cat in cat_list for ind in ind_list  ]
    return index_list

def get_image_grid(args):
    merge_fig(args)
    cell_list = web_merge(args)
    return cell_list


if __name__ == '__main__':
    args = parse_args()

    # cp_fig()

    merge_fig(args)
    # web_merge(args)

    # merge_teaser(args)
    # teaser_web(args)


    # get_video_grid(args)
    # make_video_web(args)