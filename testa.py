from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
import pdb

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width') 
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str,
                    metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='6', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


dataset = args.dataset

if dataset == 'sysu':
    data_path = './SYSU-MM01/'
    # test_mode = [1, 2] # IR -> Visible
    test_mode = [2, 1] # Visible -> IR
    model_path = '' ### Please fill in with your own model path

elif dataset =='regdb':   
    data_path = './RegDB/'
    test_mode = [2, 1] # Visible -> IR
    # test_mode = [1, 2] # [1, 2] IR -> Visible

    model_path = '' ### Please fill in with your own model path

elif dataset =='ntu':    
    data_path = './NTU-Corridor/'
    test_mode = [2, 1] # Visible -> IR
    # test_mode = [1, 2] # [1, 2] IR -> Visible

    model_path = '' ### Please fill in with your own model path
   
if os.path.isfile(model_path):
    print('Model path is correct!')
else:
    print('MODEL PATH ERROR!')
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
pool_dim = 2048 
print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id': 
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])


class ChannelRGB(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, idx = 0):
        self.idx = idx

       
    def __call__(self, img):

        if self.idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif self.idx ==1:
            # random select G Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif self.idx ==2:
            # random select B Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            img = img

        return img
    
transform_visible_list = [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize]


# transform_visible_list = transform_visible_list +  [ChannelRGB(idx=2)]
transform_visible = transforms.Compose( transform_visible_list )

end = time.time()

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)

            flip_input = fliplr(input)
     
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, input, test_mode[0])         

            flip_input = Variable(flip_input.cuda())
            flip_feat_pool, flip_feat_fc = net(flip_input, flip_input, flip_input, test_mode[0])

            feature_pool = (feat_pool.detach() + flip_feat_pool.detach())/2
            feature_fc = (feat_fc.detach() + flip_feat_fc.detach())/2
            
            fnorm_pool = torch.norm(feature_pool, p=2, dim=1, keepdim=True)
            feature_pool = feature_pool.div(fnorm_pool.expand_as(feature_pool))
            
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            
            gall_feat_pool[ptr:ptr+batch_num,: ] = feature_pool.cpu().numpy()
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_pool, gall_feat_fc
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
     
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, input, test_mode[1])
            
            flip_input = Variable(flip_input.cuda())
            flip_feat_pool, flip_feat_fc = net(flip_input, flip_input, flip_input, test_mode[1])
            
            feature_pool = (feat_pool.detach() + flip_feat_pool.detach())/2
            feature_fc = (feat_fc.detach() + flip_feat_fc.detach())/2
            
            fnorm_pool = torch.norm(feature_pool, p=2, dim=1, keepdim=True)
            feature_pool = feature_pool.div(fnorm_pool.expand_as(feature_pool))
            
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))


            query_feat_pool[ptr:ptr+batch_num,: ] = feature_pool.cpu().numpy()
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool, query_feat_fc


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
                .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode, ir2rgb=test_mode[0])
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))  
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    for trial in range(10):
        ### We add test_mode because of our bi-directional evaluation protocol
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial, ir2rgb=test_mode[0])
        ngall = len(gall_label)

        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
        print("  ------------------------------")

        ### We add test_mode because of our bi-directional evaluation protocol
        if test_mode[0] == 1:
            trial_gallset = TestData(gall_img, gall_label, transform=transform_visible, img_size=(args.img_w, args.img_h))
        if test_mode[0] == 2:
            trial_gallset = TestData(gall_img, gall_label, transform=transform_visible, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10

    cmc_pool = all_cmc_pool / 10
    mAP_pool = all_mAP_pool / 10
    mINP_pool = all_mINP_pool / 10


    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))


elif dataset == 'regdb':

    print('==> Resuming from checkpoint..')

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(model_path))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
            .format(args.resume, checkpoint['epoch']))  
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

    for trial in range(10):
        test_trial = trial + 1

        ### We add test_mode[0] part because of the bi-directional evaluation protocol
        if test_mode[0] == 1:
            query_modal = 'thermal'
            gall_modal = 'visible'
        elif test_mode[0] == 2:
            query_modal = 'visible'
            gall_modal = 'thermal'
        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal=query_modal)
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal=gall_modal)

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

        query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
        gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader)

        if args.tvsearch:
            # pool5 feature
            distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

            # fc feature
            distmat = np.matmul(gall_feat_fc , np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat,gall_label,  query_label )
        else:
            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
    
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10

    cmc_pool = all_cmc_pool / 10
    mAP_pool = all_mAP_pool / 10
    mINP_pool = all_mINP_pool / 10

    print('All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
    

elif dataset == 'ntu':

    print('==> Resuming from checkpoint..')

    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(model_path))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
            .format(args.resume, checkpoint['epoch']))  
    else:
        print('==> no checkpoint found at {}'.format(args.resume))
    
    if test_mode[0] == 1:
        query_modal = 'thermal'
        gall_modal = 'visible'
    elif test_mode[0] == 2:
        query_modal = 'visible'
        gall_modal = 'thermal'

    # testing set
    query_img, query_label = process_test_ntu(data_path, modal=query_modal)
    gall_img, gall_label = process_test_ntu(data_path, modal=gall_modal)

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    nquery = len(query_label)
    ngall = len(gall_label)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader)

    if args.tvsearch:
        # pool5 feature
        distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

        # fc feature
        distmat = np.matmul(gall_feat_fc , np.transpose(query_feat_fc))
        cmc, mAP, mINP = eval_regdb(-distmat,gall_label,  query_label )
    else:
        # pool5 feature
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

    print(
        'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print(
        'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))

