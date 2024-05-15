# python train_ext.py --dataset prcc --lr 0.1 --method adp --augc 1 --rande 0.5 --alpha 1 --square 1 --gamma 1 --gpu 1
# python testa.py  --dataset sysu --mode all --gpu 1

from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from data_loader import PRCCData, LTCCData, TestData, PRCCData
from data_manager import *
from eval_metrics import eval_regdb, eval_ltcc
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT, KLDivLoss, TripletLoss_ADP
from tensorboardX import SummaryWriter
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import pdb

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='prcc', help='dataset name: prcc or ltcc]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
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
parser.add_argument('--img_w', default=192, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw, adp')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='6', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--augc', default=0 , type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default= 0 , type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default= 0 , type=float,
                    metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1 , type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1 , type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default= 1 , type=int,
                    metavar='square', help='gamma for the hard mining')
                   
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
### We add this line for saving different logs and models for different methods
our_method = 'ce2'

if dataset == 'prcc':  
    data_path = './PRCC'
    log_path = args.log_path + 'prcc_log/'

elif dataset == 'ltcc':  
    data_path = './LTCC'
    log_path = args.log_path + 'ltcc_log/'

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'adp':
    suffix = suffix + '_{}_joint_co_nog_ch_nog_sq{}'.format(args.method, args.square)
else:
    suffix = suffix + '_{}'.format(args.method)
if args.augc==1:
    suffix = suffix + '_aug_G'  
if args.rande>0:
    suffix = suffix + '_erase_{}'.format( args.rande)
    
suffix = suffix + '_p{}_n{}_lr_{}_seed_{}'.format( args.num_pos, args.batch_size, args.lr, args.seed) 

suffix = suffix + '_' + our_method 

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]
    
transform_test = transforms.Compose( [
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.rande>0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability = args.rande)]

if args.augc ==1:
    transform_train_list = transform_train_list + [ChannelAdapGray(probability =0.5)]
    
transform_train = transforms.Compose( transform_train_list )

end = time.time()
if dataset == 'prcc':
    # training set
    trainset = PRCCData(data_dir=data_path, transform=transform_train, height=args.img_h, width=args.img_w)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_rgb_label, trainset.train_sketch_label)

    query_img, query_label, gall_img, gall_label = process_test_prcc(data_path)

elif dataset == 'ltcc':
    # training set
    trainset = LTCCData(data_dir=data_path, transform=transform_train, height=args.img_h, width=args.img_w)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_label, trainset.train_label)

    query_img, query_label, query_cloth,\
        gall_img, gall_label, gall_cloth = process_test_ltcc(data_path)

gallset  = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

if dataset == 'prcc':
    n_class = len(np.unique(trainset.train_rgb_label))
if dataset == 'ltcc':
    n_class = len(np.unique(trainset.train_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
if dataset == 'prcc':
    print('  rgb  | {:5d} | {:8d}'.format(n_class, len(trainset.train_rgb_label)))
    print('  sketch  | {:5d} | {:8d}'.format(n_class, len(trainset.train_sketch_label)))
if dataset == 'ltcc':
    print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch)
net.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
elif args.method == 'adp':
    criterion_tri = TripletLoss_ADP(alpha = args.alpha, gamma = args.gamma, square = args.square) # default
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
criterion_kl = KLDivLoss()
criterion_id.to(device)
criterion_tri.to(device)
criterion_kl.to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr

def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    kl_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(trainloader):
        
        labels = torch.cat((label1, label1), 0)
        labels = Variable(labels.cuda())
        
        input10 = Variable(input10)
        input11 = Variable(input11)
        
        input1 = torch.cat((input10, input11),0)
        input10 = input10.cuda()
        input11 = input11.cuda()

        data_time.update(time.time() - end)

        feat, out0, = net(input10, input11)

        loss_id = criterion_id(out0, labels)
        
        # loss kl
        n = out0.shape[0]//3
        out1 = out0.narrow(0,0,n)
        out2 = out0.narrow(0,2*n,n)
        loss_kl = criterion_kl(out1, Variable(out2))                                        
                                                    
        loss_tri, batch_acc = criterion_tri(feat, labels)
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_id + loss_tri + args.kl * loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        ###
        train_loss.update(loss.item(), input1.size(0))
        id_loss.update(loss_id.item(), input1.size(0))
        tri_loss.update(loss_tri.item(), input1.size(0))
        kl_loss.update(loss_kl.item(), input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'KLoss: {kl_loss.val:.4f} ({kl_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,kl_loss=kl_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat, feat_att = net(input, input, 1)

            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)

            input = Variable(input.cuda())
            feat, feat_att = net(input, input, 1)
            
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'prcc':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    if dataset == 'ltcc':
        cmc, mAP, mINP = eval_ltcc(-distmat, query_label, gall_label, query_cloth, gall_cloth)
        cmc_att, mAP_att, mINP_att = eval_ltcc(-distmat_att, query_label, gall_label, query_cloth, gall_cloth)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 90 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    if dataset == 'prcc':
        sampler = IdentitySampler(trainset.train_rgb_label, \
                              trainset.train_sketch_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)
    if dataset == 'ltcc':
        sampler = IdentitySampler(trainset.train_label, \
                              trainset.train_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path  + suffix + '_best.t')

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))