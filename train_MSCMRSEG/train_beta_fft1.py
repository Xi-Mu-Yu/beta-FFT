import argparse
import logging
import os
import random
import sys
import torch.nn as nn
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from utils.displace import process_image_batches
from skimage.measure import label
import shutil

from dataloaders.dataset import (
    BaseDataSets,
    TwoStreamBatchSampler,

)
from dataloaders.dataset_msc import WeakStrongAugment
from networks.net_factory import net_factory
from networks.unet_de import UNet_LDMV2
from utils import losses, metrics, ramps, util
from val_2D import test_single_volume_refinev2 as test_single_volume
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../data/MS_CMRSEG/mscmrseg19_split1", help="Name of Experiment")
parser.add_argument("--exp", type=str, default="msc01/diffrect", help="experiment_name")
parser.add_argument("--model", type=str, default="unet", help="model_name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument("--num_classes", type=int, default=4, help="output channel of network")
parser.add_argument("--img_channels", type=int, default=1, help="images channels, 1 if ACDC, 3 if GLAS")
parser.add_argument("--load", default=False, action="store_true", help="restore previous checkpoint")
parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="confidence threshold for using pseudo-labels",
)

parser.add_argument("--labeled_bs", type=int, default=4, help="labeled_batch_size per gpu")
parser.add_argument("--labeled_num", type=int, default=7, help="labeled data")
parser.add_argument("--refine_start", type=int, default=1000, help="start iter for rectification")
# costs
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str, default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float, default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency_rampup")
parser.add_argument('--image_size', type=list,  default=[256, 256], help='patch size of network input')
# rf
parser.add_argument("--base_chn_rf", type=int, default=64, help="rect model base channel")
parser.add_argument("--ldm_beta_sch", type=str, default='cosine', help="diffusion schedule beta")
parser.add_argument("--ts", type=int, default=10, help="ts")
parser.add_argument("--ts_sample", type=int, default=2, help="ts_sample")
parser.add_argument("--ref_consistency_weight", type=float, default=-1, help="consistency_rampup")
parser.add_argument("--no_color", default=False, action="store_true", help="no color image")
parser.add_argument("--no_blur", default=False, action="store_true", help="no blur image")
parser.add_argument("--rot", type=int, default=359, help="rotation angle")

args = parser.parse_args()
dice_loss = losses.DiceLoss(n_classes=4)
ce_loss = CrossEntropyLoss()
def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "1": 32,
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif 'Task05' in dataset:
        assert args.num_classes == 3, "Task05 only has 3 classes"
        if 'split1' in dataset:
            ref_dict = {'2': 30}
        elif 'split2' in dataset:
            ref_dict = {'2': 40}
    elif 'mscmrseg19' in dataset:
        if 'split1' in dataset:
            ref_dict = {'7': 110}
        elif 'split2' in dataset:
            ref_dict = {'7': 103}
    else:
        raise NotImplementedError
    return ref_dict[str(patiens_num)]

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)  # loss = loss_ce
    return loss_dice, loss_ce

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = 10000
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)
    batch_size = args.batch_size

    def create_model(ema=False, in_chns=1):
        model = net_factory(net_type=args.model, in_chns=in_chns, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model(in_chns=args.img_channels)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    
    labeled_idxs = list(range(0, labeled_slice))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,transform=transforms.Compose([WeakStrongAugment(args.image_size)],
    ))
        
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)


    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))  # （136，137.。。1311）
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    writer = SummaryWriter(snapshot_path + '/log')
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)
          
            # -- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)

           
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

                if iter_num % 200 == 0:
                    model.eval()
                    metric_list = 0.0
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume(
                            sampled_batch["image"],
                            sampled_batch["label"],
                            model,
                            classes=num_classes,
                        )
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)

                    performance = np.mean(metric_list, axis=0)[0]
                    mean_hd95 = np.mean(metric_list, axis=0)[1]
                    mean_jaccard = np.mean(metric_list, axis=0)[2]

                    if performance > best_performance:
                        best_performance = performance
                        logging.info("BEST PERFORMANCE UPDATED AT ITERATION %d: Dice: %f, HD95: %f" % (
                            iter_num, performance, mean_hd95))
                        save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
                        # util.save_checkpoint(epoch_num, model, optimizer, loss, save_mode_path)
                        util.save_checkpoint(iter_num, model, optimizer, loss, save_best)

                    for class_i in range(num_classes - 1):
                        logging.info(
                            "iteration %d: model_val_%d_dice : %f model_val_%d_hd95 : %f model_val_%d_jaccard : %f"
                            % (iter_num, class_i + 1, metric_list[class_i, 0], class_i + 1, metric_list[class_i, 1],
                               class_i + 1, metric_list[class_i, 2])
                        )
                    logging.info(
                        "iteration %d : model_mean_dice : %f model_mean_hd95 : %f model_mean_jaccard : %f"
                        % (iter_num, performance, mean_hd95, mean_jaccard)
                    )

                    ###############
                    # TEST, only use the result of the best val model
                    # test_func(num_classes, db_test, model, refine_model, iter_num, testloader)
                    ########################################################

                    model.train()

                if iter_num >= max_iterations:
                    break
            if iter_num >= max_iterations:
                iterator.close()
                break
def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

# following ACDC_train
def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]  # == c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)

        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    # if nms == 1:
    #     probs = get_ACDC_2DLargestCC(probs)
    return probs


def train(args,pre_snapshot_path, snapshot_path):
    # args_dict = vars(args)
    # for key, val in args_dict.items():
    #     logging.info("{}: {}".format(str(key), str(val)))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    pre_trained_model = os.path.join(pre_snapshot_path, '{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs / 2), int((args.batch_size - args.labeled_bs) / 2)

    def create_model(ema=False, in_chns=1):
        model = net_factory(net_type=args.model, in_chns=in_chns, class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model_1 = create_model()
    model_2 = create_model()
    ema_model = create_model( ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([WeakStrongAugment(args.image_size)],)
    )
    
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    
    logging.info("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)


    optimizer1 = optim.SGD(model_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    start_epoch = 0
    # load_net(ema_model, pre_trained_model)
    # load_net_opt(model_1, optimizer1, pre_trained_model)
    # load_net_opt(model_2, optimizer2, pre_trained_model)

    if 'state_dict' in torch.load(pre_trained_model).keys():
        ema_model.load_state_dict(torch.load(pre_trained_model)['state_dict'])
        model_1.load_state_dict(torch.load(pre_trained_model)['state_dict'])
        model_2.load_state_dict(torch.load(pre_trained_model)['state_dict'])
    else:
        ema_model.load_state_dict(torch.load(pre_trained_model))
        model_1.load_state_dict(torch.load(pre_trained_model))
        model_2.load_state_dict(torch.load(pre_trained_model))
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    model_1.train()
    model_2.train()
    ema_model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    dice_loss = losses.DiceLoss(n_classes=4)
    ce_loss = CrossEntropyLoss()
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            labeled_volume_batch_s = volume_batch_strong[:args.labeled_bs]
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b = get_ACDC_masks(pre_b, nms=1)
                pre_a_s = ema_model(uimg_a_s)
                pre_b_s = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)  # plab_a.shape=[6, 224, 224]
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)
                img_mask, loss_mask = generate_mask(img_a)
           
            consistency_weight = get_current_consistency_weight(iter_num//150)

            net_input_unl_1 = uimg_a * img_mask + img_a * (1 - img_mask)
            net_input_l_1 = img_b * img_mask + uimg_b * (1 - img_mask)
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0) 

            net_input_unl_2 = uimg_a_s * img_mask + img_a_s * (1 - img_mask)
            net_input_l_2 = img_b_s * img_mask + uimg_b_s * (1 - img_mask)
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            # Model1 Loss
            out_unl_1 = model_1(net_input_unl_1)
            out_l_1 = model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
            out_max_1 = torch.max(out_soft_1.detach(), dim=1)[0]
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False) 
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            # Model2 Loss
            out_unl_2 = model_2(net_input_unl_2)
            out_l_2 = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_max_2 = torch.max(out_soft_2.detach(), dim=1)[0]
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False) 
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            # Model1 & Model2 Cross Pseudo Supervision
            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))  
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))  
          

            # LICR(w)
            mix_factors = np.random.beta(
                0.1, 0.1, size=(args.labeled_bs//2, 1, 1, 1))
            mix_factors = torch.tensor(
                mix_factors, dtype=torch.float).cuda()
            unlabeled_volume_batch_0 = uimg_a
            unlabeled_volume_batch_1 = uimg_b

            # Mix images
            batch_ux_mixed = unlabeled_volume_batch_0 * (1.0 - mix_factors) + unlabeled_volume_batch_1 * mix_factors
            input_volume_batch = torch.cat(
                [labeled_volume_batch, batch_ux_mixed], dim=0)
            outputs = model_1(input_volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(
                    ema_model(unlabeled_volume_batch_0), dim=1)
                ema_output_ux1 = torch.softmax(
                    ema_model(unlabeled_volume_batch_1), dim=1)
                batch_pred_mixed = ema_output_ux0 * (1.0 - mix_factors) + ema_output_ux1 * mix_factors
            
            loss_ce_mix = ce_loss(outputs[:args.labeled_bs],
                              label_batch[:args.labeled_bs][:].long())
            loss_dice_mix = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            supervised_loss_mix = 0.5 * (loss_dice_mix + loss_ce_mix)
                
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            consistency_loss = torch.mean(
                (outputs_soft[args.labeled_bs:] - batch_pred_mixed) ** 2)*consistency_weight
            
                    #           LICR(s)
            mix_factors_s = np.random.beta(
                0.1, 0.1, size=(args.labeled_bs//2, 1, 1, 1))
            mix_factors_s = torch.tensor(
                mix_factors_s, dtype=torch.float).cuda()
            unlabeled_volume_batch_0 = uimg_a_s
            unlabeled_volume_batch_1 = uimg_b_s
            batch_ux_mixed_s = uimg_a_s* (1.0 - mix_factors_s) + uimg_a_s * mix_factors_s
            input_volume_batch_s = torch.cat(
                [labeled_volume_batch_s, batch_ux_mixed_s], dim=0)
            outputs_s = model_1(input_volume_batch_s)
            outputs_soft_s = torch.softmax(outputs_s, dim=1)
            with torch.no_grad():
                ema_output_ux0_s = torch.softmax(
                   pre_a_s, dim=1)
                ema_output_ux1_s = torch.softmax(
                    pre_b_s, dim=1)
                batch_pred_mixed_s = ema_output_ux0_s * (1.0 - mix_factors_s) + ema_output_ux1_s * mix_factors_s
            
            loss_ce_mix_s = ce_loss(outputs_s[:args.labeled_bs],
                              label_batch_strong[:args.labeled_bs][:].long())
            loss_dice_mix_s = dice_loss(
                outputs_soft_s[:args.labeled_bs], label_batch_strong[:args.labeled_bs].unsqueeze(1))
            
            supervised_loss_mix_s = 0.5 * (loss_dice_mix_s + loss_ce_mix_s)
            consistency_loss_s = torch.mean(
                (outputs_soft_s[args.labeled_bs:] - batch_pred_mixed_s) ** 2)*consistency_weight

            
            
#             FFT
            image_patch_last_FFT = process_image_batches(net_input_1, net_input_2,30)
            image_output_1_FFT = model_1(image_patch_last_FFT.unsqueeze(1))  
            image_output_soft_1_FFT = torch.softmax(image_output_1_FFT, dim=1)
            pseudo_image_output_1_FFT = torch.argmax(image_output_soft_1_FFT.detach(), dim=1, keepdim=False)
            image_output_2_FFT = model_2(image_patch_last_FFT.unsqueeze(1))
            image_output_soft_2_FFT = torch.softmax(image_output_2_FFT, dim=1)
            pseudo_image_output_2_FFT = torch.argmax(image_output_soft_2_FFT.detach(), dim=1, keepdim=False)
            # Model1 & Model2 Second Step Cross Pseudo Supervision
            pseudo_supervision5 = dice_loss(image_output_soft_1_FFT, pseudo_image_output_2_FFT.unsqueeze(1))
            pseudo_supervision6 = dice_loss(image_output_soft_2_FFT, pseudo_image_output_1_FFT.unsqueeze(1))
            

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1  + pseudo_supervision5
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2  + pseudo_supervision6
            loss = loss_1 + loss_2 + consistency_loss +supervised_loss_mix + consistency_loss_s +supervised_loss_mix_s 

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            update_model_ema(model_1, ema_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/model1_loss', loss_1, iter_num)
            writer.add_scalar('info/model2_loss', loss_2, iter_num)
            writer.add_scalar('info/model1/mix_dice', loss_dice_1, iter_num)
            writer.add_scalar('info/model1/mix_ce', loss_ce_1, iter_num)
            writer.add_scalar('info/model2/mix_dice', loss_dice_2, iter_num)
            writer.add_scalar('info/model2/mix_ce', loss_ce_2, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, model1_loss: %f, model2_loss: %f' % (iter_num, loss, loss_1, loss_2))

            if iter_num % 200 == 0:
                model_1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model_1,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                performance1 = np.mean(metric_list, axis=0)[0]

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    logging.info("BEST PERFORMANCE UPDATED AT ITERATION %d: Dice: %f" % (
                        iter_num, performance1))
                    save_best = os.path.join(snapshot_path, "iter{}_{}_model1_best_model.pth".format(iter_num,args.model))
                    torch.save(model_1.state_dict(), save_best)

                logging.info(
                    "iteration %d : model1_val_mean_dice : %f "
                    % (iter_num, performance1)
                )

                ###############
                # TEST, only use the result of the best val model
                # test_func(num_classes, db_test, model, refine_model, iter_num, testloader)
              

                ########################################################

                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"],
                        sampled_batch["label"],
                        model_2,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)

                performance2 = np.mean(metric_list, axis=0)[0]

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    logging.info("BEST PERFORMANCE UPDATED AT ITERATION %d: Dice: %f" % (
                        iter_num, performance2))
                    save_best = os.path.join(snapshot_path,
                                             "iter{}_{}_model2_best_model.pth".format(iter_num, args.model))
                    torch.save(model_2.state_dict(), save_best)

                logging.info(
                    "iteration %d : model_val_mean_dice : %f "
                    % (iter_num, performance2)
                )

                ###############
                # TEST, only use the result of the best val model
                # test_func(num_classes, db_test, model, refine_model, iter_num, testloader)
             
                ########################################################

                model_2.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break



if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        # -- path to save models
    pre_snapshot_path = "./modelmsc_betaFFT_1_re/pre_train"
    self_snapshot_path = "./modelmsc_betaFFT_1_re/self_train"
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('./train_beta_fft1.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, pre_snapshot_path, self_snapshot_path)