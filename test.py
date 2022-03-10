from __future__ import absolute_import, division, print_function

import logging
import os


import cv2
import numpy as np
import pandas
import torch
import torch.nn as nn
from imageio import imsave
from tqdm import tqdm

logger = logging.getLogger(__name__)

import os
import sys
from tensorboardX import SummaryWriter

import cfg
from utils.utils import create_logger, set_log_dir

import datasets
from celeba import AdultTable
import models_search

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def validate(args, fixed_z, epoch, gen_net: nn.Module):

    # eval mode
    gen_net.eval()

    #     generate images
    with torch.no_grad():
        # sample_imgs = gen_net(fixed_z, epoch)
        # img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

        eval_iter = args.num_eval_imgs // args.eval_batch_size
        img_list = list()
        for iter_idx in tqdm(range(eval_iter), desc="sample images"):
            z = torch.cuda.FloatTensor(
                np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim))
            )

            # Generate a batch of images
            gen_tabular_num, gen_tabular_cat = gen_net(z, epoch)

            print(gen_tabular_num.shape, gen_tabular_cat.shape)

            real_gen_tabular_cat = torch.argmax(gen_tabular_cat, -1)

            img_list.extend(
                list(
                    torch.cat([gen_tabular_num, real_gen_tabular_cat], dim=1)
                    .to("cpu")
                    .numpy()
                )
            )

        # print(len(img_list))
        # print(img_list[0])

        # sys.exit()

    return pandas.DataFrame(img_list)


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    assert args.exp_name
    #     assert args.load_path.endswith('.pth')
    assert os.path.exists(args.load_path)
    args.path_helper = set_log_dir("logs_eval", args.exp_name)
    logger = create_logger(args.path_helper["log_path"], phase="test")

    dataset = AdultTable(None)
    args.cat_size, args.cats, args.num_size = dataset.get_cat_num_size()
    dataset_name = "adult"
    gen_net = eval("models_search." + args.gen_model + ".Generator")(args=args).cuda()
    gen_net = torch.nn.DataParallel(gen_net.to("cuda:0"), device_ids=[0])

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (4, args.latent_dim)))

    # set writer
    logger.info(f"=> resuming from {args.load_path}")
    checkpoint_file = args.load_path
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    if "avg_gen_state_dict" in checkpoint:
        print(checkpoint["avg_gen_state_dict"].keys())
        gen_net.load_state_dict(checkpoint["avg_gen_state_dict"])
        epoch = checkpoint["epoch"]
        logger.info(f"=> loaded checkpoint {checkpoint_file} (epoch {epoch})")
    else:
        gen_net.load_state_dict(checkpoint)
        logger.info(f"=> loaded checkpoint {checkpoint_file}")

    logger.info(args)

    sample = validate(
        args,
        fixed_z,
        epoch,
        gen_net,
    )

    print(sample.shape)

    sample = dataset.recover_norm_data(sample)[dataset.old_header]

    # sample = sample[dataset.old_header]

    # print(sample.describe())

    filename = f"/home/fjd5166/tabular/sdv/transgan_gen_{dataset_name}_data.csv"
    sample.to_csv(
        filename,
        index=False,
    )


if __name__ == "__main__":
    main()
