import argparse
import logging
import os

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2

from model import newUNet, threeBnewUNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils import one_hot

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    image = BasicDataset.preprocess(full_img, scale_factor)
    image = torch.from_numpy(image)

    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(image)

        temp = output[0].cpu()
        temp = temp.detach().numpy()#[0]

        if net.n_classes > 1:
            probs = F.softmax(output[0], dim=1)
        else:
            probs = torch.sigmoid(output[0])[0]

        probs = probs.cpu().detach().numpy()

        # probs = numpy.squeeze(probs, axis=0)
        # probs = one_hot.onehot_to_mask(probs, palette=[[0], [20], [39], [58], [77],
        #  [96], [115], [134], [153], [172], [191], [210], [229], [248]])
        # probs = one_hot.onehot_to_mask(probs, palette=[[0], [255]])
        # probs = probs.transpose([2, 0, 1])

        # -------预测图放大-------
        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[0]),
        #         transforms.ToTensor()
        #     ]
        # )
        #
        # probs = tf(probs[0])
        # probs = probs * 255
        # probs = probs.numpy()

    # return full_mask > out_threshold
    return probs[0] #newmodel
    # return probs #UNet

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./checkpoints/3Dircadb_s1+s2_epoch100.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    img = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    img = Image.fromarray(img[1].astype(np.uint8))
    return img

def predict_function(inpath, outpath):
    args = get_args()
    in_files = inpath
    out_files = outpath

    net = newUNet(n_channels=1, n_classes=1)  #channel

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device)['net'])

    logging.info("Model loaded !")

    # for i, fn in enumerate(in_files):
    # logging.info("\nPredicting image {} ...".format(fn))

    # img = Image.open(fn)
    img = Image.open(in_files)

    mask = predict_img(net=net, full_img=img, scale_factor=args.scale,
                       out_threshold=args.mask_threshold, device=device)

    if not args.no_save:
        result = mask_to_image(mask)
        # result.save(out_files[i])
        result.save(out_files)

        # logging.info("Mask saved to {}".format(out_files[i]))

    if args.viz:
        # logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        plot_img_and_mask(img, mask)

if __name__ == "__main__":
    testpath = 'D:/DataSet/3Dircadb/next_ct/testct/1.4/'
    # testpath = 'D:/DataSet/LiTs/test/11,12CT/'
    for root, dirs, files in os.walk(testpath):
        for f in files:
            path1 = os.path.join(root, f)
            outpath = os.path.join('D:\\DataSet\\3Dircadb\\s1+s2Result\\1.4\\', f)
            predict_function(path1, outpath)