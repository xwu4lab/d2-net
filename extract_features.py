import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--image_list_file', type=str, required=True,
    help='path to a file containing a list of images to process'
)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default=None,
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--output_extension', type=str, default='.pspnet-res50-cs',
    help='extension for the output'
)
parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

parser.add_argument(
    '--model_type', type=str, default='res50',
    help='type of pretrained model'
)
parser.add_argument(
    '--truncated_blocks', type=int, default=2,
    help='number of truncated blocks'
)
parser.add_argument(
    '--edge_threshold', type=float, default=5,
    help='edge threshold to select feature'
)
parser.add_argument(
    '--dilation_blocks', type=int, default=1,
    help='use dilation or not'
)
parser.add_argument(
    '--pspnetTest', type=int, default=0,
    help='use PSPNet Model or not'
)
args = parser.parse_args()

print(args)

# calculate scaling step 
if args.model_type == 'vgg16':
    scaling_steps = 5-args.truncated_blocks
elif args.model_type == 'res50':
    scaling_steps = 6-args.dilation_blocks-args.truncated_blocks
elif args.model_type == 'res101':
    scaling_steps = 6-args.dilation_blocks-args.truncated_blocks

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_relu=args.use_relu,
    use_cuda=use_cuda,
    truncated_blocks=args.truncated_blocks,
    model_type=args.model_type,
    edge_threshold=args.edge_threshold,
    dilation_blocks=args.dilation_blocks,
    pspnetTest=args.pspnetTest
)

print(model)

if use_cuda:
    model = model.cuda()

# Process the file
with open(args.image_list_file, 'r') as f:
    lines = f.readlines()
for line in tqdm(lines, total=len(lines)):
    path = line.strip()

    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scaling_steps=scaling_steps
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1],
                scaling_steps=scaling_steps
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if args.output_type == 'npz':
        with open(path + args.output_extension, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints=keypoints,
                scores=scores,
                descriptors=descriptors
            )
    elif args.output_type == 'mat':
        with open(path + args.output_extension, 'wb') as output_file:
            scipy.io.savemat(
                output_file,
                {
                    'keypoints': keypoints,
                    'scores': scores,
                    'descriptors': descriptors
                }
            )
    else:
        raise ValueError('Unknown output type.')
