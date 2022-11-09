import torch

import lpips

from .base_model import BaseModel
from . import networks


class Pix2PixLPIPSModel(BaseModel):

    """This class implements the pix2pix L2 model, which uses L2 and LPIPS losses

    The model training requires `--dataset_model aligned` dataset
    By default, it uses a '--netG unet256' U-Net generator
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or
                               test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: lambda_L2 * ||G(A)-B||_2
        By default, we use UNet with batchnorm, and aligned datasets.
        """

        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L2', type=float, default=5., help='weight for L2 loss')
            parser.add_argument('--lambda_LPIPS', type=float, default=1., help='weight for LPIPS loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'G_LPIPS', 'G']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionLPIPS = lpips.LPIPS(net='vgg').cuda()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_G(self):
        """Calculate loss for the generator"""
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * self.opt.lambda_L2
        self.loss_G_LPIPS = self.criterionLPIPS(self.fake_B, self.real_B).mean() * self.opt.lambda_LPIPS
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L2 + self.loss_G_LPIPS
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                    # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # update G's weights
