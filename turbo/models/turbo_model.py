import itertools
import random

import torch
import lpips

from turbo.utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
try:
    from apex import amp
except ImportError as error:
    print(error)


class TurboModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_est_A: G_B(B) vs. A; D_est_B: G_A(A) vs. B.
                        D_rec_A: G_B(G_A(A)) vs. A; D_rec_B: G_A(G_B(B)) vs. B.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        # parser.set_defaults(no_dropout=True, no_antialias=True, no_antialias_up=True)  # default CycleGAN did not use dropout
        # parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--amp', action='store_true', help='enables or disables automatic mixed precision')
            parser.add_argument('--same_D', action='store_true', help='use the same discriminator for both estimation and reconstruction')
            # parser.add_argument('--amp', type=str2bool, default=False, help='enables or disables automatic mixed precision')
            # parser.add_argument('--same_D', type=str2bool, default=False, help='use the same discriminator for both estimation and reconstruction')
            parser.add_argument('--p_flip', type=float, default=0, help='probabilty of flipping real and fake for discriminator')
            parser.add_argument('--p_noise', type=float, default=0, help="probabilty of adding noise to discriminator's input")
            parser.add_argument('--w_noise', type=float, default=0, help="weight for adding noise to discriminator's input")
            parser.add_argument('--G_threshold', type=float, default=float('inf'), help="update discriminator if generator loss less than")
            parser.add_argument('--D_threshold', type=float, default=float('-inf'), help="update discriminator if discriminator loss greater than")

            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_estimation', type=float, default=0., help='use estimation loss (only with aligned dataset). Setting lambda_estimation other than 0 has an effect of scaling the weight of the estimation loss. For example, if the weight of the estimation loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_estimation = 0.1')

            parser.add_argument('--lambda_LPIPS', type=float, default=0, help='weight for LPIPS loss')

        return parser

    @staticmethod
    def create_D(nc, opt):
        """Create a discriminator"""
        return networks.define_D(nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,
                                 opt.gpu_ids)

    def __init__(self, opt):
        """Initialize the Turbo class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['cycle_A', 'idt_A', 'cycle_B', 'idt_B'] + [f'{model}_{kind}_{domain}' for domain, kind, model in itertools.product(['A', 'B'], ['est', 'rec'], ['D', 'G'])]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_est_A', 'D_est_B', 'D_rec_A', 'D_rec_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        if self.isTrain:
            if opt.lambda_estimation > 0.0:
                assert opt.dataset_mode in ['aligned', 'space'], 'Estimation loss is only supported for aligned dataset'
                self.loss_names += ['est_A', 'est_B']
            if opt.gan_mode == 'wgangp':
                self.loss_names += [f'D_{kind}_GP_{domain}' for domain, kind in itertools.product(['A', 'B'], ['est', 'rec'])]

            if opt.lambda_LPIPS > 0.0:
                self.loss_names += ['G_LPIPS_A', 'G_LPIPS_B']
                self.criterionLPIPS = lpips.LPIPS(net='vgg').cuda()

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_est_A = self.create_D(opt.input_nc, opt)
            self.netD_est_B = self.create_D(opt.output_nc, opt)
            D_parameters = [self.netD_est_A.parameters(), self.netD_est_B.parameters()]
            if opt.same_D:
                self.netD_rec_A = self.netD_est_A
                self.netD_rec_B = self.netD_est_B
            else:
                self.netD_rec_A = self.create_D(opt.input_nc, opt)
                self.netD_rec_B = self.create_D(opt.output_nc, opt)
                D_parameters.extend([self.netD_rec_A.parameters(), self.netD_rec_B.parameters()])

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.est_A_pool, self.rec_A_pool = ImagePool(opt.pool_size), ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.est_B_pool, self.rec_B_pool = ImagePool(opt.pool_size), ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionEst = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(*D_parameters), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.est_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.est_B)   # G_B(G_A(A))
        self.est_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.est_A)   # G_A(G_B(B))
        self.fake_A, self.fake_B = self.est_A.detach(), self.est_B.detach()

    def backward_D_basic(self, netD, real, fake, loss_G):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        update_loss = True
        if random.random() < self.opt.p_flip:
            real, fake = fake.detach(), real
            update_loss = False
        if random.random() < self.opt.p_noise:
            real += torch.randn_like(real) * self.opt.w_noise
            fake += torch.randn_like(fake) * self.opt.w_noise
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        if loss_G <= self.opt.G_threshold or loss_D >= self.opt.D_threshold:
            if self.opt.amp:
                with amp.scale_loss(loss_D, self.optimizer_D) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_D.backward()
        return loss_D, update_loss

    def backward_D(self):
        """Calculate GAN loss for discriminator"""
        for kind, domain in itertools.product(['est', 'rec'], ['A', 'B']):
            fake = getattr(self, f'{kind}_{domain}')
            real = getattr(self, f'real_{domain}')
            net_D = getattr(self, f'netD_{kind}_{domain}')
            if self.opt.gan_mode == 'wgangp':
                loss_D_GP, _ = networks.cal_gradient_penalty(net_D, real, fake.detach().clone(), self.device)
                loss_D_GP.backward(retain_graph=True)
                setattr(self, f'loss_D_{kind}_GP_{domain}', loss_D_GP)
            fake = getattr(self, f'{kind}_{domain}_pool').query(fake)
            loss_G = getattr(self, f'loss_G_{kind}_{domain}')
            loss_D, update_loss = self.backward_D_basic(net_D, real, fake, loss_G)
            if update_loss:
                setattr(self, f'loss_D_{kind}_{domain}', loss_D)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_est = self.opt.lambda_estimation
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # Estimation loss
        if lambda_est > 0:
            self.loss_est_A = self.criterionEst(self.est_A, self.real_A) * lambda_A * lambda_est
            self.loss_est_B = self.criterionEst(self.est_B, self.real_B) * lambda_B * lambda_est
        else:
            self.loss_est_A = 0
            self.loss_est_B = 0

        self.total_loss_G = 0
        for kind, domain in itertools.product(['est', 'rec'], ['A', 'B']):
            fake = getattr(self, f'{kind}_{domain}')
            net_D = getattr(self, f'netD_{kind}_{domain}')
            loss_G = self.criterionGAN(net_D(fake), True)
            self.total_loss_G += loss_G
            setattr(self, f'loss_G_{kind}_{domain}', loss_G)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.total_loss_G += self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_est_A + self.loss_est_B

        # LPIPS loss
        if self.opt.lambda_LPIPS > 0:
            self.loss_G_LPIPS_B = self.criterionLPIPS(self.real_B, self.fake_B).mean() * self.opt.lambda_LPIPS
            self.loss_G_LPIPS_A = self.criterionLPIPS(self.real_A, self.fake_A).mean() * self.opt.lambda_LPIPS
            self.total_loss_G += self.loss_G_LPIPS_A + self.loss_G_LPIPS_B

        self.total_loss_G.backward()

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_est_A, self.netD_rec_A, self.netD_est_B, self.netD_rec_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_est_A, self.netD_rec_A, self.netD_est_B, self.netD_rec_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for all discriminators
        self.optimizer_D.step()  # update D_A and D_B's weights
