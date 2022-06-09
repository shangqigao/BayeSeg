import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
from .UNet import Unet
from .Resnet import ResNet

def sample_normal_jit(mu, log_var):
    sigma = torch.exp(log_var / 2)
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps


class BayeSeg(nn.Module):
    def __init__(self, args, freeze_whst=False):
        super(BayeSeg,self).__init__()
        
        self.args = args
        if freeze_whst:
            for p in self.parameters():
                p.requires_grad_(False)
                
        self.tasks = args.tasks
        
        # reconstruct clean image x and infer noise
        self.res_clean = ResNet(num_out_ch = 2)
        self.res_noise = ResNet(num_out_ch = 2, num_block=6, bn=True) #推断噪声的话，也许加BN的效果会更好一些？
        # pred mu and log var unit for seg_masks: B x K x W x H
        self.unet = Unet(out_ch = 8)
        
        # postprecess
        self.softmax = nn.Softmax(dim=1)
        
        # TODO: modify Dx & Dz 
        Dx = torch.zeros([1,1,3,3],dtype=torch.float)
        Dx[:,:,1,1] = 1
        Dx[:,:,1,0] = Dx[:,:,1,2] = Dx[:,:,0,1] = Dx[:,:,2,1] = -1/4
        self.Dx = nn.Parameter(data=Dx, requires_grad=False)
        
        
    def generate_m(self, samples):
        #m : mean of noise
        feature = self.res_noise(samples)
        mu_m, log_var_m = torch.chunk(feature, 2, dim=1)
        log_var_m = torch.clamp(log_var_m, -20, 0)
        m, _ = sample_normal_jit(mu_m, log_var_m)
        return m, mu_m, log_var_m
    
    def generate_x(self, samples):
        #x : clean image
        feature = self.res_clean(samples)
        mu_x, log_var_x = torch.chunk(feature, 2, dim=1)
        log_var_x = torch.clamp(log_var_x, -20, 0)
        x, _ = sample_normal_jit(mu_x, log_var_x)
        return x, mu_x, log_var_x
    
    def generate_z(self, x):
        #z : Seg logit
        feature = self.unet(x)
        mu_z, log_var_z = torch.chunk(feature, 2, dim=1)
        log_var_z = torch.clamp(log_var_z, -20, 0)
        z, _ = sample_normal_jit(mu_z, log_var_z)
        return self.softmax(z), self.softmax(mu_z), log_var_z

    def forward(self, samples: torch.Tensor, task):
        x, mu_x, log_var_x = self.generate_x(samples)
        m, mu_m, log_var_m = self.generate_m(samples)
        z, mu_z, log_var_z = self.generate_z(x)

        K = self.tasks[task]['out_channels'] # n_class
        
        #compute VB params
        ###################################
        # noise std rho
        residual = samples - (x + m)
        mu_rho_hat = (2*self.args.gamma_rho + 1) / (residual*residual + 2*self.args.phi_rho)
        normalization = torch.sum(mu_rho_hat).detach()
        n, _ = sample_normal_jit(m, torch.log(1 / mu_rho_hat))
        
        # Image line upsilon
        alpha_upsilon_hat = 2*self.args.gamma_upsilon + K
        difference_x = F.conv2d(mu_x, self.Dx, padding=1)
        beta_upsilon_hat = torch.sum(mu_z*(difference_x*difference_x + 2*torch.exp(log_var_x)),
                                     dim = 1, keepdim = True) + 2*self.args.phi_upsilon # B x 1 x W x H
        mu_upsilon_hat = alpha_upsilon_hat / beta_upsilon_hat
       
        # Seg boundary omega
        difference_z = F.conv2d(mu_z, self.Dx.expand(K,1,3,3), padding=1, groups=K) # B x K x W x H
        alpha_omega_hat = 2*self.args.gamma_omega + 1
        pseudo_pi = torch.mean(mu_z, dim=(2,3), keepdim=True)
        beta_omega_hat = pseudo_pi*(difference_z*difference_z + 2*torch.exp(log_var_z)) + 2*self.args.phi_omega
        mu_omega_hat = alpha_omega_hat / beta_omega_hat
 
        # Seg category probability pi
        _, _, W, H = samples.shape
        alpha_pi_hat = self.args.alpha_pi + W*H/2
        beta_pi_hat = torch.sum(mu_omega_hat*(difference_z*difference_z + 2*torch.exp(log_var_z)), dim=(2,3), keepdim=True)/2 + self.args.beta_pi
        digamma_pi = torch.special.digamma(alpha_pi_hat + beta_pi_hat) - torch.special.digamma(beta_pi_hat)
        
        # compute loss-related
        kl_y = residual*mu_rho_hat.detach()*residual

        kl_mu_z = torch.sum(digamma_pi.detach()*difference_z*mu_omega_hat.detach()*difference_z, dim=1)
        kl_sigma_z = torch.sum(digamma_pi.detach()*(2*torch.exp(log_var_z)*mu_omega_hat.detach() - log_var_z), dim=1)
        
        kl_mu_x = torch.sum(difference_x*difference_x*mu_upsilon_hat.detach()*mu_z.detach(), dim=1)
        kl_sigma_x = torch.sum(2*torch.exp(log_var_x)*mu_upsilon_hat.detach()*mu_z.detach(), dim=1) - log_var_x 
        
        kl_mu_m = self.args.sigma_0*mu_m*mu_m
        kl_sigma_m = self.args.sigma_0*torch.exp(log_var_m) - log_var_m

        visualize = {'recon':torch.concat([x, mu_x, torch.exp(log_var_x/2)]),
                     'noise':torch.concat([n, m, 1/mu_rho_hat.sqrt()]),
                     'logit':torch.concat([z[:,2:3,...], mu_z[:,2:3,...], torch.exp(log_var_z/2)[:,2:3,...]]),
                     'lines':mu_upsilon_hat, 'contour': mu_omega_hat[:,2:3,...],
                    }

        #visualize = {'y': samples, 'n': n, 'm': m, 'rho': mu_rho_hat, 'x': x, 'upsilon': mu_upsilon_hat, 'z': z, 'omega': mu_omega_hat}
        
        out = {'pred_masks': mu_z, 'kl_y':kl_y,
               'kl_mu_z':kl_mu_z, 'kl_sigma_z':kl_sigma_z,
               'kl_mu_x':kl_mu_x, 'kl_sigma_x':kl_sigma_x,
               'kl_mu_m':kl_mu_m, 'kl_sigma_m':kl_sigma_m,
               'normalization': normalization,
               'rho':mu_rho_hat, 
               'omega':mu_omega_hat*digamma_pi,
               'upsilon':mu_upsilon_hat*mu_z,
               'visualize':visualize, 
              }
        return out

class SetCriterion(nn.Module):
    """ This class computes the loss for BayeSeg.
    """
    def __init__(self, losses, weight_dict, args):
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.args = args
        
    def loss_Bayes(self, outputs, targets):
        N = outputs['normalization']
        loss_y = torch.sum(outputs['kl_y']) / N
        loss_mu_m = torch.sum(outputs['kl_mu_m']) / N
        loss_sigma_m = torch.sum(outputs['kl_sigma_m']) / N
        loss_mu_x = torch.sum(outputs['kl_mu_x']) / N
        loss_sigma_x = torch.sum(outputs['kl_sigma_x']) / N
        loss_mu_z = torch.sum(outputs['kl_mu_z']) / N
        loss_sigma_z = torch.sum(outputs['kl_sigma_z']) / N
        mean_rho = torch.mean(outputs['rho'])
        mean_omega = torch.mean(outputs['omega'])
        mean_upsilon = torch.mean(outputs['upsilon'])
        loss_Bayes = loss_y + loss_mu_m + loss_sigma_m + loss_mu_x + loss_sigma_x + loss_mu_z + loss_sigma_z
        losses = {
            'mean_rho':mean_rho,
            'mean_omega':mean_omega,
            'mean_upsilon':mean_upsilon,
            'loss_Bayes':loss_Bayes,
        }
        return losses 
    
    def loss_AvgDice(self, outputs, targets):
        src_masks = outputs["pred_masks"]
        src_masks = src_masks.argmax(1)
        targets_masks = targets.argmax(1)
        avg_dice = 0
        for i in range(1,4,1):
            dice=(2*torch.sum((src_masks==i)*(targets_masks==i),(1, 2)).float())/(torch.sum(src_masks==i,(1, 2)).float()+torch.sum(targets_masks==i,(1, 2)).float()+1e-10)
            avg_dice += dice.mean()
        return {"loss_AvgDice": avg_dice/3}
 
    def loss_CrossEntropy(self, outputs, targets, eps=1e-12):
        src_masks = outputs["pred_masks"]
        y_labeled = targets[:,0:4,:,:]
        cross_entropy = -torch.sum(y_labeled * torch.log(src_masks + eps), dim = 1)
        # criterion = nn.BCEWithLogitsLoss(reduction='none')
        # raw_loss = criterion(src_masks, y_labeled)
        losses = {
                "loss_CrossEntropy": cross_entropy.mean(),
        }
        return losses
            
    def get_loss(self, loss, outputs, targets):
        loss_map = {'Bayes': self.loss_Bayes,
                    'CrossEntropy': self.loss_CrossEntropy,
                    'AvgDice': self.loss_AvgDice,
                    }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        return losses


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)
        outputs_masks = (outputs_masks.sigmoid() > self.threshold).cpu()

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = F.interpolate(
                results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            ).byte()
        return results

class Visualization(nn.Module):
    def __init__(self):
        super().__init__()
        
    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
        
    def forward(self, inputs, outputs, labels, others, epoch, writer):
        self.save_image(inputs, 'inputs', epoch, writer)
        self.save_image(outputs.float(), 'outputs', epoch, writer)
        self.save_image(labels.float(), 'labels', epoch, writer)
        self.save_image(others['recon'].float(), 'recon', epoch, writer)
        self.save_image(others['noise'].float(), 'noise', epoch, writer)
        self.save_image(others['logit'].float(), 'logit', epoch, writer)
        self.save_image(others['lines'].float(), 'lines', epoch, writer)
        self.save_image(others['contour'].float(), 'contour', epoch, writer)


def build(args):
    device = torch.device(args.device)
    model = BayeSeg(args, freeze_whst=(args.frozen_weights is not None))
    weight_dict = {
        'loss_CrossEntropy': args.CrossEntropy_loss_coef,
#        'loss_AvgDice': args.AvgDice_loss_coef,  
        'loss_Bayes':args.Bayes_loss_coef,
    }
    losses = ['CrossEntropy','AvgDice','Bayes']
    criterion = SetCriterion(losses=losses, weight_dict=weight_dict, args=args)
    criterion.to(device)
    visualizer = Visualization()
    postprocessors = {'segm': PostProcessSegm()}

    return model, criterion, postprocessors, visualizer


