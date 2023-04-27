import torch

from cc_torch import connected_components_labeling
from scipy.ndimage import label
from torch import nn
from torch.nn import functional as F

from nnunet.training.loss_functions.dice_loss import SoftDiceLoss, TopKLoss, DC_and_CE_loss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss


class LCDloss(nn.Module):
    def __init__(self, batch_lcd=True, nonlin=True):
        """Label distance and count loss."""
        super().__init__()
        self.batch_lcd = batch_lcd
        self.nonlin = nonlin

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Calculate LCD loss for network predictions.
        The loss is solely based on assumptions derived from the data itself. Thus, the ground truth does not play any
        role during calculation.

        :param x: Network prediction, shape(batch, class, x, y(, z))
        :type x: torch.Tensor
        :param y: Ground truth, label map shape((batch, 1, x, y(, z)) or (batch, x, y(, z))) or
            one_hot_encoding shape(batch, class, x, y(, z))
        :type y: torch.Tensor
        :return: summed loss values
        :rtype: torch.Tensor
        """
        lc_loss = 0
        ld_loss = 0
        bin_x = torch.where(x > 0, 1, 0).float()
        for batch_vol in torch.unbind(bin_x):
            for class_vol in torch.unbind(batch_vol.type(torch.uint8)[1:, ...]):
                if class_vol.device.type == "cuda":
                    cc = connected_components_labeling(class_vol)
                else:
                    cc = torch.from_numpy(label(class_vol)[0])
                lcount = cc.unique().count_nonzero().item()
                if lcount > 1:
                    lc_loss += 1
            if batch_vol.ndim == 4:
                conv = F.conv3d(
                    torch.unsqueeze(batch_vol[1:, ...], 1),
                    weight=torch.ones((1, 1, 3, 3, 3), dtype=bin_x.dtype, device=bin_x.device)
                )
            elif batch_vol.ndim == 3:
                conv = F.conv2d(
                    torch.unsqueeze(batch_vol[1:, ...], 1),
                    weight=torch.ones((1, 1, 3, 3), dtype=bin_x.dtype, device=bin_x.device)
                )
            else:
                raise ValueError('Only 3D and 2D data supported!')
            csum = conv.sum(dim=0, keepdim=True)
            diff = conv - csum
            mask = conv != 0
            diff *= mask
            ld_loss += diff.sum()
        lcd_loss = (lc_loss * 100) - (ld_loss / 10)
        if self.nonlin:
            lcd_loss = torch.log(lcd_loss) if lcd_loss != 0 else 0
        if self.batch_lcd:
            return lcd_loss / x.shape[0]
        return lcd_loss


class LCD_DC_CE_loss(DC_and_CE_loss):
    def __init__(self, lcd_kwargs={}, weight_lcd=1., **kwargs):
        """

        :param lcd_kwargs: LCDloss initialisation arguments
        :type lcd_kwargs: dict
        :param weight_lcd: lcd_loss weight, range 0-1, modulates lcd loss influence
        :type weight_lcd: float
        :param kwargs: positional arguments of DC_and_CE_loss, one dict per argument (soft_dice_kwargs, ce_kwargs)
        """
        super(LCD_DC_CE_loss, self).__init__(**kwargs)
        self.lcd_kwargs = lcd_kwargs
        self.weight_lcd = weight_lcd
        self.lcd = LCDloss(**lcd_kwargs)

    def forward(self, net_output, target):
        """
        :param net_output: Network prediction, shape(batch, class, x, y(, z))
        :type net_output: torch.Tensor
        :param target: Ground truth, label map shape((batch, 1, x, y(, z)) or (batch, x, y(, z))) or
            one_hot_encoding shape(batch, class, x, y(, z))
        :type target: torch.Tensor
        :return: summed loss values
        :rtype: torch.Tensor
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        lcd_loss = self.lcd(net_output, target) if self.weight_lcd != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_lcd * lcd_loss + self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result

