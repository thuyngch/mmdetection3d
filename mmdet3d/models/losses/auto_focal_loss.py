import torch, math
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.focal_loss import sigmoid_focal_loss


@LOSSES.register_module
class AutoFocalLoss(nn.Module):

	def __init__(self, use_sigmoid=False, gamma=1e-3, alpha=0.25, loss_weight=1.0, reduction='mean'):
		super(AutoFocalLoss, self).__init__()
		assert use_sigmoid is True, 'Only sigmoid focaloss supported now.'
		self.use_sigmoid = use_sigmoid
		self.loss_weight = loss_weight
		self.initial_gamma = gamma
		self.alpha = alpha
		self.p_avg = gamma
		self.reduction = reduction

	def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
		"""
		[input]
			pred: logits, shape [N, C] where C is number of foreground classes
			target: class indicies, shape [N], range [0, C] where C stands for class `background`
		"""
		assert reduction_override in (None, 'none', 'mean', 'sum')
		reduction = (
			reduction_override if reduction_override else self.reduction)

		if self.use_sigmoid:
			# Parse ignored samples
			if avg_factor is not None:
				avg_factor -= (target < 0).sum()
			weight[target < 0] = 0
			target[target < 0] = 0

			# Expand onehot labels
			cls_channels = pred.shape[1]
			bin_target = F.one_hot(target, cls_channels+1)
			bin_target = bin_target[:, :-1]	# Ignore class `background` in term of using sigmoid

			# Adjust gamma with momentum
			p = pred.sigmoid()
			p_score = p * bin_target
			p_sum = torch.sum(p_score, 1)
			n_non0 = torch.nonzero(p_sum).shape[0]

			if n_non0 !=0:
				p_avg = (torch.sum(p_sum) / n_non0).float().item()
			else:
				p_avg = self.p_avg

			if self.p_avg == self.initial_gamma:
				self.p_avg = p_avg
			else:
				self.p_avg = 0.05*p_avg + 0.95*self.p_avg

			self.gamma = -math.log(self.p_avg)

			# Compute focal loss
			loss_cls = self.loss_weight * sigmoid_focal_loss(
				pred, target, weight,
				gamma=self.gamma, alpha=self.alpha,
				reduction=reduction, avg_factor=avg_factor)
		else:
			raise NotImplementedError

		return loss_cls
