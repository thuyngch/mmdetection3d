#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from scipy.stats import norm

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.losses.focal_loss import sigmoid_focal_loss


#------------------------------------------------------------------------------
#  _expand_binary_labels
#------------------------------------------------------------------------------
def _expand_binary_labels(labels, num_fg_classes):
	"""
	[Input]
		labels: shape [N], range [0, num_fg_classes] where num_fg_classes stands for class `background`
		num_fg_classes: number of foreground classes, not include the class `background`

	[Output]
		onehot_labels: shape [N, num_fg_classes]
	"""
	onehot_labels = F.one_hot(labels, num_fg_classes+1)
	onehot_labels = onehot_labels[:, :-1]	# Ignore class `background` in term of using sigmoid
	return onehot_labels


#------------------------------------------------------------------------------
#  AutoFocalLoss
#------------------------------------------------------------------------------
@LOSSES.register_module
class AutoFocalLoss(nn.Module):

	def __init__(self, use_sigmoid=False, loss_weight=1.0, gamma=2.0, alpha=0.5, reduction='mean'):
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
			cls_channels = pred.shape[1]
			bin_target = _expand_binary_labels(target, cls_channels)
			bin_target = bin_target.float()

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

			loss_cls = self.loss_weight * sigmoid_focal_loss(
				pred, target, weight,
				gamma=self.gamma, alpha=self.alpha,
				reduction=reduction, avg_factor=avg_factor)
		else:
			raise NotImplementedError

		return loss_cls
