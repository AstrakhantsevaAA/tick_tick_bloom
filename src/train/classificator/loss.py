from torch import Tensor, sqrt, squeeze
from torch.nn import MSELoss
from torch.nn.functional import mse_loss


class DensityMSELoss(MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        eps = 1e-6
        loss = sqrt(mse_loss(squeeze(input), target, reduction=self.reduction) + eps)
        return loss
