import torch


def _lovasz_grad(gt_sorted):
  r'''Gradient of the Lovasz extension with respect to sorted errors.'''
  num = len(gt_sorted)
  positives = gt_sorted.sum()
  intersection = positives - gt_sorted.float().cumsum(0)
  union = positives + (1.0 - gt_sorted).float().cumsum(0)
  jaccard = 1.0 - intersection / union
  if num > 1:
    jaccard = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
  return jaccard


def lovasz_softmax_loss(logit, label, ignore_index=-1):
  r'''Multiclass Lovasz-Softmax loss for flattened point predictions.

  This follows the loss used by Pointcept's PTv3 S3DIS configuration while
  keeping the current project independent from the Pointcept runtime.
  '''
  label = label.reshape(-1)
  valid = label != ignore_index
  label, prob = label[valid], torch.softmax(logit[valid], dim=1)
  if label.numel() == 0:
    return logit.sum() * 0.0

  losses = []
  for category in label.unique():
    foreground = label.eq(category).type_as(prob)
    errors = torch.abs(foreground - prob[:, category])
    errors_sorted, permutation = torch.sort(errors, descending=True)
    foreground_sorted = foreground[permutation]
    losses.append(torch.dot(errors_sorted, _lovasz_grad(foreground_sorted)))
  return torch.stack(losses).mean()
