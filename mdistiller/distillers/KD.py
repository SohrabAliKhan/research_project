import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

# class wise 1
def class_wise_standardization(logits, target, num_classes=100):
    """
    Standardizes logits across the batch for each class.

    Args:
        logits (Tensor): [batch_size, num_classes]
        target (Tensor): [batch_size] - class labels for each sample
        num_classes (int): Number of classes (like 100 for CIFAR-100)

    Returns:
        standardized_logits (Tensor): Class-wise standardized logits
    """
    class_means = torch.zeros(num_classes, device=logits.device)
    class_stds = torch.zeros(num_classes, device=logits.device)

    for c in range(num_classes):
        class_mask = (target == c)
        if class_mask.sum() > 0:
            class_means[c] = logits[class_mask].mean()
            class_stds[c] = logits[class_mask].std()

    standardized_logits = torch.zeros_like(logits)

    for c in range(num_classes):
        class_mask = (target == c)
        if class_mask.sum() > 0:
            standardized_logits[class_mask] = (logits[class_mask] - class_means[c]) / (class_stds[c] + 1e-6)

    return standardized_logits
#  class wise end for 

# def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
#     logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
#     logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
#     log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
#     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
#     loss_kd *= temperature**2
#     return loss_kd


# class wise 3
def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND 
        
        # class wise 2
    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
         logits_teacher, _ = self.teacher(image)

        if self.logit_stand:
            logits_student = class_wise_standardization(logits_student, target, num_classes=logits_student.shape[1])
            logits_teacher = class_wise_standardization(logits_teacher, target, num_classes=logits_teacher.shape[1])

    # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


    # def forward_train(self, image, target, **kwargs):
    #     logits_student, _ = self.student(image)
    #     with torch.no_grad():
    #         logits_teacher, _ = self.teacher(image)

    #     # losses
    #     loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
    #     loss_kd = self.kd_loss_weight * kd_loss(
    #         logits_student, logits_teacher, self.temperature, self.logit_stand
    #     )
    #     losses_dict = {
    #         "loss_ce": loss_ce,
    #         "loss_kd": loss_kd,
    #     }
    #     return logits_student, losses_dict
