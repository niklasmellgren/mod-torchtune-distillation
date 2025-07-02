# Original: torchtune/modules/loss/kd_losses.py
#
# Modifications by Niklas Mellgren
#
# Added extra KD losses (Reverse, Symmetric, JS, TV, + chunked variants)
# Added memory-friendly “WithChunkedOutput” wrappers

from typing import List, Optional

import torch
import torch.nn.functional as F


class ForwardKLLoss(torch.nn.Module):
    """
    The Kullback-Leibler divergence loss for valid indexes.
    Implementation of https://github.com/jongwooko/distillm/blob/17c0f98bc263b1861a02d5df578c84aea652ee65/distillm/losses.py

    Args:
        ignore_index (int):  Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).
        """

        teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(student_logits)
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()
        if not normalize:
            return -torch.sum(x * mask.view(-1), dim=0)
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=x.device)
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class ReverseKLLoss(torch.nn.Module):
    """
    The reverse Kullback-Leibler divergence loss.

    Args:
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            Default: -100
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: Reverse KL divergence loss.
        """
        student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)

        inf_mask = torch.isinf(teacher_logits) | torch.isinf(student_logits)
        prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)

        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).int()

        if not normalize:
            return -torch.sum(x * mask.view(-1), dim=0)
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=x.device)
        return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class SymmetricKLLoss(torch.nn.Module):
    """
    Symmetric KL divergence loss that combines forward and reverse KL.

    Args:
        ignore_index (int): Specifies a target value that is ignored.
            Default: -100
        lambda_weight (float): Weight for combining forward and reverse KL.
            Default: 0.9
    """
    def __init__(self, ignore_index: int = -100, lambda_weight: float = 0.9):
        super().__init__()
        self.forward_kl = ForwardKLLoss(ignore_index)
        self.reverse_kl = ReverseKLLoss(ignore_index)
        self.lambda_weight = lambda_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: Symmetric KL divergence loss.
        """
        for_kl = self.forward_kl(student_logits, teacher_logits, labels, normalize)
        rev_kl = self.reverse_kl(student_logits, teacher_logits, labels, normalize)
        return (1 - self.lambda_weight) * for_kl + self.lambda_weight * rev_kl


class JSDistanceLoss(torch.nn.Module):
    """
    Jensen-Shannon distance loss.

    Args:
        ignore_index (int): Specifies a target value that is ignored.
            Default: -100
        lambda_weight (float): Weight parameter for mixing distributions.
            Default: 0.9
    """
    def __init__(self, ignore_index: int = -100, lambda_weight: float = 0.9):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_weight = lambda_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: JS distance loss.
        """
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
        mixed_probs = (1 - self.lambda_weight) * teacher_probs + self.lambda_weight * student_probs

        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        mixed_logprobs = torch.log(mixed_probs)

        mask = (labels != self.ignore_index).int()
        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)

        # Student term
        student_term = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
        student_term -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        student_loss = torch.sum(student_term, dim=-1).view(-1)

        # Teacher term
        teacher_term = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        teacher_term -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
        teacher_loss = torch.sum(teacher_term, dim=-1).view(-1)

        total_loss = (
            self.lambda_weight * -torch.sum(student_loss * mask.view(-1), dim=0) +
            (1 - self.lambda_weight) * -torch.sum(teacher_loss * mask.view(-1), dim=0)
        )

        if not normalize:
            return total_loss
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=student_logits.device)
        return total_loss / torch.sum(mask.view(-1), dim=0)


class TVDistanceLoss(torch.nn.Module):
    """
    Total Variation distance loss.

    Args:
        ignore_index (int): Specifies a target value that is ignored.
            Default: -100
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): logits from student model of shape
                (batch_size*num_tokens, vocab_size).
            teacher_logits (torch.Tensor): logits from teacher model of shape
                (batch_size*num_tokens, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape
                (batch_size, vocab_size).
            normalize (bool): Whether to normalize the loss by the number of unmasked elements.

        Returns:
            torch.Tensor: TV distance loss.
        """
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)

        mask = (labels != self.ignore_index).int()
        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
        prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)

        if not normalize:
            return torch.sum(x * mask.view(-1), dim=0)
        if torch.sum(mask.view(-1), dim=0) == 0:
            return torch.tensor(0.0, device=x.device)
        return torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class ForwardKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Forward KL with chunked outputs that saves memory by only upcasting one chunk at a time.

    Since the model is trained with bf16, before computing KL divergence, we have to upcast
    it to fp32 for better accuracy and stability. When upcasting happens, the memory usage doubles.
    Models like llama3 have large vocabulary size and, therefore, have a large output
    result (bsz, num_tokens, vocab_size). If we chunk on the token level, you can still compute
    the cross entropy normally, but upcasting only one chunk at a time saves considerable memory.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into. Each chunk has shape
            (batch_size, num_tokens / num_output_chunks, vocab_size).
            Default: 8
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
            The loss is divided over non-ignored targets.
            Default: -100
    """

    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.fkl_loss = ForwardKLLoss(ignore_index)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model of length
                ``self.num_output_chunks``, where each chunk has shape
                (batch_size, num_tokens / num_output_chunks, vocab_size).
            labels (torch.Tensor): Ground truth labels of shape (batch_size, num_tokens).

        Returns:
            torch.Tensor: KL divergence loss of shape (1,).

        Example:
            >>> loss_fn = ForwardKLWithChunkedOutputLoss()
            >>>
            >>> h = torch.tensor([bsz, num_tokens, dim])
            >>> output_chunks = [model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> teacher_chunks = [teacher_model.output(chunk) for chunk in h.chunk(num_chunks, dim=1)]
            >>> labels = torch.tensor([bsz, num_tokens])
            >>> loss = loss_fn(output_chunks, teacher_chunks, labels)
        """

        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        mask = (labels != self.ignore_index).int()
        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        total_fkl_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_fkl_loss += self.fkl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_fkl_loss / torch.sum(mask.view(-1), dim=0)


class ReverseKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Reverse KL with chunked outputs that saves memory by only upcasting one chunk at a time.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into.
            Default: 8
        ignore_index (int): Specifies a target value that is ignored.
            Default: -100
    """
    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.rkl_loss = ReverseKLLoss(ignore_index)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model.
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Reverse KL divergence loss.
        """
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        mask = (labels != self.ignore_index).int()
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        total_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_loss += self.rkl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_loss / torch.sum(mask.view(-1), dim=0)


class SymmetricKLWithChunkedOutputLoss(torch.nn.Module):
    """
    Symmetric KL divergence with chunked outputs that saves memory by only upcasting one chunk at a time.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into.
            Default: 8
        ignore_index (int): Specifies a target value that is ignored.
            Default: -100
        lambda_weight (float): Weight for combining forward and reverse KL.
            Default: 0.9
    """
    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100, lambda_weight: float = 0.9):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.lambda_weight = lambda_weight
        self.skl_loss = SymmetricKLLoss(ignore_index, lambda_weight)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model.
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Symmetric KL divergence loss.
        """
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        mask = (labels != self.ignore_index).int()
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        total_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_loss += self.skl_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_loss / torch.sum(mask.view(-1), dim=0)


class JSDistanceWithChunkedOutputLoss(torch.nn.Module):
    """
    Jensen-Shannon distance with chunked outputs that saves memory by only upcasting one chunk at a time.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into.
            Default: 8
        ignore_index (int): Specifies a target value that is ignored.
            Default: -100
        lambda_weight (float): Weight parameter for mixing distributions.
            Default: 0.9
    """
    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100, lambda_weight: float = 0.9):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.lambda_weight = lambda_weight
        self.js_loss = JSDistanceLoss(ignore_index, lambda_weight)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model.
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: JS distance loss.
        """
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        mask = (labels != self.ignore_index).int()
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        total_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_loss += self.js_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_loss / torch.sum(mask.view(-1), dim=0)


class TVDistanceWithChunkedOutputLoss(torch.nn.Module):
    """
    Total Variation distance with chunked outputs that saves memory by only upcasting one chunk at a time.

    Args:
        num_output_chunks (int): Number of chunks to chunk the output into.
            Default: 8
        ignore_index (int): Specifies a target value that is ignored.
            Default: -100
    """
    def __init__(self, num_output_chunks: int = 8, ignore_index: int = -100):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.tv_loss = TVDistanceLoss(ignore_index)

    def forward(
        self,
        student_logits: List[torch.Tensor],
        teacher_logits: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_logits (List[torch.Tensor]): List of chunked logits from student model.
            teacher_logits (List[torch.Tensor]): List of chunked logits from teacher model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: TV distance loss.
        """
        teacher_logits = [
            teacher_logits_chunk.reshape(-1, teacher_logits_chunk.size(-1))
            for teacher_logits_chunk in teacher_logits
        ]
        student_logits = [
            student_logits_chunk.reshape(-1, student_logits_chunk.size(-1))
            for student_logits_chunk in student_logits
        ]
        mask = (labels != self.ignore_index).int()
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]

        total_loss = 0.0
        for student_chunk, teacher_chunk, label_chunk in zip(
            student_logits, teacher_logits, labels
        ):
            total_loss += self.tv_loss(
                student_chunk, teacher_chunk, label_chunk, normalize=False
            )

        return total_loss / torch.sum(mask.view(-1), dim=0)

