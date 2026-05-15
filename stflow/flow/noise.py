import torch


class PriorSampler:
    def __init__(self, prior_sample_type, device="cpu", **kwargs):
        self.prior_sample_type = prior_sample_type
        self.device = device

        if prior_sample_type == "gaussian":
            self.prior_sampler = lambda shape: gaussian_prior(shape, device=device)
        elif prior_sample_type == "zero":
            self.prior_sampler = lambda shape: all_zeros(shape, device=device)
        elif prior_sample_type == "zinb":
            # https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py#L433
            from scvi.distributions import ZeroInflatedNegativeBinomial

            # scvi-tools draws on CPU; we move once at the end of sample().
            prior_sampler = ZeroInflatedNegativeBinomial(
                                        total_count=kwargs.get("total_count", None),
                                        logits=kwargs.get("logits", None),
                                        zi_logits=kwargs.get("zi_logits", None),
                                    )
            self.prior_sampler = lambda shape: prior_sampler.sample(shape).squeeze(-1)
        else:
            raise ValueError("Invalid prior sample type")

    def sample(self, shape):
        s = self.prior_sampler(shape)
        if s.device != torch.device(self.device):
            s = s.to(self.device, non_blocking=True)
        return s


def gaussian_prior(shape, device="cpu"):
    return torch.randn(shape, device=device)


def all_zeros(shape, device="cpu"):
    return torch.zeros(shape, device=device)
