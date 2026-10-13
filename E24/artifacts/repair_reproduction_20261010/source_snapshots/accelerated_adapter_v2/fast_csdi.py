"""Sample-independent parallel DDPM; 50 original reverse steps are retained."""
import torch


def sample(core, observed_data, cond_mask, side_info, n_samples, chunk_size=5,
           initial_noise=None, step_noise=None):
    if core.is_unconditional:
        raise ValueError('parallel sampler is for the pinned conditional CSDI')
    batch, features, length = observed_data.shape
    result = observed_data.new_empty((batch, n_samples, features, length))
    for first in range(0, n_samples, chunk_size):
        count = min(chunk_size, n_samples-first)
        data = observed_data.repeat(count, 1, 1)
        mask = cond_mask.repeat(count, 1, 1)
        side = side_info.repeat(count, 1, 1, 1)
        current = (torch.randn_like(data) if initial_noise is None
                   else initial_noise[first:first+count].reshape_as(data))
        condition = (mask * data).unsqueeze(1)
        for step in range(core.num_steps-1, -1, -1):
            inputs = torch.cat([condition, ((1-mask) * current).unsqueeze(1)], dim=1)
            prediction = core.diffmodel(inputs, side, torch.tensor([step], device=data.device))
            coeff1 = 1 / core.alpha_hat[step] ** .5
            coeff2 = (1-core.alpha_hat[step]) / (1-core.alpha[step]) ** .5
            current = coeff1 * (current - coeff2 * prediction)
            if step > 0:
                noise = (torch.randn_like(current) if step_noise is None
                         else step_noise[step, first:first+count].reshape_as(current))
                sigma = ((1-core.alpha[step-1]) / (1-core.alpha[step]) * core.beta[step]) ** .5
                current = current + sigma * noise
        result[:, first:first+count] = current.reshape(count, batch, features, length).permute(1, 0, 2, 3)
    return result
