import numpy as np
import torch

def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs, f"{x_shape[0]}, {t.shape}"
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

def denoising_step(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   sampling_type='ddim',
                   eta=0.0,
                   learn_sigma=False,
                   index=None,
                   t_edit=0,
                   hs_coeff=(1.0),
                   delta_h=None,
                   use_mask=False,
                   dt_lambda=1,
                   ignore_timestep=False,
                   image_space_noise=0,
                   dt_end = 999,
                   warigari=False,
                   ):

    # Compute noise and variance
    model = models

    et, et_modified, delta_h, middle_h = model(
        xt, 
        t, 
        index = index, 
        t_edit = t_edit, 
        hs_coeff = hs_coeff, 
        delta_h = delta_h, 
        ignore_timestep = ignore_timestep, 
        use_mask = use_mask)

    if learn_sigma:
        et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
        if index is not None:
            et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
        logvar = logvar_learned
    else:
        logvar = extract(logvars, t, xt.shape)

    if type(image_space_noise) != int:
        if t[0] >= t_edit:
            index = 0
            if type(image_space_noise) == torch.nn.parameter.Parameter:
                et_modified = et + image_space_noise * hs_coeff[1]
            else:
                # print(type(image_space_noise))
                temb = models.module.get_temb(t)
                et_modified = et + image_space_noise(et, temb) * 0.01

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)
    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

    xt_next = torch.zeros_like(xt)
    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        noise = torch.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':
        if index is not None:
            x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
        else:
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        # Deterministic.
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    if dt_lambda != 1 and t[0] >= dt_end:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

    # Asyrp & DiffStyle
    if not warigari or index is None:
        return xt_next, x0_t, delta_h, middle_h

    # Warigari by young-hyun, Not in the paper
    else:
        # will be updated
        return xt_next, x0_t, delta_h, middle_h

def denoising_step_edit(xt, t, t_next, *,
                   models,
                   logvars,
                   b,
                   sampling_type='ddim',
                   eta=0.0,
                   learn_sigma=False,
                   index=None,
                   t_edit=0,
                   hs_coeff=(1.0),
                   delta_h=None,
                   use_mask=False,
                   dt_lambda=1,
                   ignore_timestep=False,
                   image_space_noise=0,
                   dt_end = 999,
                   warigari=False,
                   ):

    # Compute noise and variance
    model = models

    et, et_modified, delta_h, middle_h = model(
        xt, 
        t, 
        index = index, 
        t_edit = t_edit, 
        hs_coeff = hs_coeff, 
        delta_h = delta_h, 
        ignore_timestep = ignore_timestep, 
        use_mask = use_mask)

    if learn_sigma:
        et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
        if index is not None:
            et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
        logvar = logvar_learned
    else:
        logvar = extract(logvars, t, xt.shape)

    if type(image_space_noise) != int:
        if t[0] >= t_edit:
            index = 0
            if type(image_space_noise) == torch.nn.parameter.Parameter:
                et_modified = et + image_space_noise * hs_coeff[1]
            else:
                # print(type(image_space_noise))
                temb = models.module.get_temb(t)
                et_modified = et + image_space_noise(et, temb) * 0.01

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)
    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

    xt_next = torch.zeros_like(xt)
    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        noise = torch.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    ###########
    # Original
    ###########
    # elif sampling_type == 'ddim':
    #     if index is not None:
    #         x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
    #     else:
    #         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

    #     # Deterministic.
    #     if eta == 0:
    #         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    #     # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
    #     else:
    #         c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
    #         c2 = ((1 - at_next) - c1 ** 2).sqrt()
    #         xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    #########
    # Modified v1
    #########
    # elif sampling_type == 'ddim':
    #     # Compute Fidelity Content explicitly (original latent reconstruction)
    #     x0_fidelity = (xt - et * (1 - at).sqrt()) / at.sqrt()

    #     if index is not None:
    #         # Compute explicit Semantic Shift (edit direction)
    #         semantic_shift = (et_modified - et) * (1 - at).sqrt() / at.sqrt()

    #         # Combine fidelity and semantic explicitly with a tunable factor
    #         lambda_sem = 0.25  # Adjust this parameter for edit strength
    #         x0_t = x0_fidelity + lambda_sem * semantic_shift
    #     else:
    #         # If no explicit semantic edits, use pure fidelity reconstruction
    #         x0_t = x0_fidelity

    #######
    # Modified v2
    #######
    # elif sampling_type == 'ddim':
    #     # Compute Fidelity Content explicitly (original latent reconstruction)
    #     x0_fidelity = (xt - et * (1 - at).sqrt()) / at.sqrt()

    #     if index is not None:
    #         # Compute explicit Semantic Shift (edit direction)
    #         semantic_shift = (et_modified - et) * (1 - at).sqrt() / at.sqrt()

    #         # Normalize semantic shift to avoid excessive magnitudes
    #         semantic_shift_norm = semantic_shift.norm(p=2, dim=[1,2,3], keepdim=True)
    #         semantic_shift = semantic_shift / (semantic_shift_norm + 1e-6)

    #         # Explicitly scale the semantic shift to preserve fidelity
    #         lambda_sem = 0.05  # Start small, then tune upwards carefully
    #         x0_t = x0_fidelity + lambda_sem * semantic_shift * x0_fidelity.norm(p=2, dim=[1,2,3], keepdim=True)
    #     else:
    #         # Pure fidelity reconstruction if no edits
    #         x0_t = x0_fidelity

    #     # Deterministic DDIM update with fidelity-semantic balance
    #     if eta == 0:
    #         xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
    #     else:
    #         c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    #         c2 = ((1 - at_next) - c1 ** 2).sqrt()
    #         xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)
    
    #########
    # Modified v3 SLERP DDIM
    #########
    elif sampling_type == 'ddim':
        # Fidelity content (original identity latent reconstruction)
        x0_fidelity = (xt - et * (1 - at).sqrt()) / at.sqrt()

        if index is not None:
            # Fully semantic edited latent
            x0_semantic = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()

            # Geodesic interpolation (SLERP) for manifold-aware blending
            alpha = 0.1

            def slerp(a, b, alpha):
                shape = a.shape
                a_flat = a.view(shape[0], -1)
                b_flat = b.view(shape[0], -1)

                dot = (a_flat * b_flat).sum(dim=-1, keepdim=True)
                dot = torch.clamp(dot, -0.9995, 0.9995)  # stability
                theta = torch.acos(dot)
                sin_theta = torch.sin(theta)

                weight_a = torch.sin((1 - alpha) * theta) / (sin_theta + 1e-6)
                weight_b = torch.sin(alpha * theta) / (sin_theta + 1e-6)

                return (weight_a * a_flat + weight_b * b_flat).view(shape)


            ## LDAE: https://github.com/GabrieleLozupone/LDAE/blob/6becf9aee482ae5c9ad419444e6457cb10cf4117/scripts/interpolation_evaluation.py
            # def slerp2(a, b, alpha):
            #     dot = torch.dot(a.view(-1), b.view(-1)) / (torch.norm(a) * torch.norm(b))
            #     theta = torch.arccos(dot)
            #     sin_theta = torch.sin(theta)
            #     return a * torch.sin((1.0 - alpha) * theta) / sin_theta + b * torch.sin(alpha * theta) / sin_theta

            ## ISODiff: https://github.com/isno0907/isodiff/blob/master/ddpm_interpolation.py
            # def slerp3(t, v0, v1, DOT_THRESHOLD=0.9995):
            #     """helper function to spherically interpolate two arrays v1 v2"""
            #     inputs_are_torch = isinstance(v0, torch.Tensor)

            #     if inputs_are_torch:
            #         input_device = v0.device
            #         v0 = v0.cpu().numpy()
            #         v1 = v1.cpu().numpy()
            #         t = t.cpu().numpy()

            #     dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))

            #     if np.abs(dot) > DOT_THRESHOLD:
            #         v2 = (1 - t) * v0 + t * v1
            #     else:
            #         theta_0 = np.arccos(dot)
            #         sin_theta_0 = np.sin(theta_0)
            #         theta_t = theta_0 * t
            #         sin_theta_t = np.sin(theta_t)
            #         s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            #         s1 = sin_theta_t / sin_theta_0
            #         v2 = s0 * v0 + s1 * v1

            #     if inputs_are_torch:
            #         v2 = torch.from_numpy(v2).to(input_device)

            #     return v2


            x0_t = slerp(x0_fidelity, x0_semantic, alpha)

        else:
            # Pure fidelity reconstruction if no edits
            x0_t = x0_fidelity

        # Deterministic DDIM update
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        else:
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    #################################################

    if dt_lambda != 1 and t[0] >= dt_end:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

    # Asyrp & DiffStyle
    if not warigari or index is None:
        return xt_next, x0_t, delta_h, middle_h

    # Warigari by young-hyun, Not in the paper
    else:
        # will be updated
        return xt_next, x0_t, delta_h, middle_h




# def consistency_fast_update(xt, fidelity_latent, orth_semantic_latent, alpha):
#     """
#     Explicitly leverage consistency to ensure fast, manifold-consistent updates.
#     """
#     # Combine latents explicitly with consistency assumption
#     x0_combined = fidelity_latent + alpha * orth_semantic_latent

#     # Apply consistency condition explicitly:
#     # A single step enforcing consistency across time (simplified Euler-inspired update clearly)
#     return xt + (x0_combined - xt).detach()  # Ensures stable gradient explicitly

# def orthogonalize(u, v):
#     '''
#     Orthogonalize semantic latent explicitly w.r.t fidelity latent
#     '''
#     proj = (u * v).sum(dim=[1,2,3], keepdim=True) / (v.norm(dim=[1,2,3], keepdim=True)**2 + 1e-6)
#     return u - proj * v

# def CLO_denoising_step(xt, t, t_next, models, b, eta=0.0, 
#                        index=None, t_edit=0, hs_coeff=(1.0,),
#                        delta_h=None, ignore_timestep=False, alpha=0.3):
    
#     et, et_modified, delta_h, middle_h = models(
#         xt, t, index=index, t_edit=t_edit,
#         hs_coeff=hs_coeff, delta_h=delta_h,
#         ignore_timestep=ignore_timestep
#     )

#     at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)
#     at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

#     # Explicit fidelity reconstruction (identity-preserving latent)
#     x0_fidelity = (xt - et * (1 - at).sqrt()) / at.sqrt()

#     # Fully semantic edited latent
#     x0_semantic = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()

#     x0_semantic_orth = orthogonalize(x0_semantic - x0_fidelity, x0_fidelity)

#     # Explicit consistency-based "fast step" (using theoretical grounding)
#     x0_t_consistent = consistency_fast_update(xt, x0_fidelity, x0_semantic_orth, alpha)

#     # Explicit single-step DDIM (CLO) update
#     xt_next = at_next.sqrt() * x0_t_consistent + (1 - at_next).sqrt() * et

#     return xt_next, x0_t_consistent, delta_h, middle_h