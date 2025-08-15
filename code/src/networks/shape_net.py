import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from ..engine.embedders import get_embedder,ProgressiveHashGrid

def implicit_normalizer(x):
    return (x+7.0)/14.0

def compute_aabb(points: torch.Tensor):    
    min_point, _ = points.min(dim=0)
    max_point, _ = points.max(dim=0)
    return min_point, max_point

def assert_in_0_1(tensor: torch.Tensor):
    assert torch.all((tensor >= 0) & (tensor <= 1)), "Tensor contains values outside [0, 1]"

class ImplicitNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()

        if opt.nettype == "object":
            opt.dims=[64,64]
            self.feat_size=64
        else :
            self.feat_size=opt.feature_vector_size
        dims = [opt.d_in] + list(opt.dims) + [opt.d_out + self.feat_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embedder_obj = None
        self.opt = opt
        self.body_specs = body_specs

        if opt.nettype == "object":
            self.skip_in=[]
            self.obj=True
        else :
            self.obj=False

        if opt.multires > 0:
            if opt.nettype == "object":
                embedder_obj=ProgressiveHashGrid(
                    input_dims=3,
                    log2_hashmap_size=19,
                    base_resolution=16,
                    n_levels=16,
                    n_features_per_level=2,
                    max_resolution=2048,
                    start_level=4,
                )
                input_ch=embedder_obj.out_dim
            else :
                embedder_obj, input_ch = get_embedder(
                    opt.multires,
                    input_dims=opt.d_in,
                    mode=body_specs.embedding,
                    barf_s=args.barf_s,
                    barf_e=args.barf_e,
                    no_barf=args.no_barf,
                )
            self.embedder_obj = embedder_obj
            dims[0] = input_ch

        self.cond = opt.cond
        if self.cond == "pose":
            self.cond_layer = [0]
            self.cond_dim = body_specs.pose_dim
        elif self.cond == "frame":
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding
        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if self.cond != "none" and l in self.cond_layer:
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)
            if opt.init == "geometry":
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
            if opt.init == "zero":
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond, current_epoch=None):
        if input.ndim == 2:
            input = input.unsqueeze(0)
        num_batch, num_point, num_dim = input.shape

        if num_batch * num_point == 0:
            return input

        input = input.reshape(num_batch * num_point, num_dim)

        if self.cond != "none" and self.obj==False :
            num_batch, num_cond = cond[self.cond].shape
            try:
                input_cond = (
                    cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)
                )
            except:
                import pdb

                pdb.set_trace()
            if num_cond == 45:
                # no pose dependent for MANO
                input_cond = input_cond * 0.0

            input_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.embedder_obj is not None:
            input = self.embedder_obj.embed(input)
            input=input.to(torch.float32)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.obj==False:
                if self.cond != "none" and l in self.cond_layer:
                    x = torch.cat([x, input_cond], dim=-1)
                if l in self.skip_in:
                    x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)

        x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)
