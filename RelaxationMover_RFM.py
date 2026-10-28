import pyrfm
import torch
import time
from dataclasses import dataclass

pou_list = {
    'PsiA': pyrfm.PsiA,
    'PsiB': pyrfm.PsiB,
}

rf_list = {
    'tanh': pyrfm.RFTanH,
    'tanh3': pyrfm.RFTanH3
}

@dataclass
class RFMConfig:
    n_samples_inner : int = 10000     # internal
    n_samples_boundary : int = 1000     # boundary
    n_samples_interface : int = 1000     # interface
    corner1 : torch.Tensor = None # corner points with dirichlet condition
    corner2 : torch.Tensor = None # corner points with zero gradient
    inner_lambda : float = 1.0 # weight for inner residual
    boundary_lambda : float = 1.0 # weight for boundary residual
    interface_lambda : float = 1.0 # weight for interface residual (only for PsiA)
    corner_lambda : float = 1.0 # weight for corner residual
    n_hidden : int = 1000       # hidden dimension
    n_subdomains : int = 1  # subdivisions per dimension
    gamma : float = 2.0 # random feature gamma parameter
    pou : str = 'PsiA'  # partition of unity type, PsiA or PsiB
    rf : str = 'tanh' # random feature class, tanh / tanh3
    damp_value : float = None # damping value
    seed : int = 100 # random feature seed
    use_transnet : bool = True # False keeps the default pyRFM weights and biases

@dataclass
class IterationConfig:
    maxiter : int = 200
    rtol : float = 1e-5
    pseudo_dt : float = 0.1
    adaptive_dt : bool = True # use False for fixed-step experiments
    early_stop_num : int = 10
    verbose : bool = True
    min_pseudo_dt : float = 1e-5
    max_pseudo_dt : float = 1.0
    min_eigenvalue : float = 1e-5

class RelaxationMover_RFM:
    def __init__(self, domain: pyrfm.GeometryBase, cfg: RFMConfig):
        self.domain = domain

        self.build_model(cfg)
        if cfg.use_transnet:
            self.adjust_model(cfg)

        start = time.perf_counter()
        self.build_features(cfg)
        self.preprocess(cfg)
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize(self.model.device)
        self.build_time = time.perf_counter() - start

    def build_model(self, cfg: RFMConfig):
        self.model = pyrfm.RFMBase(dim=2, n_hidden=cfg.n_hidden, domain=self.domain, n_subdomains=cfg.n_subdomains, pou=pou_list[cfg.pou], rf=rf_list[cfg.rf], seed=cfg.seed, dtype=self.domain.dtype, device=self.domain.device)
        self.n_hiddens = cfg.n_hidden * self.model.submodels.numel()
        self.model.W = torch.zeros((self.n_hiddens, 1), dtype=self.model.dtype, device=self.model.device)

    def adjust_model(self, cfg: RFMConfig):
        # To TransNet
        for submodel in self.model.submodels.flat_data:
            submodel.weights = torch.randn((submodel.dim, submodel.n_hidden), generator=submodel.gen, dtype=submodel.dtype, device=submodel.device)
            submodel.weights = submodel.weights / torch.norm(submodel.weights, dim=0, keepdim=True)
            submodel.weights = submodel.weights * cfg.gamma

            submodel.biases = torch.rand((1, submodel.n_hidden), generator=submodel.gen, dtype=submodel.dtype, device=submodel.device)
            submodel.biases = submodel.biases * cfg.gamma
            submodel.empty_cache()

    def build_features(self, cfg: RFMConfig):
        self.x_in = self.domain.in_sample(cfg.n_samples_inner, with_boundary=True).to(dtype=self.model.dtype, device=self.model.device)
        self.A_in_x = self.model.features_derivative(self.x_in, axis=0).cat(dim=1)
        self.A_in_y = self.model.features_derivative(self.x_in, axis=1).cat(dim=1)
        self.A_in_xx = self.model.features_second_derivative(self.x_in, axis1=0, axis2=0).cat(dim=1)
        self.A_in_xy = self.model.features_second_derivative(self.x_in, axis1=0, axis2=1).cat(dim=1)
        self.A_in_yy = self.model.features_second_derivative(self.x_in, axis1=1, axis2=1).cat(dim=1)

        (x_on, x_on_normal) = self.domain.on_sample(cfg.n_samples_boundary, with_normal=True, separate=False)
        x_on = x_on.to(self.x_in)
        x_on_normal = x_on_normal.to(self.x_in)
        A_on_x = self.model.features_derivative(x_on, axis=0).cat(dim=1)
        A_on_y = self.model.features_derivative(x_on, axis=1).cat(dim=1)
        self.n_samples_boundary = x_on.shape[0]

        A_list = [[self.A_in_xx + self.A_in_yy],
                  [A_on_x*x_on_normal[:, [0]] + A_on_y*x_on_normal[:, [1]]]]
        
        self.penalty = torch.cat([self.x_in.new_full((self.A_in_x.shape[0], 1), cfg.inner_lambda),
                                  self.x_in.new_full((A_on_x.shape[0], 1), cfg.boundary_lambda)])
        
        if cfg.corner1 is not None:
            self.x_corner1 = cfg.corner1.to(dtype=self.model.dtype, device=self.model.device)
            A_corner1 = self.model.features(self.x_corner1).cat(dim=1)
            A_list.append([A_corner1])
            self.penalty = torch.cat([self.penalty, self.x_in.new_full((A_corner1.shape[0], 1), cfg.corner_lambda)], dim=0)

        if cfg.corner2 is not None:
            self.x_corner2 = cfg.corner2.to(dtype=self.model.dtype, device=self.model.device)
            A_corner2_x = self.model.features_derivative(self.x_corner2, axis=0).cat(dim=1)
            A_corner2_y = self.model.features_derivative(self.x_corner2, axis=1).cat(dim=1)
            A_list.append([A_corner2_x])
            self.penalty = torch.cat([self.penalty, self.x_in.new_full((A_corner2_x.shape[0], 1), cfg.corner_lambda)], dim=0)
            A_list.append([A_corner2_y])
            self.penalty = torch.cat([self.penalty, self.x_in.new_full((A_corner2_y.shape[0], 1), cfg.corner_lambda)], dim=0)
        
        if cfg.pou == 'PsiA' and cfg.n_subdomains > 1:
            A_interface = self.model.add_c_condition(cfg.n_samples_interface).cat(dim=1)
            self.n_samples_interface = A_interface.shape[0]
            A_list.append([A_interface])
            self.penalty = torch.cat([self.penalty, self.x_in.new_full((A_interface.shape[0], 1), cfg.interface_lambda)], dim=0)

        self.A = pyrfm.concat_blocks(A_list)

    def preprocess(self, cfg: RFMConfig):
        self.damp = 0.0
        if cfg.damp_value is not None:
            self.damp = cfg.damp_value

        self.model.compute(self.A, penalty=self.penalty, damp=self.damp)
    
    def calculate_minimal_eigenvalue(self):
        a = 1 + self.phi_in_xx
        c = 1 + self.phi_in_yy
        b = self.phi_in_xy
        tr = a + c
        gap = torch.sqrt((a - c)**2 + 4 * b**2)
        self.min_eigenvalue = (0.5 * (tr - gap)).min()
    
    def evaluate(self, iter):
        phi_in_x = self.A_in_x @ self.model.W
        phi_in_y = self.A_in_y @ self.model.W
        self.phi_in_xx = self.A_in_xx @ self.model.W
        self.phi_in_xy = self.A_in_xy @ self.model.W
        self.phi_in_yy = self.A_in_yy @ self.model.W

        # calculate minimal eigenvalue of (I + H) to check convexity
        self.calculate_minimal_eigenvalue()

        # update physical coordinates
        self.x_in_new = self.x_in + torch.cat([phi_in_x, phi_in_y], dim=1)
            
        # evaluate monitor function on new mesh
        self.monitor = self.monitor_func(self.x_in_new).reshape(-1, 1)
        #self.monitor = self.monitor_func(self.x_in).reshape(-1, 1)

        # evaluate theta and residual
        self.jacobian_det = (1 + self.phi_in_xx) * (1 + self.phi_in_yy) - self.phi_in_xy**2
        self.theta_val = self.monitor * self.jacobian_det
        self.theta_mean = self.theta_val.mean()
        self.residual_CV = torch.std(self.theta_val, unbiased=False) / torch.abs(self.theta_mean).clamp_min(1e-12)

        if self.verbose:
            print(f"Iteration {iter}: "
                f"Min_eigen_value = {self.min_eigenvalue:.6f}, "
                f"Residual_CV = {self.residual_CV:.8e}, "
                f"pseudo_dt = {self.pseudo_dt:.10f}")

    def backup(self):
        self.W_backup = self.model.W.clone()
        self.x_in_new_backup = self.x_in_new.clone()
        self.min_eigenvalue_backup = self.min_eigenvalue.clone()

    def rollback(self, iter):
        self.model.W = self.W_backup.clone()
        self.evaluate(iter)

    def solve(self):
        # evaluate phi^{k+1}
        f_in = - (self.theta_val - self.theta_mean) * self.pseudo_dt + self.phi_in_xx + self.phi_in_yy
        f_on = self.x_in.new_zeros((self.n_samples_boundary, 1))

        B_list = [[f_in], [f_on]]

        if hasattr(self, "x_corner1"):
            B_list.append([self.x_in.new_zeros((self.x_corner1.shape[0], 1))])

        if hasattr(self, "x_corner2"): 
            B_list.append([self.x_in.new_zeros((self.x_corner2.shape[0], 1))])
            B_list.append([self.x_in.new_zeros((self.x_corner2.shape[0], 1))])

        if hasattr(self, "n_samples_interface"):
            B_list.append([self.x_in.new_zeros((self.n_samples_interface, 1))])

        B = pyrfm.concat_blocks(B_list)
        self.model.solve(B, verbose=False)
            
    @torch.no_grad()
    def relaxation(self, monitor_func, cfg: IterationConfig):
        self.monitor_func = monitor_func
        self.pseudo_dt = cfg.pseudo_dt
        self.verbose = cfg.verbose
        self.model.W = self.x_in.new_zeros((self.n_hiddens, 1))
        best_W = self.model.W.clone()

        if self.model.device.type == 'cuda':
            torch.cuda.synchronize(self.model.device)
        start = time.perf_counter()

        self.evaluate(0)

        self.best_CV_iteration = 0
        self.CV_residual_record = []
        self.CV_residual_record.append(self.residual_CV.cpu().item())   
        self.best_CV_residual = self.residual_CV.clone()

        self.best_relative_move_rate_iteration = 0
        self.best_relative_move_rate = 1e30
        self.relative_move_rate_record = []
        self.relative_move_rate_record.append(None)

        self.pseudo_dt_record = []
        self.pseudo_dt_record.append(self.pseudo_dt)

        self.stop_reason = 'maxiter'
        early_stop_count = 0
        for i in range(1, cfg.maxiter + 1):
            self.backup()
            self.solve()
            self.evaluate(i)

            self.pseudo_dt_record.append(self.pseudo_dt)

            self.CV_residual_record.append(self.residual_CV.cpu().item())

            relative_move = (torch.linalg.vector_norm(self.x_in_new - self.x_in_new_backup) / torch.linalg.vector_norm(self.x_in_new_backup)).item()
            relative_move_rate = relative_move / self.pseudo_dt
            self.relative_move_rate_record.append(relative_move_rate)

            bad_step = cfg.adaptive_dt and (self.theta_mean < 0 or self.min_eigenvalue < 0.5 * self.min_eigenvalue_backup or self.min_eigenvalue < cfg.min_eigenvalue)
            if bad_step:
                self.rollback(i)
                self.pseudo_dt *= 0.5
                if self.pseudo_dt < cfg.min_pseudo_dt:
                    self.stop_reason = 'min_pseudo_dt'
                    break
                continue

            if self.residual_CV < self.best_CV_residual:
                self.best_CV_residual = self.residual_CV.clone()
                self.best_CV_iteration = i

            if self.relative_move_rate_record[-1] < self.best_relative_move_rate:
                self.best_relative_move_rate = self.relative_move_rate_record[-1]
                best_W = self.model.W.clone()
                self.best_relative_move_rate_iteration = i
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= cfg.early_stop_num:
                self.stop_reason = 'relative_move_rate_stagnation'
                break

            grow_dt = cfg.adaptive_dt and \
                      (len(self.pseudo_dt_record) > 3 and self.pseudo_dt_record[-1] == self.pseudo_dt_record[-2] == self.pseudo_dt_record[-3]) and \
                      self.min_eigenvalue > 0.9 * self.min_eigenvalue_backup and \
                      self.relative_move_rate_record[-1] > 0.8 * self.relative_move_rate_record[-2] and \
                      self.relative_move_rate_record[-1] > 1e-3
            if grow_dt:
                self.pseudo_dt = min(self.pseudo_dt * 1.1, cfg.max_pseudo_dt)

            if self.relative_move_rate_record[-1] < cfg.rtol:
                self.stop_reason = 'relative_move_rate'
                break

        self.model.W = best_W
        if self.model.device.type == 'cuda':
            torch.cuda.synchronize(self.model.device)
        self.run_time = time.perf_counter() - start
        print(f"Stopped ({self.stop_reason}), best CV: ({self.best_CV_iteration}, {self.best_CV_residual:.8e}), best relative move rate: ({self.best_relative_move_rate_iteration}, {self.best_relative_move_rate:.8e}), run time: {self.run_time:.2f}s")

    @torch.no_grad()
    def move(self, V):
        V = V.to(dtype=self.model.dtype, device=self.model.device)
        dx = (self.model.features_derivative(V, axis=0).cat(dim=1) @ self.model.W)
        dy = (self.model.features_derivative(V, axis=1).cat(dim=1) @ self.model.W)
        new_V = V.clone()
        new_V[:, 0] = new_V[:, 0] + dx[:, 0]
        new_V[:, 1] = new_V[:, 1] + dy[:, 0]
        return new_V
