from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ProductKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor, nn
from torch.nn import functional as F


LOGGER = logging.getLogger(__name__)


def _ensure_2d(x: Tensor, *, name: str) -> Tensor:
    """Return a two-dimensional tensor view."""
    if x.ndim == 1:
        return x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [N] or [N, D], received {tuple(x.shape)}.")
    return x


def _ensure_1d(x: Tensor, *, name: str) -> Tensor:
    """Return a one-dimensional tensor view."""
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.squeeze(-1)
    if x.ndim != 1:
        raise ValueError(f"{name} must have shape [N] or [N, 1], received {tuple(x.shape)}.")
    return x


def _to_log_variance(variance: Tensor, floor: float) -> Tensor:
    """Convert variances into numerically stable log-variance features."""
    return torch.log(torch.clamp(variance, min=floor))


def _probability_to_logit(probability: Tensor, floor: float) -> Tensor:
    """Convert probabilities into stable logits."""
    clipped = torch.clamp(probability, min=floor, max=1.0 - floor)
    return torch.log(clipped) - torch.log1p(-clipped)


def _diagonal_gaussian_samples(
    mean: Tensor,
    variance: Tensor,
    *,
    sample_count: int,
    floor: float,
    stochastic: bool,
) -> Tensor:
    """Draw deterministic or stochastic samples from a diagonal Gaussian.

    Args:
        mean: Mean tensor with shape `[N]` or `[N, H]`.
        variance: Variance tensor with the same shape as `mean`.
        sample_count: Number of pathwise samples.
        floor: Numerical variance floor.
        stochastic: Whether to use random reparameterized samples.

    Returns:
        Samples with shape `[S, N]` or `[S, N, H]`.
    """
    safe_variance = torch.clamp(variance, min=floor)
    safe_std = torch.sqrt(safe_variance)
    if sample_count <= 1:
        return mean.unsqueeze(0)
    expand_shape = (sample_count,) + mean.shape
    if stochastic:
        noise = torch.randn(expand_shape, device=mean.device, dtype=mean.dtype)
    else:
        quantiles = torch.linspace(
            0.5 / float(sample_count),
            1.0 - (0.5 / float(sample_count)),
            sample_count,
            device=mean.device,
            dtype=mean.dtype,
        )
        base_noise = math.sqrt(2.0) * torch.erfinv((2.0 * quantiles) - 1.0)
        view_shape = (sample_count,) + (1,) * mean.ndim
        noise = base_noise.view(view_shape).expand(expand_shape)
    return mean.unsqueeze(0) + noise * safe_std.unsqueeze(0)


@dataclass
class SparseGPConfig:
    """Configuration for the sparse spatiotemporal GP state model."""

    input_dim: int
    inducing_points: int = 128
    spatial_dims: tuple[int, ...] = ()
    temporal_dims: tuple[int, ...] = ()
    spatial_kernel: str = "matern32"
    temporal_kernel: str = "rbf"
    mean_init: float = 0.0
    initial_lengthscale: float = 1.0
    initial_outputscale: float = 1.0
    initial_noise: float = 1e-2
    jitter: float = 1e-5
    variance_floor: float = 1e-6

    def __post_init__(self) -> None:
        self.spatial_dims = tuple(int(dim) for dim in self.spatial_dims)
        self.temporal_dims = tuple(int(dim) for dim in self.temporal_dims)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.inducing_points <= 0:
            raise ValueError("inducing_points must be positive.")
        if len(set(self.spatial_dims)) != len(self.spatial_dims):
            raise ValueError("spatial_dims must not contain duplicates.")
        if len(set(self.temporal_dims)) != len(self.temporal_dims):
            raise ValueError("temporal_dims must not contain duplicates.")
        overlap = set(self.spatial_dims).intersection(self.temporal_dims)
        if overlap:
            raise ValueError(f"spatial_dims and temporal_dims must be disjoint, overlap={sorted(overlap)}.")
        for dim in self.spatial_dims + self.temporal_dims:
            if dim < 0 or dim >= self.input_dim:
                raise ValueError(
                    f"Kernel active dimension {dim} is out of bounds for input_dim={self.input_dim}."
                )
        if self.input_dim > 2 and not self.spatial_dims and not self.temporal_dims:
            raise ValueError(
                "For input_dim > 2, explicitly set spatial_dims and temporal_dims to avoid "
                "mixing encoded temporal features into the spatial kernel."
            )


@dataclass
class ObservationModelConfig:
    """Configuration for dynamic silence detection."""

    threshold_mode: str = "stddev"
    diagnosis_mode: str = "pointwise"
    use_pi_ssd: bool = True
    use_dbn: bool = True
    stddev_multiplier: float = 2.5
    quantile_level: float = 0.95
    hybrid_floor: float = 0.5
    variance_floor: float = 1e-6
    threshold_floor: float = 1e-4
    context_dim: int = 0
    continuous_metadata_dim: int = 0
    num_sensor_types: int = 0
    num_sensor_groups: int = 0
    num_sensor_modalities: int = 0
    num_installation_environments: int = 0
    num_maintenance_states: int = 0
    link_fit_steps: int = 25
    link_learning_rate: float = 5e-2
    link_weight_decay: float = 0.0
    use_uncertainty_feature: bool = True
    use_observation_noise: bool = True
    require_calibration_split_for_quantile: bool = True
    temporal_smoothing: float = 0.6
    nwp_context_index: int | None = None
    nwp_anchor_weight: float = 0.5
    diagnosis_embedding_dim: int = 8
    diagnosis_hidden_dim: int = 32
    dbn_num_states: int = 4
    dbn_transition_smoothing: float = 0.8
    self_supervised_weight: float = 0.1
    physics_consistency_weight: float = 0.1
    masking_probability: float = 0.15
    corruption_probability_start: float = 0.05
    corruption_probability_end: float = 0.2
    corruption_noise_std: float = 0.05
    use_latent_ode: bool = True
    latent_ode_hidden_dim: int = 32
    latent_ode_weight: float = 0.05
    use_fault_head: bool = False
    fault_head_hidden_dim: int = 32
    fault_self_supervised_weight: float = 0.2
    fault_corruption_probability: float = 0.2
    fault_score_state_weight: float = 0.5
    fault_score_embedding_weight: float = 0.05
    fault_score_temporal_weight: float = 0.35
    fault_score_persistence_weight: float = 0.2
    fault_score_probability_weight: float = 0.8
    fault_target_false_alarm_rate: float = 0.05

    def __post_init__(self) -> None:
        valid_modes = {"stddev", "quantile", "hybrid"}
        valid_diagnosis_modes = {"pointwise", "temporal", "temporal_nwp"}
        if self.threshold_mode not in valid_modes:
            raise ValueError(f"threshold_mode must be one of {sorted(valid_modes)}.")
        if self.diagnosis_mode not in valid_diagnosis_modes:
            raise ValueError(f"diagnosis_mode must be one of {sorted(valid_diagnosis_modes)}.")
        if not 0.0 < self.quantile_level < 1.0:
            raise ValueError("quantile_level must lie in (0, 1).")
        if self.link_fit_steps < 0:
            raise ValueError("link_fit_steps must be non-negative.")
        if self.link_learning_rate <= 0.0:
            raise ValueError("link_learning_rate must be positive.")
        if not 0.0 <= self.temporal_smoothing < 1.0:
            raise ValueError("temporal_smoothing must lie in [0, 1).")
        if self.nwp_context_index is not None and self.nwp_context_index < 0:
            raise ValueError("nwp_context_index must be non-negative when provided.")
        if self.nwp_anchor_weight < 0.0:
            raise ValueError("nwp_anchor_weight must be non-negative.")
        if self.diagnosis_embedding_dim <= 0:
            raise ValueError("diagnosis_embedding_dim must be positive.")
        if self.diagnosis_hidden_dim <= 0:
            raise ValueError("diagnosis_hidden_dim must be positive.")
        if self.dbn_num_states <= 1:
            raise ValueError("dbn_num_states must be greater than 1.")
        if not 0.0 <= self.dbn_transition_smoothing <= 1.0:
            raise ValueError("dbn_transition_smoothing must lie in [0, 1].")
        if self.self_supervised_weight < 0.0:
            raise ValueError("self_supervised_weight must be non-negative.")
        if self.physics_consistency_weight < 0.0:
            raise ValueError("physics_consistency_weight must be non-negative.")
        if not 0.0 <= self.masking_probability < 1.0:
            raise ValueError("masking_probability must lie in [0, 1).")
        if not 0.0 <= self.corruption_probability_start < 1.0:
            raise ValueError("corruption_probability_start must lie in [0, 1).")
        if not 0.0 <= self.corruption_probability_end < 1.0:
            raise ValueError("corruption_probability_end must lie in [0, 1).")
        if self.corruption_probability_start > self.corruption_probability_end:
            raise ValueError("corruption_probability_start must not exceed corruption_probability_end.")
        if self.corruption_noise_std < 0.0:
            raise ValueError("corruption_noise_std must be non-negative.")
        if self.latent_ode_hidden_dim <= 0:
            raise ValueError("latent_ode_hidden_dim must be positive.")
        if self.latent_ode_weight < 0.0:
            raise ValueError("latent_ode_weight must be non-negative.")
        if self.fault_head_hidden_dim <= 0:
            raise ValueError("fault_head_hidden_dim must be positive.")
        if self.fault_self_supervised_weight < 0.0:
            raise ValueError("fault_self_supervised_weight must be non-negative.")
        if not 0.0 <= self.fault_corruption_probability < 1.0:
            raise ValueError("fault_corruption_probability must lie in [0, 1).")
        if self.fault_score_state_weight < 0.0:
            raise ValueError("fault_score_state_weight must be non-negative.")
        if self.fault_score_embedding_weight < 0.0:
            raise ValueError("fault_score_embedding_weight must be non-negative.")
        if self.fault_score_temporal_weight < 0.0:
            raise ValueError("fault_score_temporal_weight must be non-negative.")
        if self.fault_score_persistence_weight < 0.0:
            raise ValueError("fault_score_persistence_weight must be non-negative.")
        if self.fault_score_probability_weight < 0.0:
            raise ValueError("fault_score_probability_weight must be non-negative.")
        if not 0.0 < self.fault_target_false_alarm_rate < 1.0:
            raise ValueError("fault_target_false_alarm_rate must lie in (0, 1).")


@dataclass
class MissingMechanismConfig:
    """Configuration for structural missingness modeling."""

    mode: str = "selection"
    assumption: str = "sensor_conditional"
    inference_strategy: str = "plug_in"
    context_dim: int = 0
    x_dim: int = 0
    continuous_metadata_dim: int = 0
    hidden_dim: int = 64
    trunk_depth: int = 2
    encoder_hidden_dim: int = 64
    encoder_depth: int = 2
    sensor_embedding_dim: int = 8
    num_sensor_types: int = 0
    num_sensor_groups: int = 0
    num_sensor_modalities: int = 0
    num_installation_environments: int = 0
    num_maintenance_states: int = 0
    include_x: bool = False
    include_m: bool = False
    include_s: bool = True
    include_dynamic_residual: bool = True
    include_dynamic_threshold: bool = True
    include_normalized_residual: bool = True
    include_dynamic_feature_availability: bool = True
    use_dynamic_features_for_training: bool = True
    use_dynamic_features_for_prediction: bool = True
    use_dynamic_features_for_policy: bool = False
    use_group_heads: bool = True
    use_observed_y_in_variational_encoder: bool = True
    use_observation_mask_in_variational_encoder: bool = True
    use_sensor_health_latent: bool = True
    health_latent_dim: int = 4
    variance_floor: float = 1e-6
    probability_floor: float = 1e-4
    positive_class_weight: float = 1.0
    reconstruction_weight: float = 1.0
    kl_weight: float = 1e-3
    health_kl_weight: float = 1e-3
    health_reconstruction_weight: float = 0.1
    variational_logvar_clip: float = 4.0
    generative_samples: int = 4
    use_temporal_transition_prior: bool = True
    transition_time_index: int | None = None
    transition_group_key: str = "sensor_instance"
    transition_hidden_dim: int = 32
    transition_weight: float = 5e-2
    health_transition_weight: float = 1e-2

    def __post_init__(self) -> None:
        valid_modes = {"selection", "pattern_mixture"}
        valid_assumptions = {"homogeneous", "sensor_conditional"}
        valid_inference_strategies = {"plug_in", "joint_variational", "joint_generative"}
        valid_transition_group_keys = {
            "sensor_instance",
            "sensor_group",
            "sensor_type",
            "sensor_modality",
            "installation_environment",
            "maintenance_state",
            "global",
        }
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}.")
        if self.assumption not in valid_assumptions:
            raise ValueError(f"assumption must be one of {sorted(valid_assumptions)}.")
        if self.inference_strategy not in valid_inference_strategies:
            raise ValueError(
                f"inference_strategy must be one of {sorted(valid_inference_strategies)}."
            )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.trunk_depth <= 0:
            raise ValueError("trunk_depth must be positive.")
        if self.encoder_hidden_dim <= 0:
            raise ValueError("encoder_hidden_dim must be positive.")
        if self.encoder_depth <= 0:
            raise ValueError("encoder_depth must be positive.")
        if self.health_latent_dim <= 0:
            raise ValueError("health_latent_dim must be positive.")
        if self.positive_class_weight <= 0.0:
            raise ValueError("positive_class_weight must be positive.")
        if self.reconstruction_weight < 0.0:
            raise ValueError("reconstruction_weight must be non-negative.")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative.")
        if self.health_kl_weight < 0.0:
            raise ValueError("health_kl_weight must be non-negative.")
        if self.health_reconstruction_weight < 0.0:
            raise ValueError("health_reconstruction_weight must be non-negative.")
        if self.variational_logvar_clip <= 0.0:
            raise ValueError("variational_logvar_clip must be positive.")
        if self.generative_samples <= 0:
            raise ValueError("generative_samples must be positive.")
        if self.transition_group_key not in valid_transition_group_keys:
            raise ValueError(
                f"transition_group_key must be one of {sorted(valid_transition_group_keys)}."
            )
        if self.transition_hidden_dim <= 0:
            raise ValueError("transition_hidden_dim must be positive.")
        if self.transition_weight < 0.0:
            raise ValueError("transition_weight must be non-negative.")
        if self.health_transition_weight < 0.0:
            raise ValueError("health_transition_weight must be non-negative.")
        if self.transition_time_index is not None and self.transition_time_index < 0:
            raise ValueError("transition_time_index must be non-negative when provided.")


@dataclass
class SensorMetadataBatch:
    """Categorical and continuous sensor metadata for heterogeneous missingness."""

    sensor_instance: Tensor | None = None
    sensor_type: Tensor | None = None
    sensor_group: Tensor | None = None
    sensor_modality: Tensor | None = None
    installation_environment: Tensor | None = None
    maintenance_state: Tensor | None = None
    continuous: Tensor | None = None

    @classmethod
    def from_input(
        cls,
        metadata: "SensorMetadataBatch | Mapping[str, Tensor] | None",
    ) -> "SensorMetadataBatch":
        """Build a metadata batch from either a mapping or an existing dataclass."""
        if metadata is None:
            return cls()
        if isinstance(metadata, cls):
            return metadata
        return cls(
            sensor_instance=metadata.get("sensor_instance"),
            sensor_type=metadata.get("sensor_type"),
            sensor_group=metadata.get("sensor_group"),
            sensor_modality=metadata.get("sensor_modality"),
            installation_environment=metadata.get("installation_environment"),
            maintenance_state=metadata.get("maintenance_state"),
            continuous=metadata.get("continuous"),
        )


class SpatiotemporalSparseGP(ApproximateGP):
    """Sparse variational spatiotemporal GP for latent field inference."""

    def __init__(self, config: SparseGPConfig, inducing_points: Tensor | None = None) -> None:
        resolved_inducing = inducing_points
        if resolved_inducing is None:
            resolved_inducing = self._default_inducing_points(config)
        resolved_inducing = _ensure_2d(resolved_inducing, name="inducing_points")
        if resolved_inducing.shape[-1] != config.input_dim:
            raise ValueError(
                "Inducing point dimensionality mismatch: "
                f"expected {config.input_dim}, received {resolved_inducing.shape[-1]}."
            )

        variational_distribution = CholeskyVariationalDistribution(
            resolved_inducing.shape[-2],
        )
        variational_strategy = VariationalStrategy(
            self,
            resolved_inducing,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.config = config

        spatial_dims, temporal_dims = self._resolve_active_dims(config)
        self.mean_module = ConstantMean()
        self.mean_module.initialize(constant=config.mean_init)
        self.covar_module = self._build_covariance_module(
            config=config,
            spatial_dims=spatial_dims,
            temporal_dims=temporal_dims,
        )

    @staticmethod
    def _default_inducing_points(config: SparseGPConfig) -> Tensor:
        """Create stable initial inducing points before data-driven reinitialization."""
        inducing = torch.zeros(config.inducing_points, config.input_dim)
        inducing[:, 0] = torch.linspace(-1.0, 1.0, config.inducing_points)
        if config.input_dim > 1:
            phase = torch.linspace(0.0, 2.0 * math.pi, config.inducing_points)
            for dim in range(1, config.input_dim):
                inducing[:, dim] = torch.sin(phase + dim)
        return inducing

    @staticmethod
    def _resolve_active_dims(config: SparseGPConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Resolve spatial and temporal dimensions for kernel factorization."""
        if config.spatial_dims or config.temporal_dims:
            return config.spatial_dims, config.temporal_dims
        if config.input_dim == 1:
            return (0,), ()
        if config.input_dim == 2:
            return (0, 1), ()
        raise ValueError(
            "For input_dim > 2, explicitly set spatial_dims and temporal_dims in SparseGPConfig."
        )

    @staticmethod
    def _build_kernel(kind: str, dims: tuple[int, ...], initial_lengthscale: float) -> gpytorch.kernels.Kernel:
        """Construct a kernel over the provided active dimensions."""
        if kind == "rbf":
            kernel = RBFKernel(ard_num_dims=len(dims), active_dims=dims)
        elif kind == "matern12":
            kernel = MaternKernel(nu=0.5, ard_num_dims=len(dims), active_dims=dims)
        elif kind == "matern52":
            kernel = MaternKernel(nu=2.5, ard_num_dims=len(dims), active_dims=dims)
        else:
            kernel = MaternKernel(nu=1.5, ard_num_dims=len(dims), active_dims=dims)
        kernel.lengthscale = initial_lengthscale
        return kernel

    def _build_covariance_module(
        self,
        *,
        config: SparseGPConfig,
        spatial_dims: tuple[int, ...],
        temporal_dims: tuple[int, ...],
    ) -> ScaleKernel:
        """Build a factored spatial-temporal covariance module."""
        factors: list[gpytorch.kernels.Kernel] = []
        if spatial_dims:
            factors.append(self._build_kernel(config.spatial_kernel, spatial_dims, config.initial_lengthscale))
        if temporal_dims:
            factors.append(self._build_kernel(config.temporal_kernel, temporal_dims, config.initial_lengthscale))
        if not factors:
            factors.append(self._build_kernel(config.spatial_kernel, (0,), config.initial_lengthscale))
        kernel = factors[0] if len(factors) == 1 else ProductKernel(*factors)
        scaled = ScaleKernel(kernel)
        scaled.outputscale = config.initial_outputscale
        return scaled

    def initialize_inducing_points(self, x: Tensor) -> None:
        """Reinitialize inducing points from observed inputs."""
        x = _ensure_2d(x, name="x")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"x must have shape [N, {self.config.input_dim}], received {tuple(x.shape)}."
            )
        if x.shape[0] < self.config.inducing_points:
            repeats = math.ceil(self.config.inducing_points / max(x.shape[0], 1))
            candidates = x.repeat((repeats, 1))
        else:
            stride = max(x.shape[0] // self.config.inducing_points, 1)
            candidates = x[::stride]
        inducing = candidates[: self.config.inducing_points].detach().clone()
        with torch.no_grad():
            self.variational_strategy.inducing_points.copy_(inducing)

    def build_likelihood(self) -> GaussianLikelihood:
        """Create a numerically stable Gaussian likelihood for the latent field."""
        likelihood = GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(self.config.variance_floor),
        )
        likelihood.noise = max(self.config.initial_noise, self.config.variance_floor)
        return likelihood

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Compute the sparse GP prior/posterior over latent states.

        Args:
            x: Input coordinates with shape `[N, D]`.

        Returns:
            Multivariate normal distribution over the latent field.
        """
        x = _ensure_2d(x, name="x")
        # x: [N, D]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def predict(
        self,
        x: Tensor,
        likelihood: Likelihood | None = None,
    ) -> MultivariateNormal:
        """Return the predictive distribution over latent or observed states.

        Args:
            x: Query coordinates with shape `[N, D]`.
            likelihood: Optional observation likelihood.

        Returns:
            Predictive multivariate normal distribution.
        """
        self.eval()
        x = _ensure_2d(x, name="x")
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
            self.config.jitter
        ):
            posterior = self(x)
            if likelihood is None:
                return posterior
            return likelihood(posterior)

    def posterior_summary(
        self,
        x: Tensor,
        likelihood: Likelihood | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return predictive mean and variance summaries.

        Args:
            x: Query coordinates with shape `[N, D]`.
            likelihood: Optional observation likelihood.

        Returns:
            Tuple of predictive mean and variance with shape `[N]`.
        """
        predictive = self.predict(x=x, likelihood=likelihood)
        mean = predictive.mean
        variance = torch.clamp(predictive.variance, min=self.config.variance_floor)
        return mean, variance


class DynamicObservationModel(nn.Module):
    """Observation model with uncertainty-aware and sequence-aware silence detection."""

    def __init__(self, config: ObservationModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObservationModelConfig()
        self.observation_scale = nn.Parameter(torch.tensor(1.0))
        self.observation_bias = nn.Parameter(torch.tensor(0.0))
        self.uncertainty_scale = nn.Parameter(torch.tensor(0.0))
        self.context_projection = (
            nn.Linear(self.config.context_dim, 1, bias=False) if self.config.context_dim > 0 else None
        )
        self.continuous_projection = (
            nn.Linear(self.config.continuous_metadata_dim, 1, bias=False)
            if self.config.continuous_metadata_dim > 0
            else None
        )
        self.sensor_type_bias = self._maybe_bias_embedding(self.config.num_sensor_types)
        self.sensor_group_bias = self._maybe_bias_embedding(self.config.num_sensor_groups)
        self.sensor_modality_bias = self._maybe_bias_embedding(self.config.num_sensor_modalities)
        self.installation_bias = self._maybe_bias_embedding(self.config.num_installation_environments)
        self.maintenance_bias = self._maybe_bias_embedding(self.config.num_maintenance_states)
        diagnosis_input_dim = 4 + self.config.context_dim
        if self.config.use_pi_ssd:
            self.diagnosis_encoder = nn.Sequential(
                nn.Linear(diagnosis_input_dim, self.config.diagnosis_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.diagnosis_hidden_dim, self.config.diagnosis_embedding_dim),
            )
            self.diagnosis_decoder = nn.Sequential(
                nn.Linear(self.config.diagnosis_embedding_dim, self.config.diagnosis_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.diagnosis_hidden_dim, diagnosis_input_dim),
            )
            if self.config.use_latent_ode:
                self.latent_dynamics = nn.Sequential(
                    nn.Linear(self.config.diagnosis_embedding_dim + 1, self.config.latent_ode_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.config.latent_ode_hidden_dim, self.config.diagnosis_embedding_dim),
                )
            else:
                self.latent_dynamics = None
        else:
            self.diagnosis_encoder = None
            self.diagnosis_decoder = None
            self.latent_dynamics = None
        if self.config.use_dbn:
            self.dbn_emission = nn.Sequential(
                nn.Linear(self.config.diagnosis_embedding_dim + 2, self.config.diagnosis_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.diagnosis_hidden_dim, self.config.dbn_num_states),
            )
            self.dbn_transition_logits = nn.Parameter(torch.zeros(self.config.dbn_num_states, self.config.dbn_num_states))
        else:
            self.dbn_emission = None
            self.dbn_transition_logits = None
        fault_input_dim = diagnosis_input_dim + self.config.diagnosis_embedding_dim + 1
        if self.config.use_fault_head:
            self.fault_head = nn.Sequential(
                nn.Linear(fault_input_dim, self.config.fault_head_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.fault_head_hidden_dim, 1),
            )
        else:
            self.fault_head = None
        self.register_buffer("calibrated_quantile", torch.tensor(float("nan")))

    @staticmethod
    def _maybe_bias_embedding(cardinality: int) -> nn.Embedding | None:
        """Create a scalar bias embedding when metadata cardinality is available."""
        if cardinality <= 0:
            return None
        return nn.Embedding(cardinality, 1)

    def _metadata_bias(
        self,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Assemble metadata-conditioned observation offsets."""
        metadata = SensorMetadataBatch.from_input(sensor_metadata)
        bias = torch.zeros(batch_size, device=device, dtype=dtype)
        specs = [
            (metadata.sensor_type, self.sensor_type_bias),
            (metadata.sensor_group, self.sensor_group_bias),
            (metadata.sensor_modality, self.sensor_modality_bias),
            (metadata.installation_environment, self.installation_bias),
            (metadata.maintenance_state, self.maintenance_bias),
        ]
        for tensor, embedding in specs:
            if tensor is None or embedding is None:
                continue
            index = _ensure_1d(tensor, name="observation_metadata").long().to(device=device)
            bias = bias + embedding(index).squeeze(-1).to(dtype=dtype)
        return bias

    def expected_observation(
        self,
        z_mean: Tensor,
        *,
        z_var: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Map latent predictions into the expected observation space.

        Args:
            z_mean: Latent predictive means with shape `[N]`.
            z_var: Optional latent predictive variances with shape `[N]`.
            context: Optional contextual features with shape `[N, C]`.
            sensor_metadata: Optional heterogeneous sensor metadata.

        Returns:
            Expected observations with shape `[N]`.
        """
        z_mean = _ensure_1d(z_mean, name="z_mean")
        expected = self.observation_scale * z_mean + self.observation_bias
        if self.config.use_uncertainty_feature and z_var is not None:
            safe_var = torch.clamp(_ensure_1d(z_var, name="z_var"), min=self.config.variance_floor)
            expected = expected + self.uncertainty_scale * torch.sqrt(safe_var)
        if context is not None and self.context_projection is not None:
            expected = expected + self.context_projection(_ensure_2d(context, name="context")).squeeze(-1)
        if (
            sensor_metadata is not None
            and SensorMetadataBatch.from_input(sensor_metadata).continuous is not None
            and self.continuous_projection is not None
        ):
            continuous = _ensure_2d(
                SensorMetadataBatch.from_input(sensor_metadata).continuous,
                name="sensor_metadata.continuous",
            ).to(device=z_mean.device, dtype=z_mean.dtype)
            expected = expected + self.continuous_projection(continuous).squeeze(-1)
        expected = expected + self._metadata_bias(
            sensor_metadata,
            batch_size=z_mean.shape[0],
            device=z_mean.device,
            dtype=z_mean.dtype,
        )
        return expected

    def compute_residual(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        *,
        z_var: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Compute absolute residuals for dynamic silence detection."""
        y_true = _ensure_1d(y_true, name="y_true")
        expected = self.expected_observation(
            z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
        )
        return torch.abs(y_true - expected)

    def fit_observation_link(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        *,
        z_var: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        steps: int | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, list[float]]:
        """Fit the observation-link parameters using robust regression.

        Args:
            y_true: Observed targets with shape `[N]`.
            z_mean: Latent predictive means with shape `[N]`.
            z_var: Optional latent predictive variances with shape `[N]`.
            context: Optional contextual features with shape `[N, C]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            steps: Optional optimization step override.
            learning_rate: Optional learning-rate override.

        Returns:
            Optimization history keyed by `"loss"`.
        """
        y_true = _ensure_1d(y_true, name="y_true")
        z_mean = _ensure_1d(z_mean, name="z_mean")
        observed_mask = torch.isfinite(y_true)
        history = {"loss": []}
        if self.config.use_pi_ssd:
            history["self_supervised_loss"] = []
            history["reconstruction_loss"] = []
            history["physics_loss"] = []
            history["latent_ode_loss"] = []
            history["curriculum_mask_probability"] = []
        if self.config.use_fault_head:
            history["fault_self_supervised_loss"] = []
            history["fault_probability_mean"] = []
            history["fault_corruption_rate"] = []
        if not observed_mask.any():
            return history
        active_steps = self.config.link_fit_steps if steps is None else steps
        if active_steps <= 0:
            return history

        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.link_learning_rate if learning_rate is None else learning_rate,
            weight_decay=self.config.link_weight_decay,
        )
        filtered_context = None if context is None else _ensure_2d(context, name="context")[observed_mask]
        filtered_metadata = SensorMetadataBatch.from_input(sensor_metadata)
        filtered_metadata = SensorMetadataBatch(
            sensor_type=None if filtered_metadata.sensor_type is None else filtered_metadata.sensor_type[observed_mask],
            sensor_group=None if filtered_metadata.sensor_group is None else filtered_metadata.sensor_group[observed_mask],
            sensor_modality=(
                None if filtered_metadata.sensor_modality is None else filtered_metadata.sensor_modality[observed_mask]
            ),
            installation_environment=(
                None
                if filtered_metadata.installation_environment is None
                else filtered_metadata.installation_environment[observed_mask]
            ),
            maintenance_state=(
                None if filtered_metadata.maintenance_state is None else filtered_metadata.maintenance_state[observed_mask]
            ),
            continuous=None if filtered_metadata.continuous is None else filtered_metadata.continuous[observed_mask],
        )
        filtered_var = None if z_var is None else _ensure_1d(z_var, name="z_var")[observed_mask]

        for step_index in range(active_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.expected_observation(
                z_mean[observed_mask],
                z_var=filtered_var,
                context=filtered_context,
                sensor_metadata=filtered_metadata,
            )
            loss = F.huber_loss(prediction, y_true[observed_mask], reduction="mean")
            if self.config.use_pi_ssd:
                progress = 1.0 if active_steps == 1 else float(step_index) / float(active_steps - 1)
                ssd_components = self.diagnosis_self_supervision_components(
                    y_true=y_true[observed_mask],
                    z_mean=z_mean[observed_mask],
                    z_var=filtered_var if filtered_var is not None else torch.ones_like(z_mean[observed_mask]),
                    context=filtered_context,
                    sensor_metadata=filtered_metadata,
                    training_progress=progress,
                )
                ssd_loss = ssd_components["total_loss"]
                loss = loss + ssd_loss
                history["self_supervised_loss"].append(float(ssd_loss.item()))
                history["reconstruction_loss"].append(float(ssd_components["reconstruction_loss"].item()))
                history["physics_loss"].append(float(ssd_components["physics_loss"].item()))
                history["latent_ode_loss"].append(float(ssd_components["latent_ode_loss"].item()))
                history["curriculum_mask_probability"].append(float(ssd_components["mask_probability"].item()))
            if self.config.use_fault_head:
                fault_components = self.fault_self_supervision_components(
                    y_true=y_true[observed_mask],
                    z_mean=z_mean[observed_mask],
                    z_var=filtered_var if filtered_var is not None else torch.ones_like(z_mean[observed_mask]),
                    context=filtered_context,
                    sensor_metadata=filtered_metadata,
                )
                fault_loss = self.config.fault_self_supervised_weight * fault_components["fault_loss"]
                loss = loss + fault_loss
                history["fault_self_supervised_loss"].append(float(fault_components["fault_loss"].item()))
                history["fault_probability_mean"].append(float(fault_components["fault_probability_mean"].item()))
                history["fault_corruption_rate"].append(float(fault_components["fault_corruption_rate"].item()))
            loss.backward()
            optimizer.step()
            history["loss"].append(float(loss.item()))
        self.eval()
        return history

    def calibrate(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        *,
        z_var: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Calibrate the quantile threshold from residuals."""
        residuals = self.compute_residual(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
        )
        quantile = torch.quantile(residuals, self.config.quantile_level)
        self.calibrated_quantile.copy_(quantile.detach())
        return quantile

    def _compute_threshold(
        self,
        z_var: Tensor,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Compute an uncertainty-aware residual threshold."""
        z_var = _ensure_1d(z_var, name="z_var")
        predictive_std = torch.sqrt(torch.clamp(z_var, min=self.config.variance_floor))
        stddev_threshold = self.config.stddev_multiplier * predictive_std

        if self.config.threshold_mode == "stddev":
            return torch.clamp(stddev_threshold, min=self.config.threshold_floor)

        if self.config.threshold_mode == "quantile":
            if calibration_residuals is not None:
                threshold = torch.quantile(
                    _ensure_1d(calibration_residuals, name="calibration_residuals"),
                    self.config.quantile_level,
                )
            elif not torch.isnan(self.calibrated_quantile):
                threshold = self.calibrated_quantile
            else:
                raise ValueError(
                    "Quantile thresholding requires calibration residuals or a prior calibrate() call."
                )
            return torch.full_like(stddev_threshold, torch.clamp(threshold, min=self.config.threshold_floor))

        floor = torch.full_like(stddev_threshold, max(self.config.hybrid_floor, self.config.threshold_floor))
        return torch.maximum(stddev_threshold, floor)

    def _smooth_sequence(self, values: Tensor) -> Tensor:
        """Apply one-pass exponential smoothing over the current batch order."""
        values = _ensure_1d(values, name="values")
        if values.numel() <= 1 or self.config.diagnosis_mode == "pointwise":
            return values
        alpha = self.config.temporal_smoothing
        smoothed = torch.empty_like(values)
        smoothed[0] = values[0]
        for index in range(1, values.numel()):
            smoothed[index] = alpha * smoothed[index - 1] + (1.0 - alpha) * values[index]
        return smoothed

    def _nwp_gradient_mismatch(
        self,
        y_true: Tensor,
        *,
        context: Tensor | None = None,
    ) -> Tensor:
        """Measure disagreement between observation and anchor-sequence gradients."""
        y_true = _ensure_1d(y_true, name="y_true")
        mismatch = torch.zeros_like(y_true)
        if (
            context is None
            or self.config.nwp_context_index is None
            or self.config.nwp_context_index >= context.shape[-1]
        ):
            return mismatch
        anchor = _ensure_2d(context, name="context")[:, self.config.nwp_context_index]
        observed_grad = torch.zeros_like(y_true)
        anchor_grad = torch.zeros_like(anchor)
        observed_grad[1:] = y_true[1:] - y_true[:-1]
        anchor_grad[1:] = anchor[1:] - anchor[:-1]
        mismatch = torch.abs(observed_grad - anchor_grad)
        return self._smooth_sequence(mismatch)

    def _diagnosis_feature_stack(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Assemble sequence-diagnostic features with shape `[N, F]`."""
        residuals = self.compute_residual(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
        )
        threshold = self._compute_threshold(z_var=z_var, calibration_residuals=calibration_residuals)
        base_score = residuals / torch.clamp(threshold, min=self.config.threshold_floor)
        mismatch = self._nwp_gradient_mismatch(y_true=y_true, context=context)
        features = [
            residuals.unsqueeze(-1),
            threshold.unsqueeze(-1),
            base_score.unsqueeze(-1),
            mismatch.unsqueeze(-1),
        ]
        if context is not None:
            features.append(_ensure_2d(context, name="context").to(device=y_true.device, dtype=y_true.dtype))
        elif self.config.context_dim > 0:
            features.append(torch.zeros(y_true.shape[0], self.config.context_dim, device=y_true.device, dtype=y_true.dtype))
        return torch.cat(features, dim=-1)

    def diagnosis_embedding(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Compute PI-SSD-style diagnosis embeddings with shape `[N, E]`."""
        features = self._diagnosis_feature_stack(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        if self.diagnosis_encoder is None:
            width = min(features.shape[-1], self.config.diagnosis_embedding_dim)
            if width == self.config.diagnosis_embedding_dim:
                return features[:, :width]
            return torch.cat(
                [
                    features[:, :width],
                    torch.zeros(
                        features.shape[0],
                        self.config.diagnosis_embedding_dim - width,
                        device=features.device,
                        dtype=features.dtype,
                    ),
                ],
                dim=-1,
            )
        return self.diagnosis_encoder(features)

    def _fault_head_input(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Assemble self-supervised fault features with shape `[N, F]`."""
        features = self._diagnosis_feature_stack(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        embedding = self.diagnosis_embedding(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        if self.config.use_dbn:
            with torch.no_grad():
                state_probs = self.infer_sensor_state_probs(
                    y_true=y_true,
                    z_mean=z_mean,
                    z_var=z_var,
                    context=context,
                    sensor_metadata=sensor_metadata,
                    calibration_residuals=calibration_residuals,
                )
            normal_probability = state_probs[:, :1]
            state_deviation = 1.0 - normal_probability
        else:
            state_deviation = torch.zeros(
                y_true.shape[0],
                1,
                device=y_true.device,
                dtype=y_true.dtype,
            )
        return torch.cat([features, embedding, state_deviation], dim=-1)

    def _sample_fault_corruption(
        self,
        y_true: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Generate synthetic corruptions for self-supervised fault learning."""
        y_true = _ensure_1d(y_true, name="y_true")
        if y_true.numel() == 0:
            return y_true, torch.zeros_like(y_true, dtype=torch.bool)
        scale = torch.clamp(y_true.std(unbiased=False), min=1.0)
        corrupted = y_true.clone()
        fault_mask = torch.rand_like(y_true) < self.config.fault_corruption_probability
        if not fault_mask.any():
            return corrupted, fault_mask
        scenario_draw = torch.randint(0, 4, (y_true.shape[0],), device=y_true.device)
        spike_mask = fault_mask & (scenario_draw == 0)
        freeze_mask = fault_mask & (scenario_draw == 1)
        drift_mask = fault_mask & (scenario_draw == 2)
        dropout_mask = fault_mask & (scenario_draw == 3)

        if spike_mask.any():
            signs = torch.where(
                torch.rand(int(spike_mask.sum().item()), device=y_true.device) > 0.5,
                1.0,
                -1.0,
            )
            corrupted[spike_mask] = corrupted[spike_mask] + signs * 3.5 * scale
        if freeze_mask.any():
            shifted = torch.roll(y_true, shifts=1)
            shifted[0] = y_true[0]
            corrupted[freeze_mask] = shifted[freeze_mask]
        if drift_mask.any():
            ramp = torch.linspace(-1.0, 1.0, y_true.shape[0], device=y_true.device, dtype=y_true.dtype)
            corrupted[drift_mask] = corrupted[drift_mask] + 2.0 * scale * ramp[drift_mask]
        if dropout_mask.any():
            corrupted[dropout_mask] = y_true.mean() - 4.0 * scale
        return corrupted, fault_mask

    def fault_self_supervision_components(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return self-supervised fault-classification loss and probability stats."""
        if not self.config.use_fault_head or self.fault_head is None:
            zero = z_mean.new_tensor(0.0)
            return {
                "fault_loss": zero,
                "fault_probability_mean": zero,
                "fault_corruption_rate": zero,
            }
        corrupted_y, fault_mask = self._sample_fault_corruption(y_true)
        fault_input = self._fault_head_input(
            corrupted_y,
            z_mean,
            z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        logits = self.fault_head(fault_input).squeeze(-1)
        targets = fault_mask.float()
        positive_count = float(targets.sum().item())
        negative_count = float(targets.numel() - positive_count)
        pos_weight = (
            logits.new_tensor(max(negative_count / max(positive_count, 1.0), 1.0))
            if positive_count > 0.0
            else logits.new_tensor(1.0)
        )
        loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probability = torch.sigmoid(logits)
        return {
            "fault_loss": loss,
            "fault_probability_mean": probability.mean(),
            "fault_corruption_rate": targets.mean(),
        }

    def fault_probability(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Estimate fault probability with shape `[N]` from observation diagnostics."""
        if not self.config.use_fault_head or self.fault_head is None:
            threshold = self._compute_threshold(z_var=z_var, calibration_residuals=calibration_residuals)
            residual = self.compute_residual(
                y_true=y_true,
                z_mean=z_mean,
                z_var=z_var,
                context=context,
                sensor_metadata=sensor_metadata,
            )
            return torch.sigmoid((residual / torch.clamp(threshold, min=self.config.threshold_floor)) - 1.0)
        fault_input = self._fault_head_input(
            y_true,
            z_mean,
            z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        return torch.sigmoid(self.fault_head(fault_input).squeeze(-1))

    def _curriculum_mask_probability(self, training_progress: float | None = None) -> float:
        """Return the current corruption probability for PI-SSD curriculum training."""
        if training_progress is None:
            return float(self.config.masking_probability)
        progress = max(0.0, min(1.0, float(training_progress)))
        return (
            (1.0 - progress) * self.config.corruption_probability_start
            + progress * self.config.corruption_probability_end
        )

    def _corrupt_diagnosis_features(
        self,
        features: Tensor,
        *,
        training_progress: float | None = None,
    ) -> tuple[Tensor, Tensor, float]:
        """Apply curriculum masking and additive noise to diagnosis features `[N, F]`."""
        mask_probability = self._curriculum_mask_probability(training_progress)
        feature_mask = torch.rand_like(features) < mask_probability
        noise_scale = self.config.corruption_noise_std * (1.0 + mask_probability)
        noisy_features = features + noise_scale * torch.randn_like(features)
        corrupted = torch.where(feature_mask, torch.zeros_like(noisy_features), noisy_features)
        return corrupted, feature_mask, mask_probability

    def _latent_ode_regularization(
        self,
        features: Tensor,
        embedding: Tensor,
    ) -> Tensor:
        """Compute a latent ODE-lite next-step consistency penalty over `[N, E]` embeddings."""
        if (
            not self.config.use_latent_ode
            or self.latent_dynamics is None
            or embedding.shape[0] <= 1
            or self.diagnosis_encoder is None
        ):
            return embedding.new_tensor(0.0)
        with torch.no_grad():
            target_embedding = self.diagnosis_encoder(features)
        feature_delta = torch.norm(features[1:] - features[:-1], dim=-1, keepdim=True)
        dynamics_input = torch.cat([embedding[:-1], torch.log1p(feature_delta)], dim=-1)
        predicted_delta = self.latent_dynamics(dynamics_input)
        predicted_next = embedding[:-1] + predicted_delta
        return F.huber_loss(predicted_next, target_embedding[1:].detach(), reduction="mean")

    def diagnosis_self_supervision_components(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
        training_progress: float | None = None,
    ) -> dict[str, Tensor]:
        """Return PI-SSD reconstruction, physics, and latent-dynamics losses."""
        if not self.config.use_pi_ssd or self.diagnosis_encoder is None or self.diagnosis_decoder is None:
            zero = z_mean.new_tensor(0.0)
            return {
                "total_loss": zero,
                "reconstruction_loss": zero,
                "physics_loss": zero,
                "latent_ode_loss": zero,
                "mask_probability": zero,
            }
        features = self._diagnosis_feature_stack(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        corrupted, mask, mask_probability = self._corrupt_diagnosis_features(
            features,
            training_progress=training_progress,
        )
        embedding = self.diagnosis_encoder(corrupted)
        reconstructed = self.diagnosis_decoder(embedding)
        reconstruction_loss = F.mse_loss(
            reconstructed[mask],
            features[mask],
            reduction="mean",
        ) if mask.any() else F.mse_loss(reconstructed, features, reduction="mean")
        if embedding.shape[0] <= 1:
            physics_loss = embedding.new_tensor(0.0)
        else:
            embedding_delta = torch.norm(embedding[1:] - embedding[:-1], dim=-1)
            observed_grad = torch.abs(y_true[1:] - y_true[:-1])
            if (
                context is not None
                and self.config.nwp_context_index is not None
                and self.config.nwp_context_index < context.shape[-1]
            ):
                anchor = _ensure_2d(context, name="context")[:, self.config.nwp_context_index]
                anchor_grad = torch.abs(anchor[1:] - anchor[:-1])
                target_grad = torch.maximum(observed_grad, anchor_grad)
            else:
                target_grad = observed_grad
            target_grad = torch.log1p(target_grad)
            physics_loss = F.mse_loss(embedding_delta, target_grad, reduction="mean")
        latent_ode_loss = self._latent_ode_regularization(features, embedding)
        total_loss = (
            self.config.self_supervised_weight * reconstruction_loss
            + self.config.physics_consistency_weight * physics_loss
            + self.config.latent_ode_weight * latent_ode_loss
        )
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "physics_loss": physics_loss,
            "latent_ode_loss": latent_ode_loss,
            "mask_probability": features.new_tensor(mask_probability),
        }

    def self_supervised_diagnosis_loss(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
        training_progress: float | None = None,
    ) -> Tensor:
        """Compute a lightweight PI-SSD-style self-supervised objective."""
        components = self.diagnosis_self_supervision_components(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
            training_progress=training_progress,
        )
        return components["total_loss"]

    def infer_sensor_state_probs(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Infer DBN-style diagnostic state probabilities with shape `[N, S]`."""
        if not self.config.use_dbn or self.dbn_emission is None or self.dbn_transition_logits is None:
            score = self.diagnostic_score(
                y_true=y_true,
                z_mean=z_mean,
                z_var=z_var,
                context=context,
                sensor_metadata=sensor_metadata,
                calibration_residuals=calibration_residuals,
            )
            state_probs = torch.zeros(
                score.shape[0],
                self.config.dbn_num_states,
                device=score.device,
                dtype=score.dtype,
            )
            state_probs[:, 0] = (score <= 1.0).float()
            if self.config.dbn_num_states > 1:
                state_probs[:, 1] = (score > 1.0).float()
            if self.config.dbn_num_states > 2:
                state_probs[:, 2:] = 0.0
            normalizer = torch.clamp(state_probs.sum(dim=-1, keepdim=True), min=1.0)
            return state_probs / normalizer

        embedding = self.diagnosis_embedding(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        residuals = self.compute_residual(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
        )
        threshold = self._compute_threshold(z_var=z_var, calibration_residuals=calibration_residuals)
        emission_input = torch.cat(
            [
                embedding,
                residuals.unsqueeze(-1),
                threshold.unsqueeze(-1),
            ],
            dim=-1,
        )
        emission_probs = torch.softmax(self.dbn_emission(emission_input), dim=-1)
        transition = torch.softmax(self.dbn_transition_logits, dim=-1)
        filtered = torch.empty_like(emission_probs)
        filtered[0] = emission_probs[0]
        for index in range(1, emission_probs.shape[0]):
            predicted = filtered[index - 1] @ transition
            posterior = emission_probs[index] * torch.clamp(predicted, min=self.config.threshold_floor)
            posterior = posterior / torch.clamp(posterior.sum(), min=self.config.threshold_floor)
            filtered[index] = (
                self.config.dbn_transition_smoothing * posterior
                + (1.0 - self.config.dbn_transition_smoothing) * emission_probs[index]
            )
        return filtered / torch.clamp(filtered.sum(dim=-1, keepdim=True), min=self.config.threshold_floor)

    def infer_sensor_state_labels(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Return argmax DBN state labels with shape `[N]`."""
        return torch.argmax(
            self.infer_sensor_state_probs(
                y_true=y_true,
                z_mean=z_mean,
                z_var=z_var,
                context=context,
                sensor_metadata=sensor_metadata,
                calibration_residuals=calibration_residuals,
            ),
            dim=-1,
        )

    def diagnostic_score(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> Tensor:
        """Return a sequence-aware diagnostic score with shape `[N]`.

        The default `pointwise` mode reduces to the residual-to-threshold ratio.
        `temporal` adds exponential smoothing over the ordered batch, and
        `temporal_nwp` further augments the score with gradient disagreement
        against an anchor context channel such as ERA5/NWP guidance.
        """
        residuals = self.compute_residual(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
        )
        threshold = self._compute_threshold(z_var=z_var, calibration_residuals=calibration_residuals)
        base_score = residuals / torch.clamp(threshold, min=self.config.threshold_floor)
        if self.config.diagnosis_mode == "pointwise":
            score = base_score
        else:
            score = self._smooth_sequence(base_score)
            if self.config.diagnosis_mode == "temporal_nwp":
                mismatch = self._nwp_gradient_mismatch(y_true=y_true, context=context)
                score = score + self.config.nwp_anchor_weight * (
                    mismatch / torch.clamp(threshold, min=self.config.threshold_floor)
                )
        if self.config.use_pi_ssd:
            embedding = self.diagnosis_embedding(
                y_true=y_true,
                z_mean=z_mean,
                z_var=z_var,
                context=context,
                sensor_metadata=sensor_metadata,
                calibration_residuals=calibration_residuals,
            )
            score = score + 0.1 * torch.norm(embedding, dim=-1)
        return score

    def detect_dynamic_silence(
        self,
        y_true: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Detect dynamic silence events via uncertainty-aware residual testing.

        Args:
            y_true: Observed targets with shape `[N]`.
            z_mean: Latent predictive means with shape `[N]`.
            z_var: Latent predictive variances with shape `[N]`.
            calibration_residuals: Optional calibration residuals for quantile mode.

        Returns:
            Tuple of residuals, thresholds, and silence flags, each with shape `[N]`.
        """
        residuals = self.compute_residual(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
        )
        threshold = self._compute_threshold(z_var=z_var, calibration_residuals=calibration_residuals)
        score = self.diagnostic_score(
            y_true=y_true,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            calibration_residuals=calibration_residuals,
        )
        flags = score > 1.0
        return residuals, threshold, flags


class MissingMechanism(nn.Module):
    """Structural missingness model with scalable plug-in and joint latent variants.

    The baseline `plug_in` strategy conditions directly on sparse-GP posterior
    summaries from M1. `joint_variational` adds an amortized latent adapter on
    top of those summaries. `joint_generative` goes one step further and
    optimizes a sampled ELBO-style objective with observation reconstruction and
    missingness likelihood terms under reparameterized latent and sensor-health
    posteriors, while still anchoring the latent prior to the sparse GP.
    """

    def __init__(self, config: MissingMechanismConfig | None = None) -> None:
        super().__init__()
        self.config = config or MissingMechanismConfig()

        self.sensor_type_embedding = self._maybe_embedding(self.config.num_sensor_types)
        self.sensor_group_embedding = self._maybe_embedding(self.config.num_sensor_groups)
        self.sensor_modality_embedding = self._maybe_embedding(self.config.num_sensor_modalities)
        self.installation_embedding = self._maybe_embedding(self.config.num_installation_environments)
        self.maintenance_embedding = self._maybe_embedding(self.config.num_maintenance_states)

        metadata_dim = self._metadata_embedding_dim()
        scalar_metadata_dim = self.config.continuous_metadata_dim
        optional_dim = (
            int(self.config.include_m)
            + int(self.config.include_s)
            + int(self.config.include_dynamic_residual)
            + int(self.config.include_dynamic_threshold)
            + int(self.config.include_normalized_residual)
            + int(self.config.include_dynamic_feature_availability)
        )
        aux_dim = self.config.context_dim + scalar_metadata_dim + optional_dim
        if self.config.include_x:
            aux_dim += self.config.x_dim
        self.aux_feature_dim = aux_dim + metadata_dim

        health_feature_dim = self.config.health_latent_dim if self.config.use_sensor_health_latent else 0
        if self.config.mode == "selection":
            input_dim = 2 + health_feature_dim + self.aux_feature_dim
            self.selection_trunk = self._make_mlp(input_dim=input_dim, output_dim=self.config.hidden_dim)
            self.selection_head = nn.Linear(self.config.hidden_dim, 1)
            self.latent_branch = None
            self.pattern_branch = None
            self.fusion = None
        else:
            latent_input_dim = 2 + health_feature_dim
            pattern_input_dim = max(self.aux_feature_dim, 1)
            self.latent_branch = self._make_mlp(input_dim=latent_input_dim, output_dim=self.config.hidden_dim)
            self.pattern_branch = self._make_mlp(input_dim=pattern_input_dim, output_dim=self.config.hidden_dim)
            self.fusion = self._make_mlp(
                input_dim=3 * self.config.hidden_dim,
                output_dim=self.config.hidden_dim,
            )
            self.selection_trunk = None
            self.selection_head = nn.Linear(self.config.hidden_dim, 1)

        if (
            self.config.assumption == "sensor_conditional"
            and self.config.use_group_heads
            and self.config.num_sensor_groups > 0
        ):
            self.group_weight = nn.Embedding(self.config.num_sensor_groups, self.config.hidden_dim)
            self.group_bias = nn.Embedding(self.config.num_sensor_groups, 1)
        else:
            self.group_weight = None
            self.group_bias = None

        if self.config.inference_strategy in {"joint_variational", "joint_generative"}:
            encoder_input_dim = 2 + self.aux_feature_dim
            if self.config.use_observed_y_in_variational_encoder:
                encoder_input_dim += 1
            if self.config.use_observation_mask_in_variational_encoder:
                encoder_input_dim += 1
            self.variational_encoder = self._make_variational_mlp(
                input_dim=encoder_input_dim,
                output_dim=self.config.encoder_hidden_dim,
            )
            self.variational_head = nn.Linear(self.config.encoder_hidden_dim, 2)
            self.observation_decoder = self._make_variational_mlp(
                input_dim=2 + self.aux_feature_dim + health_feature_dim,
                output_dim=self.config.encoder_hidden_dim,
            )
            self.observation_head = nn.Linear(self.config.encoder_hidden_dim, 2)
        else:
            self.variational_encoder = None
            self.variational_head = None
            self.observation_decoder = None
            self.observation_head = None
        if self.config.use_sensor_health_latent:
            health_encoder_input_dim = 2 + self.aux_feature_dim
            if self.config.use_observed_y_in_variational_encoder:
                health_encoder_input_dim += 1
            if self.config.use_observation_mask_in_variational_encoder:
                health_encoder_input_dim += 1
            self.health_encoder = self._make_variational_mlp(
                input_dim=health_encoder_input_dim,
                output_dim=self.config.encoder_hidden_dim,
            )
            self.health_head = nn.Linear(self.config.encoder_hidden_dim, 2 * self.config.health_latent_dim)
            self.health_decoder = nn.Sequential(
                nn.Linear(self.config.health_latent_dim, self.config.encoder_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.encoder_hidden_dim, 2),
            )
        else:
            self.health_encoder = None
            self.health_head = None
            self.health_decoder = None
        if self.config.inference_strategy == "joint_generative" and self.config.use_temporal_transition_prior:
            transition_input_dim = 2 + health_feature_dim + self.aux_feature_dim
            self.latent_transition_model = nn.Sequential(
                nn.Linear(transition_input_dim, self.config.transition_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.transition_hidden_dim, self.config.transition_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.config.transition_hidden_dim, 2),
            )
            if self.config.use_sensor_health_latent:
                self.health_transition_model = nn.Sequential(
                    nn.Linear(transition_input_dim, self.config.transition_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.config.transition_hidden_dim, self.config.transition_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.config.transition_hidden_dim, 2 * self.config.health_latent_dim),
                )
            else:
                self.health_transition_model = None
        else:
            self.latent_transition_model = None
            self.health_transition_model = None

    def _maybe_embedding(self, cardinality: int) -> nn.Embedding | None:
        """Create an embedding only when categorical support is configured."""
        if self.config.assumption != "sensor_conditional" or cardinality <= 0:
            return None
        return nn.Embedding(cardinality, self.config.sensor_embedding_dim)

    def _metadata_embedding_dim(self) -> int:
        """Return the concatenated embedding dimension used by the sensor-aware variant."""
        if self.config.assumption != "sensor_conditional":
            return 0
        embeddings = [
            self.sensor_type_embedding,
            self.sensor_group_embedding,
            self.sensor_modality_embedding,
            self.installation_embedding,
            self.maintenance_embedding,
        ]
        width = 0
        for module in embeddings:
            width += self.config.sensor_embedding_dim if module is not None else 1
        return width

    def _make_mlp(self, *, input_dim: int, output_dim: int) -> nn.Sequential:
        """Build a small, stable MLP trunk."""
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(self.config.trunk_depth):
            layers.append(nn.Linear(in_features, self.config.hidden_dim))
            layers.append(nn.SiLU())
            in_features = self.config.hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def _make_variational_mlp(self, *, input_dim: int, output_dim: int) -> nn.Sequential:
        """Build the amortized encoder/decoder trunk for joint-style inference."""
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(self.config.encoder_depth):
            layers.append(nn.Linear(in_features, self.config.encoder_hidden_dim))
            layers.append(nn.SiLU())
            in_features = self.config.encoder_hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def _categorical_features(
        self,
        tensor: Tensor | None,
        embedding: nn.Embedding | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return either an embedding or a scalar fallback for categorical metadata."""
        width = self.config.sensor_embedding_dim if embedding is not None else 1
        if tensor is None:
            return torch.zeros(batch_size, width, device=device, dtype=dtype)
        values = _ensure_1d(tensor, name="categorical_metadata").to(device=device)
        if embedding is not None:
            return embedding(values.long())
        return values.float().unsqueeze(-1).to(dtype=dtype)

    def _assemble_auxiliary_features(
        self,
        *,
        batch_size: int,
        context: Tensor | None,
        M: Tensor | None,
        S: Tensor | None,
        dynamic_residual: Tensor | None,
        dynamic_threshold: Tensor | None,
        dynamic_feature_available: Tensor | None,
        X: Tensor | None,
        sensor_metadata: SensorMetadataBatch,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Assemble context and metadata features for missingness modeling."""
        features: list[Tensor] = []

        if context is not None:
            features.append(_ensure_2d(context, name="context").to(device=device, dtype=dtype))
        elif self.config.context_dim > 0:
            features.append(torch.zeros(batch_size, self.config.context_dim, device=device, dtype=dtype))

        if self.config.include_m:
            if M is None:
                features.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))
            else:
                features.append(_ensure_1d(M, name="M").to(device=device, dtype=dtype).unsqueeze(-1))

        if self.config.include_s:
            if S is None:
                features.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))
            else:
                features.append(_ensure_1d(S, name="S").to(device=device, dtype=dtype).unsqueeze(-1))

        if self.config.include_dynamic_residual:
            if dynamic_residual is None:
                features.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))
            else:
                features.append(
                    _ensure_1d(dynamic_residual, name="dynamic_residual")
                    .to(device=device, dtype=dtype)
                    .unsqueeze(-1)
                )

        if self.config.include_dynamic_threshold:
            if dynamic_threshold is None:
                features.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))
            else:
                features.append(
                    _ensure_1d(dynamic_threshold, name="dynamic_threshold")
                    .to(device=device, dtype=dtype)
                    .unsqueeze(-1)
                )

        if self.config.include_normalized_residual:
            if dynamic_residual is None or dynamic_threshold is None:
                features.append(torch.zeros(batch_size, 1, device=device, dtype=dtype))
            else:
                normalized_residual = _ensure_1d(dynamic_residual, name="dynamic_residual").to(
                    device=device,
                    dtype=dtype,
                ) / torch.clamp(
                    _ensure_1d(dynamic_threshold, name="dynamic_threshold").to(device=device, dtype=dtype),
                    min=self.config.variance_floor,
                )
                features.append(normalized_residual.unsqueeze(-1))

        if self.config.include_dynamic_feature_availability:
            if dynamic_feature_available is None:
                inferred = float(any(value is not None for value in (S, dynamic_residual, dynamic_threshold)))
                features.append(torch.full((batch_size, 1), inferred, device=device, dtype=dtype))
            else:
                features.append(
                    _ensure_1d(dynamic_feature_available, name="dynamic_feature_available")
                    .to(device=device, dtype=dtype)
                    .unsqueeze(-1)
                )

        if self.config.include_x:
            if X is None:
                features.append(torch.zeros(batch_size, self.config.x_dim, device=device, dtype=dtype))
            else:
                features.append(_ensure_2d(X, name="X").to(device=device, dtype=dtype))

        if sensor_metadata.continuous is not None:
            features.append(
                _ensure_2d(sensor_metadata.continuous, name="sensor_metadata.continuous").to(
                    device=device,
                    dtype=dtype,
                )
            )
        elif self.config.continuous_metadata_dim > 0:
            features.append(torch.zeros(batch_size, self.config.continuous_metadata_dim, device=device, dtype=dtype))

        if self.config.assumption == "sensor_conditional":
            features.extend(
                [
                    self._categorical_features(
                        sensor_metadata.sensor_type,
                        self.sensor_type_embedding,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    ),
                    self._categorical_features(
                        sensor_metadata.sensor_group,
                        self.sensor_group_embedding,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    ),
                    self._categorical_features(
                        sensor_metadata.sensor_modality,
                        self.sensor_modality_embedding,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    ),
                    self._categorical_features(
                        sensor_metadata.installation_environment,
                        self.installation_embedding,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    ),
                    self._categorical_features(
                        sensor_metadata.maintenance_state,
                        self.maintenance_embedding,
                        batch_size=batch_size,
                        device=device,
                        dtype=dtype,
                    ),
                ]
            )

        if not features:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        return torch.cat(features, dim=-1)

    def _resolve_observation_features(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        target_missing: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Resolve observed-value and availability features for the variational encoder."""
        if y is None:
            observed_y = torch.zeros(batch_size, device=device, dtype=dtype)
            finite_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            raw_y = _ensure_1d(y, name="y").to(device=device, dtype=dtype)
            finite_mask = torch.isfinite(raw_y)
            observed_y = torch.where(finite_mask, raw_y, torch.zeros_like(raw_y))

        if observation_available is not None:
            available = _ensure_1d(observation_available, name="observation_available").to(
                device=device,
                dtype=dtype,
            )
        elif target_missing is not None:
            available = 1.0 - _ensure_1d(target_missing, name="target_missing").to(
                device=device,
                dtype=dtype,
            )
        else:
            available = finite_mask.to(dtype=dtype)

        available = torch.clamp(available, min=0.0, max=1.0)
        observed_y = torch.where(available > 0.0, observed_y, torch.zeros_like(observed_y))
        return observed_y, available

    def _infer_latent_posterior_from_aux(
        self,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        aux_features: Tensor,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        target_missing: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Refine latent summaries with an amortized joint-style encoder."""
        safe_var = torch.clamp(_ensure_1d(z_var, name="z_var"), min=self.config.variance_floor)
        z_mean = _ensure_1d(z_mean, name="z_mean")
        if (
            self.config.inference_strategy not in {"joint_variational", "joint_generative"}
            or self.variational_encoder is None
        ):
            zero = z_mean.new_tensor(0.0)
            observed_y, available = self._resolve_observation_features(
                batch_size=z_mean.shape[0],
                device=z_mean.device,
                dtype=z_mean.dtype,
                y=y,
                observation_available=observation_available,
                target_missing=target_missing,
            )
            return z_mean, safe_var, {
                "kl_loss": zero,
                "reconstruction_loss": zero,
                "observed_y": observed_y,
                "observation_available": available,
            }

        observed_y, available = self._resolve_observation_features(
            batch_size=z_mean.shape[0],
            device=z_mean.device,
            dtype=z_mean.dtype,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        encoder_parts = [
            z_mean.unsqueeze(-1),
            _to_log_variance(safe_var, self.config.variance_floor).unsqueeze(-1),
        ]
        if self.config.use_observed_y_in_variational_encoder:
            encoder_parts.append(observed_y.unsqueeze(-1))
        if self.config.use_observation_mask_in_variational_encoder:
            encoder_parts.append(available.unsqueeze(-1))
        if aux_features.numel() > 0:
            encoder_parts.append(aux_features)
        encoder_input = torch.cat(encoder_parts, dim=-1)
        hidden = self.variational_encoder(encoder_input)
        delta_mean, delta_logvar = self.variational_head(hidden).unbind(dim=-1)
        delta_logvar = torch.clamp(
            delta_logvar,
            min=-self.config.variational_logvar_clip,
            max=self.config.variational_logvar_clip,
        )
        posterior_mean = z_mean + delta_mean
        posterior_var = torch.clamp(safe_var * torch.exp(delta_logvar), min=self.config.variance_floor)
        kl_terms = 0.5 * (
            torch.log(safe_var)
            - torch.log(posterior_var)
            + (posterior_var + (posterior_mean - z_mean).pow(2)) / safe_var
            - 1.0
        )
        return posterior_mean, posterior_var, {
            "kl_loss": kl_terms.mean(),
            "reconstruction_loss": z_mean.new_tensor(0.0),
            "observed_y": observed_y,
            "observation_available": available,
        }

    def infer_latent_posterior(
        self,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        X: Tensor | None = None,
        target_missing: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Infer a refined latent posterior used by the joint-style M3 path."""
        z_mean = _ensure_1d(z_mean, name="z_mean")
        z_var = _ensure_1d(z_var, name="z_var")
        metadata = SensorMetadataBatch.from_input(sensor_metadata)
        aux_features = self._assemble_auxiliary_features(
            batch_size=z_mean.shape[0],
            context=context,
            M=M,
            S=S,
            dynamic_residual=dynamic_residual,
            dynamic_threshold=dynamic_threshold,
            dynamic_feature_available=dynamic_feature_available,
            X=X,
            sensor_metadata=metadata,
            device=z_mean.device,
            dtype=z_mean.dtype,
        )
        latent_mean, latent_var, variational_stats = self._infer_latent_posterior_from_aux(
            z_mean,
            z_var,
            aux_features=aux_features,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        health_stats = self._infer_health_posterior_from_aux(
            z_mean=latent_mean,
            z_var=latent_var,
            aux_features=aux_features,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        merged_stats = dict(variational_stats)
        merged_stats.update(health_stats)
        return latent_mean, latent_var, merged_stats

    def _decode_observation_distribution(
        self,
        *,
        latent_location: Tensor,
        latent_variance: Tensor,
        aux_features: Tensor,
        health_latent: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Decode the observation-space Gaussian under the current latent state."""
        if self.observation_decoder is None or self.observation_head is None:
            raise RuntimeError("Observation decoder is unavailable for the current inference strategy.")
        decoder_parts = [
            latent_location.unsqueeze(-1),
            torch.sqrt(torch.clamp(latent_variance, min=self.config.variance_floor)).unsqueeze(-1),
        ]
        if self.config.use_sensor_health_latent:
            if health_latent is None:
                health_latent = torch.zeros(
                    latent_location.shape[0],
                    self.config.health_latent_dim,
                    device=latent_location.device,
                    dtype=latent_location.dtype,
                )
            decoder_parts.append(health_latent)
        if aux_features.numel() > 0:
            decoder_parts.append(aux_features)
        decoder_input = torch.cat(decoder_parts, dim=-1)
        hidden = self.observation_decoder(decoder_input)
        reconstruction_mean, reconstruction_scale = self.observation_head(hidden).unbind(dim=-1)
        reconstruction_var = torch.clamp(
            F.softplus(reconstruction_scale) + self.config.variance_floor,
            min=self.config.variance_floor,
        )
        return reconstruction_mean, reconstruction_var

    def _generative_sample_paths(
        self,
        *,
        latent_mean: Tensor,
        latent_var: Tensor,
        health_mean: Tensor,
        health_var: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample latent and health paths for the generative ELBO approximation."""
        sample_count = self.config.generative_samples
        stochastic = self.training
        latent_samples = _diagonal_gaussian_samples(
            latent_mean,
            latent_var,
            sample_count=sample_count,
            floor=self.config.variance_floor,
            stochastic=stochastic,
        )
        health_samples = _diagonal_gaussian_samples(
            health_mean,
            health_var,
            sample_count=sample_count,
            floor=self.config.variance_floor,
            stochastic=stochastic,
        )
        return latent_samples, health_samples

    def infer_health_posterior(
        self,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        X: Tensor | None = None,
        target_missing: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Infer latent sensor-health summaries with shape `[N, H]`.

        Args:
            z_mean: Posterior mean summaries with shape `[N]`.
            z_var: Posterior variance summaries with shape `[N]`.
            y: Optional observed targets with shape `[N]`.
            observation_available: Optional availability mask with shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence indicator with shape `[N]`.
            dynamic_residual: Optional residual feature with shape `[N]`.
            dynamic_threshold: Optional threshold feature with shape `[N]`.
            dynamic_feature_available: Optional availability flag with shape `[N]`.
            X: Optional original inputs with shape `[N, D]`.
            target_missing: Optional training-time missingness indicator with shape `[N]`.

        Returns:
            Dictionary containing `health_mean`, `health_var`, and auxiliary
            regularization losses used by the joint variational path.
        """
        z_mean = _ensure_1d(z_mean, name="z_mean")
        z_var = _ensure_1d(z_var, name="z_var")
        metadata = SensorMetadataBatch.from_input(sensor_metadata)
        aux_features = self._assemble_auxiliary_features(
            batch_size=z_mean.shape[0],
            context=context,
            M=M,
            S=S,
            dynamic_residual=dynamic_residual,
            dynamic_threshold=dynamic_threshold,
            dynamic_feature_available=dynamic_feature_available,
            X=X,
            sensor_metadata=metadata,
            device=z_mean.device,
            dtype=z_mean.dtype,
        )
        return self._infer_health_posterior_from_aux(
            z_mean=z_mean,
            z_var=z_var,
            aux_features=aux_features,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )

    def _reconstruction_loss(
        self,
        *,
        posterior_mean: Tensor,
        posterior_var: Tensor,
        aux_features: Tensor,
        observed_y: Tensor,
        observation_available: Tensor,
    ) -> Tensor:
        """Compute Gaussian reconstruction loss over available observations."""
        if self.observation_decoder is None or self.observation_head is None:
            return posterior_mean.new_tensor(0.0)
        observed_mask = observation_available > 0.0
        if not observed_mask.any():
            return posterior_mean.new_tensor(0.0)
        reconstruction_mean, reconstruction_var = self._decode_observation_distribution(
            latent_location=posterior_mean,
            latent_variance=posterior_var,
            aux_features=aux_features,
        )
        losses = 0.5 * (
            torch.log(reconstruction_var)
            + (observed_y - reconstruction_mean).pow(2) / reconstruction_var
        )
        return losses[observed_mask].mean()

    def _sampled_reconstruction_loss(
        self,
        *,
        latent_samples: Tensor,
        latent_var: Tensor,
        health_samples: Tensor,
        aux_features: Tensor,
        observed_y: Tensor,
        observation_available: Tensor,
    ) -> Tensor:
        """Approximate expected observation NLL under sampled latent paths."""
        if self.observation_decoder is None or self.observation_head is None:
            return latent_var.new_tensor(0.0)
        observed_mask = observation_available > 0.0
        if not observed_mask.any():
            return latent_var.new_tensor(0.0)
        sample_losses: list[Tensor] = []
        for sample_index in range(latent_samples.shape[0]):
            reconstruction_mean, reconstruction_var = self._decode_observation_distribution(
                latent_location=latent_samples[sample_index],
                latent_variance=latent_var,
                aux_features=aux_features,
                health_latent=health_samples[sample_index],
            )
            losses = 0.5 * (
                torch.log(reconstruction_var)
                + (observed_y - reconstruction_mean).pow(2) / reconstruction_var
            )
            sample_losses.append(losses[observed_mask].mean())
        return torch.stack(sample_losses).mean()

    def _infer_health_posterior_from_aux(
        self,
        *,
        z_mean: Tensor,
        z_var: Tensor,
        aux_features: Tensor,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        target_missing: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Infer latent sensor-health summaries used by the joint JVI path."""
        batch_size = z_mean.shape[0]
        if not self.config.use_sensor_health_latent or self.health_encoder is None or self.health_head is None:
            zero_health = z_mean.new_zeros(batch_size, self.config.health_latent_dim)
            zero_scalar = z_mean.new_tensor(0.0)
            return {
                "health_mean": zero_health,
                "health_var": torch.ones_like(zero_health),
                "health_kl_loss": zero_scalar,
                "health_reconstruction_loss": zero_scalar,
            }
        safe_var = torch.clamp(z_var, min=self.config.variance_floor)
        observed_y, available = self._resolve_observation_features(
            batch_size=batch_size,
            device=z_mean.device,
            dtype=z_mean.dtype,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        encoder_parts = [
            z_mean.unsqueeze(-1),
            _to_log_variance(safe_var, self.config.variance_floor).unsqueeze(-1),
        ]
        if self.config.use_observed_y_in_variational_encoder:
            encoder_parts.append(observed_y.unsqueeze(-1))
        if self.config.use_observation_mask_in_variational_encoder:
            encoder_parts.append(available.unsqueeze(-1))
        if aux_features.numel() > 0:
            encoder_parts.append(aux_features)
        health_hidden = self.health_encoder(torch.cat(encoder_parts, dim=-1))
        health_mean, health_logvar = torch.chunk(self.health_head(health_hidden), 2, dim=-1)
        health_logvar = torch.clamp(
            health_logvar,
            min=-self.config.variational_logvar_clip,
            max=self.config.variational_logvar_clip,
        )
        health_var = torch.exp(health_logvar)
        health_kl_terms = 0.5 * (health_var + health_mean.pow(2) - 1.0 - health_logvar)
        health_kl_loss = health_kl_terms.sum(dim=-1).mean()
        health_reconstruction_loss = z_mean.new_tensor(0.0)
        if self.health_decoder is not None:
            decoded_mean, decoded_scale = self.health_decoder(health_mean).unbind(dim=-1)
            decoded_var = torch.clamp(F.softplus(decoded_scale) + self.config.variance_floor, min=self.config.variance_floor)
            target_residual = torch.abs(observed_y - z_mean)
            observed_mask = available > 0.0
            if observed_mask.any():
                recon_losses = 0.5 * (
                    torch.log(decoded_var)
                    + (target_residual - decoded_mean).pow(2) / decoded_var
                )
                health_reconstruction_loss = recon_losses[observed_mask].mean()
        return {
            "health_mean": health_mean,
            "health_var": health_var,
            "health_kl_loss": health_kl_loss,
            "health_reconstruction_loss": health_reconstruction_loss,
        }

    def _transition_group_tensor(
        self,
        metadata: SensorMetadataBatch,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Resolve a grouping tensor for sequence-aware temporal regularization."""
        key = self.config.transition_group_key
        if key == "global":
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        tensor = getattr(metadata, key, None)
        if tensor is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        return _ensure_1d(tensor, name=key).long().to(device=device)

    def _temporal_transition_losses(
        self,
        *,
        latent_mean: Tensor,
        latent_var: Tensor,
        health_mean: Tensor,
        health_var: Tensor,
        aux_features: Tensor,
        X: Tensor | None,
        sensor_metadata: SensorMetadataBatch,
    ) -> tuple[Tensor, Tensor]:
        """Compute sequence-aware transition KL penalties for the generative path."""
        zero = latent_mean.new_tensor(0.0)
        if (
            self.config.inference_strategy != "joint_generative"
            or not self.config.use_temporal_transition_prior
            or self.latent_transition_model is None
            or X is None
            or self.config.transition_time_index is None
        ):
            return zero, zero
        X = _ensure_2d(X, name="X")
        if self.config.transition_time_index >= X.shape[1]:
            return zero, zero

        time_values = X[:, self.config.transition_time_index]
        group_ids = self._transition_group_tensor(
            sensor_metadata,
            batch_size=latent_mean.shape[0],
            device=latent_mean.device,
        )
        transition_losses: list[Tensor] = []
        health_transition_losses: list[Tensor] = []

        for group_id in torch.unique(group_ids):
            member_mask = group_ids == group_id
            member_indices = member_mask.nonzero(as_tuple=False).squeeze(-1)
            if member_indices.numel() < 2:
                continue
            ordered = member_indices[torch.argsort(time_values[member_indices])]
            prev_index = ordered[:-1]
            next_index = ordered[1:]
            delta_t = torch.clamp(
                (time_values[next_index] - time_values[prev_index]).abs().unsqueeze(-1),
                min=self.config.variance_floor,
            )
            transition_input_parts = [
                latent_mean[prev_index].unsqueeze(-1),
                torch.log(delta_t),
            ]
            if self.config.use_sensor_health_latent:
                transition_input_parts.append(health_mean[prev_index])
            if aux_features.numel() > 0:
                transition_input_parts.append(aux_features[prev_index])
            transition_input = torch.cat(transition_input_parts, dim=-1)

            pred_mean, pred_logvar = self.latent_transition_model(transition_input).unbind(dim=-1)
            pred_logvar = torch.clamp(
                pred_logvar,
                min=-self.config.variational_logvar_clip,
                max=self.config.variational_logvar_clip,
            )
            pred_var = torch.exp(pred_logvar)
            next_var = torch.clamp(latent_var[next_index], min=self.config.variance_floor)
            transition_kl = 0.5 * (
                torch.log(pred_var)
                - torch.log(next_var)
                + (next_var + (latent_mean[next_index] - pred_mean).pow(2)) / pred_var
                - 1.0
            )
            transition_losses.append(transition_kl.mean())

            if (
                self.config.use_sensor_health_latent
                and self.health_transition_model is not None
                and health_mean.numel() > 0
            ):
                health_pred = self.health_transition_model(transition_input)
                pred_health_mean, pred_health_logvar = torch.chunk(health_pred, 2, dim=-1)
                pred_health_logvar = torch.clamp(
                    pred_health_logvar,
                    min=-self.config.variational_logvar_clip,
                    max=self.config.variational_logvar_clip,
                )
                pred_health_var = torch.exp(pred_health_logvar)
                next_health_var = torch.clamp(
                    health_var[next_index],
                    min=self.config.variance_floor,
                )
                health_kl = 0.5 * (
                    torch.log(pred_health_var)
                    - torch.log(next_health_var)
                    + (next_health_var + (health_mean[next_index] - pred_health_mean).pow(2)) / pred_health_var
                    - 1.0
                )
                health_transition_losses.append(health_kl.sum(dim=-1).mean())

        if not transition_losses:
            return zero, zero
        transition_loss = torch.stack(transition_losses).mean()
        if not health_transition_losses:
            return transition_loss, zero
        return transition_loss, torch.stack(health_transition_losses).mean()

    def _compute_logits_from_aux(
        self,
        *,
        z_mean: Tensor,
        z_var: Tensor,
        aux_features: Tensor,
        sensor_group: Tensor | None,
        logit_scale: float,
        health_mean: Tensor | None = None,
    ) -> Tensor:
        """Compute missingness logits from latent summaries and prepared features."""
        latent_features = torch.stack(
            [z_mean, _to_log_variance(z_var, self.config.variance_floor)],
            dim=-1,
        )
        if health_mean is not None and self.config.use_sensor_health_latent:
            latent_features = torch.cat([latent_features, health_mean], dim=-1)
        batch_size = z_mean.shape[0]
        device = z_mean.device
        dtype = z_mean.dtype

        if self.config.mode == "selection":
            full_features = torch.cat([latent_features, aux_features], dim=-1)
            hidden = self.selection_trunk(full_features)
        else:
            latent_hidden = self.latent_branch(latent_features)
            if aux_features.numel() == 0:
                aux_features = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            pattern_hidden = self.pattern_branch(aux_features)
            hidden = self.fusion(
                torch.cat(
                    [latent_hidden, pattern_hidden, latent_hidden * pattern_hidden],
                    dim=-1,
                )
            )

        logits = self.selection_head(hidden).squeeze(-1)
        logits = logits + self._group_head_adjustment(hidden=hidden, sensor_group=sensor_group)
        return logits * float(logit_scale)

    def _sampled_missingness_probability(
        self,
        *,
        latent_mean: Tensor,
        latent_var: Tensor,
        aux_features: Tensor,
        sensor_group: Tensor | None,
        health_mean: Tensor,
        health_var: Tensor,
        logit_scale: float,
    ) -> Tensor:
        """Approximate `p(R=0 | q(z, h))` under sampled latent paths."""
        latent_samples, health_samples = self._generative_sample_paths(
            latent_mean=latent_mean,
            latent_var=latent_var,
            health_mean=health_mean,
            health_var=health_var,
        )
        sample_probabilities: list[Tensor] = []
        for sample_index in range(latent_samples.shape[0]):
            logits = self._compute_logits_from_aux(
                z_mean=latent_samples[sample_index],
                z_var=latent_var,
                aux_features=aux_features,
                sensor_group=sensor_group,
                logit_scale=logit_scale,
                health_mean=health_samples[sample_index],
            )
            sample_probabilities.append(torch.sigmoid(logits))
        mean_probability = torch.stack(sample_probabilities, dim=0).mean(dim=0)
        return torch.clamp(
            mean_probability,
            min=self.config.probability_floor,
            max=1.0 - self.config.probability_floor,
        )

    def _sampled_missingness_loss(
        self,
        *,
        target_missing: Tensor,
        latent_mean: Tensor,
        latent_var: Tensor,
        aux_features: Tensor,
        sensor_group: Tensor | None,
        health_mean: Tensor,
        health_var: Tensor,
        sample_weight: Tensor | None,
        latent_samples: Tensor | None = None,
        health_samples: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Approximate expected Bernoulli NLL under sampled latent paths."""
        if latent_samples is None or health_samples is None:
            latent_samples, health_samples = self._generative_sample_paths(
                latent_mean=latent_mean,
                latent_var=latent_var,
                health_mean=health_mean,
                health_var=health_var,
            )
        sample_losses: list[Tensor] = []
        sample_probabilities: list[Tensor] = []
        pos_weight = None
        if self.config.positive_class_weight != 1.0:
            pos_weight = latent_mean.new_tensor(self.config.positive_class_weight)
        for sample_index in range(latent_samples.shape[0]):
            logits = self._compute_logits_from_aux(
                z_mean=latent_samples[sample_index],
                z_var=latent_var,
                aux_features=aux_features,
                sensor_group=sensor_group,
                logit_scale=1.0,
                health_mean=health_samples[sample_index],
            )
            if sample_weight is None:
                sample_losses.append(
                    F.binary_cross_entropy_with_logits(logits, target_missing, pos_weight=pos_weight)
                )
            else:
                losses = F.binary_cross_entropy_with_logits(
                    logits,
                    target_missing,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                weighted = (losses * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1.0)
                sample_losses.append(weighted)
            sample_probabilities.append(torch.sigmoid(logits))
        mean_probability = torch.stack(sample_probabilities, dim=0).mean(dim=0)
        mean_probability = torch.clamp(
            mean_probability,
            min=self.config.probability_floor,
            max=1.0 - self.config.probability_floor,
        )
        return torch.stack(sample_losses).mean(), mean_probability

    def _group_head_adjustment(self, hidden: Tensor, sensor_group: Tensor | None) -> Tensor:
        """Apply sensor-group-specific head corrections in the heterogeneous variant."""
        if self.group_weight is None or self.group_bias is None or sensor_group is None:
            return torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        group_index = _ensure_1d(sensor_group, name="sensor_group").long().to(hidden.device)
        weight = self.group_weight(group_index)
        bias = self.group_bias(group_index).squeeze(-1)
        return (hidden * weight).sum(dim=-1) + bias

    def logit(
        self,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        X: Tensor | None = None,
        logit_scale: float = 1.0,
        target_missing: Tensor | None = None,
    ) -> Tensor:
        """Compute missingness logits under either homogeneous or heterogeneous assumptions.

        Args:
            z_mean: Posterior mean summaries with shape `[N]`.
            z_var: Posterior variance summaries with shape `[N]`.
            y: Optional observed targets with shape `[N]` used by the
                joint-variational encoder.
            observation_available: Optional observation-availability mask with
                shape `[N]`.
            context: Optional contextual features with shape `[N, C]`.
            sensor_metadata: Optional categorical and continuous metadata.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence indicator with shape `[N]`.
            X: Optional raw coordinate/features with shape `[N, D]`.
            logit_scale: Optional multiplicative MNAR-strength parameter.
            target_missing: Optional training-time missingness labels used to
                expose `R` to the joint variational encoder.

        Returns:
            Missingness logits with shape `[N]`.
        """
        z_mean = _ensure_1d(z_mean, name="z_mean")
        z_var = _ensure_1d(z_var, name="z_var")
        if z_mean.shape[0] != z_var.shape[0]:
            raise ValueError("z_mean and z_var must have the same leading dimension.")

        batch_size = z_mean.shape[0]
        device = z_mean.device
        dtype = z_mean.dtype
        metadata = SensorMetadataBatch.from_input(sensor_metadata)

        aux_features = self._assemble_auxiliary_features(
            batch_size=batch_size,
            context=context,
            M=M,
            S=S,
            dynamic_residual=dynamic_residual,
            dynamic_threshold=dynamic_threshold,
            dynamic_feature_available=dynamic_feature_available,
            X=X,
            sensor_metadata=metadata,
            device=device,
            dtype=dtype,
        )
        latent_mean, latent_var, _ = self._infer_latent_posterior_from_aux(
            z_mean,
            z_var,
            aux_features=aux_features,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        health_stats = self._infer_health_posterior_from_aux(
            z_mean=latent_mean,
            z_var=latent_var,
            aux_features=aux_features,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        if self.config.inference_strategy == "joint_generative":
            probabilities = self._sampled_missingness_probability(
                latent_mean=latent_mean,
                latent_var=latent_var,
                aux_features=aux_features,
                sensor_group=metadata.sensor_group,
                health_mean=health_stats["health_mean"],
                health_var=health_stats["health_var"],
                logit_scale=logit_scale,
            )
            return _probability_to_logit(probabilities, self.config.probability_floor)
        return self._compute_logits_from_aux(
            z_mean=latent_mean,
            z_var=latent_var,
            aux_features=aux_features,
            sensor_group=metadata.sensor_group,
            logit_scale=logit_scale,
            health_mean=health_stats["health_mean"],
        )

    def predict_proba(
        self,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        X: Tensor | None = None,
        logit_scale: float = 1.0,
    ) -> Tensor:
        """Return structural missingness probabilities with shape `[N]`."""
        if self.config.inference_strategy == "joint_generative":
            z_mean = _ensure_1d(z_mean, name="z_mean")
            z_var = _ensure_1d(z_var, name="z_var")
            metadata = SensorMetadataBatch.from_input(sensor_metadata)
            aux_features = self._assemble_auxiliary_features(
                batch_size=z_mean.shape[0],
                context=context,
                M=M,
                S=S,
                dynamic_residual=dynamic_residual,
                dynamic_threshold=dynamic_threshold,
                dynamic_feature_available=dynamic_feature_available,
                X=X,
                sensor_metadata=metadata,
                device=z_mean.device,
                dtype=z_mean.dtype,
            )
            latent_mean, latent_var, _ = self._infer_latent_posterior_from_aux(
                z_mean,
                z_var,
                aux_features=aux_features,
                y=y,
                observation_available=observation_available,
            )
            health_stats = self._infer_health_posterior_from_aux(
                z_mean=latent_mean,
                z_var=latent_var,
                aux_features=aux_features,
                y=y,
                observation_available=observation_available,
            )
            return self._sampled_missingness_probability(
                latent_mean=latent_mean,
                latent_var=latent_var,
                aux_features=aux_features,
                sensor_group=metadata.sensor_group,
                health_mean=health_stats["health_mean"],
                health_var=health_stats["health_var"],
                logit_scale=logit_scale,
            )
        logits = self.logit(
            z_mean=z_mean,
            z_var=z_var,
            y=y,
            observation_available=observation_available,
            context=context,
            sensor_metadata=sensor_metadata,
            M=M,
            S=S,
            dynamic_residual=dynamic_residual,
            dynamic_threshold=dynamic_threshold,
            dynamic_feature_available=dynamic_feature_available,
            X=X,
            logit_scale=logit_scale,
        )
        probabilities = torch.sigmoid(logits)
        return torch.clamp(
            probabilities,
            min=self.config.probability_floor,
            max=1.0 - self.config.probability_floor,
        )

    def forward(
        self,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        X: Tensor | None = None,
        logit_scale: float = 1.0,
    ) -> Tensor:
        """Compute structural missingness probabilities.

        Args:
            z_mean: Posterior means with shape `[N]`.
            z_var: Posterior variances with shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            X: Optional original inputs with shape `[N, D]`.

        Returns:
            Missingness probabilities with shape `[N]`.
        """
        return self.predict_proba(
            z_mean=z_mean,
            z_var=z_var,
            y=y,
            observation_available=observation_available,
            context=context,
            sensor_metadata=sensor_metadata,
            M=M,
            S=S,
            dynamic_residual=dynamic_residual,
            dynamic_threshold=dynamic_threshold,
            dynamic_feature_available=dynamic_feature_available,
            X=X,
            logit_scale=logit_scale,
        )

    def compute_loss_components(
        self,
        target_missing: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        X: Tensor | None = None,
        sample_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute scalar loss terms for structural missingness training.

        Args:
            target_missing: Missingness indicators with shape `[N]`, where `1` means missing.
            z_mean: Posterior means with shape `[N]`.
            z_var: Posterior variances with shape `[N]`.
            y: Optional observed targets with shape `[N]` used by the
                joint-variational encoder and reconstruction term.
            observation_available: Optional observation-availability mask with
                shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            X: Optional original inputs with shape `[N, D]`.
            sample_weight: Optional observation weights with shape `[N]`.

        Returns:
            Dictionary containing total, missingness, reconstruction, and KL losses.
        """
        target_missing = _ensure_1d(target_missing, name="target_missing").float()
        z_mean = _ensure_1d(z_mean, name="z_mean")
        z_var = _ensure_1d(z_var, name="z_var")
        metadata = SensorMetadataBatch.from_input(sensor_metadata)
        aux_features = self._assemble_auxiliary_features(
            batch_size=z_mean.shape[0],
            context=context,
            M=M,
            S=S,
            dynamic_residual=dynamic_residual,
            dynamic_threshold=dynamic_threshold,
            dynamic_feature_available=dynamic_feature_available,
            X=X,
            sensor_metadata=metadata,
            device=z_mean.device,
            dtype=z_mean.dtype,
        )
        latent_mean, latent_var, variational_stats = self._infer_latent_posterior_from_aux(
            z_mean,
            z_var,
            aux_features=aux_features,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        health_stats = self._infer_health_posterior_from_aux(
            z_mean=latent_mean,
            z_var=latent_var,
            aux_features=aux_features,
            y=y,
            observation_available=observation_available,
            target_missing=target_missing,
        )
        target_missing = target_missing.to(z_mean)
        if sample_weight is not None:
            sample_weight = _ensure_1d(sample_weight, name="sample_weight").to(z_mean)
        if self.config.inference_strategy == "joint_generative":
            latent_samples, health_samples = self._generative_sample_paths(
                latent_mean=latent_mean,
                latent_var=latent_var,
                health_mean=health_stats["health_mean"],
                health_var=health_stats["health_var"],
            )
            missingness_loss, mean_probability = self._sampled_missingness_loss(
                target_missing=target_missing,
                latent_mean=latent_mean,
                latent_var=latent_var,
                aux_features=aux_features,
                sensor_group=metadata.sensor_group,
                health_mean=health_stats["health_mean"],
                health_var=health_stats["health_var"],
                sample_weight=sample_weight,
                latent_samples=latent_samples,
                health_samples=health_samples,
            )
            logits = _probability_to_logit(mean_probability, self.config.probability_floor)
            reconstruction_loss = self._sampled_reconstruction_loss(
                latent_samples=latent_samples,
                latent_var=latent_var,
                health_samples=health_samples,
                aux_features=aux_features,
                observed_y=variational_stats["observed_y"],
                observation_available=variational_stats["observation_available"],
            )
        else:
            logits = self._compute_logits_from_aux(
                z_mean=latent_mean,
                z_var=latent_var,
                aux_features=aux_features,
                sensor_group=metadata.sensor_group,
                logit_scale=1.0,
                health_mean=health_stats["health_mean"],
            )
            target_missing = target_missing.to(logits)
            pos_weight = None
            if self.config.positive_class_weight != 1.0:
                pos_weight = logits.new_tensor(self.config.positive_class_weight)
            if sample_weight is None:
                missingness_loss = F.binary_cross_entropy_with_logits(logits, target_missing, pos_weight=pos_weight)
            else:
                losses = F.binary_cross_entropy_with_logits(
                    logits,
                    target_missing,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                missingness_loss = (losses * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1.0)
            reconstruction_loss = self._reconstruction_loss(
                posterior_mean=latent_mean,
                posterior_var=latent_var,
                aux_features=aux_features,
                observed_y=variational_stats["observed_y"],
                observation_available=variational_stats["observation_available"],
            )
        kl_loss = variational_stats["kl_loss"]
        health_kl_loss = health_stats["health_kl_loss"]
        health_reconstruction_loss = health_stats["health_reconstruction_loss"]
        transition_loss, health_transition_loss = self._temporal_transition_losses(
            latent_mean=latent_mean,
            latent_var=latent_var,
            health_mean=health_stats["health_mean"],
            health_var=health_stats["health_var"],
            aux_features=aux_features,
            X=X,
            sensor_metadata=metadata,
        )
        total_loss = missingness_loss
        if self.config.inference_strategy in {"joint_variational", "joint_generative"}:
            total_loss = (
                total_loss
                + self.config.reconstruction_weight * reconstruction_loss
                + self.config.kl_weight * kl_loss
                + self.config.health_kl_weight * health_kl_loss
                + self.config.health_reconstruction_weight * health_reconstruction_loss
            )
        if self.config.inference_strategy == "joint_generative":
            total_loss = (
                total_loss
                + self.config.transition_weight * transition_loss
                + self.config.health_transition_weight * health_transition_loss
            )
        return {
            "total_loss": total_loss,
            "missingness_loss": missingness_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "health_kl_loss": health_kl_loss,
            "health_reconstruction_loss": health_reconstruction_loss,
            "transition_loss": transition_loss,
            "health_transition_loss": health_transition_loss,
        }

    def compute_loss(
        self,
        target_missing: Tensor,
        z_mean: Tensor,
        z_var: Tensor,
        *,
        y: Tensor | None = None,
        observation_available: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        X: Tensor | None = None,
        sample_weight: Tensor | None = None,
    ) -> Tensor:
        """Compute the scalar structural missingness training loss."""
        return self.compute_loss_components(
            target_missing=target_missing,
            z_mean=z_mean,
            z_var=z_var,
            y=y,
            observation_available=observation_available,
            context=context,
            sensor_metadata=sensor_metadata,
            M=M,
            S=S,
            dynamic_residual=dynamic_residual,
            dynamic_threshold=dynamic_threshold,
            dynamic_feature_available=dynamic_feature_available,
            X=X,
            sample_weight=sample_weight,
        )["total_loss"]
