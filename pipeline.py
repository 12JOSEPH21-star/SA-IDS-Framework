from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import gpytorch
import torch
from torch import Tensor, nn

from models import (
    DynamicObservationModel,
    MissingMechanism,
    MissingMechanismConfig,
    ObservationModelConfig,
    SensorMetadataBatch,
    SparseGPConfig,
    SpatiotemporalSparseGP,
)
from policy import InformationPolicyConfig, LazyGreedyOptimizer
from reliability import ConformalConfig, ConformalPredictor


LOGGER = logging.getLogger(__name__)


def _slice_optional(tensor: Tensor | None, index: Tensor) -> Tensor | None:
    """Slice an optional tensor batch."""
    if tensor is None:
        return None
    return tensor[index]


def _slice_metadata(
    metadata: SensorMetadataBatch | Mapping[str, Tensor] | None,
    index: Tensor,
) -> SensorMetadataBatch:
    """Slice sensor metadata for minibatched missingness training."""
    batch = SensorMetadataBatch.from_input(metadata)
    return SensorMetadataBatch(
        sensor_type=_slice_optional(batch.sensor_type, index),
        sensor_group=_slice_optional(batch.sensor_group, index),
        sensor_modality=_slice_optional(batch.sensor_modality, index),
        installation_environment=_slice_optional(batch.installation_environment, index),
        maintenance_state=_slice_optional(batch.maintenance_state, index),
        continuous=_slice_optional(batch.continuous, index),
    )


def _coerce_tensor_2d(x: Tensor, *, name: str) -> Tensor:
    """Require a two-dimensional tensor."""
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], received {tuple(x.shape)}.")
    return x


def _coerce_tensor_1d(x: Tensor, *, name: str) -> Tensor:
    """Require a one-dimensional tensor."""
    if x.ndim == 2 and x.shape[-1] == 1:
        return x.squeeze(-1)
    if x.ndim != 1:
        raise ValueError(f"{name} must have shape [N] or [N, 1], received {tuple(x.shape)}.")
    return x


@dataclass
class StateTrainingConfig:
    """Training configuration for the sparse GP state model."""

    epochs: int = 100
    training_strategy: str = "sequential"
    learning_rate: float = 1e-2
    batch_size: int = 512
    weight_decay: float = 0.0
    max_grad_norm: float | None = 10.0
    reinitialize_inducing_points: bool = True
    joint_missingness_weight: float = 1.0

    def __post_init__(self) -> None:
        valid_strategies = {"sequential", "joint_variational"}
        if self.training_strategy not in valid_strategies:
            raise ValueError(f"training_strategy must be one of {sorted(valid_strategies)}.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.joint_missingness_weight < 0.0:
            raise ValueError("joint_missingness_weight must be non-negative.")


@dataclass
class MissingnessTrainingConfig:
    """Training configuration for the structural missingness model."""

    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 1024
    weight_decay: float = 1e-4
    max_grad_norm: float | None = 10.0


@dataclass
class SilenceAwareIDSConfig:
    """Configuration bundle for the full silence-aware sensing pipeline."""

    state: SparseGPConfig
    observation: ObservationModelConfig = field(default_factory=ObservationModelConfig)
    missingness: MissingMechanismConfig = field(default_factory=MissingMechanismConfig)
    policy: InformationPolicyConfig = field(default_factory=InformationPolicyConfig)
    reliability: ConformalConfig = field(default_factory=ConformalConfig)
    state_training: StateTrainingConfig = field(default_factory=StateTrainingConfig)
    missingness_training: MissingnessTrainingConfig = field(default_factory=MissingnessTrainingConfig)
    use_m2: bool = True
    use_m3: bool = True
    use_m5: bool = True
    homogeneous_missingness: bool = False
    sensor_conditional_missingness: bool = True

    def __post_init__(self) -> None:
        if self.homogeneous_missingness and self.sensor_conditional_missingness:
            raise ValueError(
                "homogeneous_missingness and sensor_conditional_missingness cannot both be enabled."
            )
        if self.use_m3 and not (self.homogeneous_missingness or self.sensor_conditional_missingness):
            raise ValueError(
                "When use_m3=True, choose either homogeneous_missingness or sensor_conditional_missingness."
            )


@dataclass
class PipelineFitSummary:
    """Training and calibration artifacts returned by the unified fit routine."""

    state_history: dict[str, list[float]]
    observation_history: dict[str, list[float]] | None = None
    dynamic_silence_threshold: float | None = None
    missingness_history: dict[str, list[float]] | None = None
    conformal_quantile: float | None = None


@dataclass
class AblationOutcome:
    """Structured output for one ablation configuration."""

    variant_name: str
    metrics: dict[str, float]
    report: dict[str, Any]
    selection: dict[str, Any] | None = None


class SilenceAwareIDS(nn.Module):
    """Unified silence-aware information-driven sensing pipeline."""

    def __init__(
        self,
        config: SilenceAwareIDSConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = self._coerce_config(config)

        self.state_model = SpatiotemporalSparseGP(config=self.config.state)
        self.state_likelihood = self.state_model.build_likelihood()
        self.observation_model = DynamicObservationModel(self.config.observation)

        missingness_config = self._resolved_missingness_config(self.config)
        self.use_m3 = self.config.use_m3
        self.use_m5 = self.config.use_m5
        self.missingness_model = MissingMechanism(missingness_config) if self.use_m3 else None
        self.policy_optimizer = LazyGreedyOptimizer(self.config.policy)
        self.reliability_model = ConformalPredictor(self.config.reliability) if self.use_m5 else None

    @staticmethod
    def _coerce_dataclass(
        payload: Any,
        config_type: type[Any],
        *,
        default: Any | None = None,
    ) -> Any:
        """Coerce dictionaries into typed dataclass configs."""
        if payload is None:
            if default is None:
                return config_type()
            return default
        if isinstance(payload, config_type):
            return payload
        if isinstance(payload, Mapping):
            return config_type(**payload)
        raise TypeError(f"Expected {config_type.__name__} or mapping, received {type(payload)!r}.")

    @classmethod
    def _coerce_config(cls, config: SilenceAwareIDSConfig | Mapping[str, Any]) -> SilenceAwareIDSConfig:
        """Build a pipeline config from either a mapping or dataclass."""
        if isinstance(config, SilenceAwareIDSConfig):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("config must be a SilenceAwareIDSConfig or mapping.")
        return SilenceAwareIDSConfig(
            state=cls._coerce_dataclass(config.get("state"), SparseGPConfig),
            observation=cls._coerce_dataclass(
                config.get("observation"),
                ObservationModelConfig,
                default=ObservationModelConfig(),
            ),
            missingness=cls._coerce_dataclass(
                config.get("missingness"),
                MissingMechanismConfig,
                default=MissingMechanismConfig(),
            ),
            policy=cls._coerce_dataclass(
                config.get("policy"),
                InformationPolicyConfig,
                default=InformationPolicyConfig(),
            ),
            reliability=cls._coerce_dataclass(
                config.get("reliability"),
                ConformalConfig,
                default=ConformalConfig(),
            ),
            state_training=cls._coerce_dataclass(
                config.get("state_training"),
                StateTrainingConfig,
                default=StateTrainingConfig(),
            ),
            missingness_training=cls._coerce_dataclass(
                config.get("missingness_training"),
                MissingnessTrainingConfig,
                default=MissingnessTrainingConfig(),
            ),
            use_m2=bool(config.get("use_m2", True)),
            use_m3=bool(config.get("use_m3", True)),
            use_m5=bool(config.get("use_m5", True)),
            homogeneous_missingness=bool(config.get("homogeneous_missingness", False)),
            sensor_conditional_missingness=bool(config.get("sensor_conditional_missingness", True)),
        )

    @staticmethod
    def _resolved_missingness_config(config: SilenceAwareIDSConfig) -> MissingMechanismConfig:
        """Resolve the explicit M3 ablation assumption."""
        assumption = "homogeneous" if config.homogeneous_missingness else "sensor_conditional"
        base = config.missingness
        return MissingMechanismConfig(
            mode=base.mode,
            assumption=assumption,
            inference_strategy=base.inference_strategy,
            context_dim=base.context_dim,
            x_dim=base.x_dim,
            continuous_metadata_dim=base.continuous_metadata_dim,
            hidden_dim=base.hidden_dim,
            trunk_depth=base.trunk_depth,
            encoder_hidden_dim=base.encoder_hidden_dim,
            encoder_depth=base.encoder_depth,
            sensor_embedding_dim=base.sensor_embedding_dim,
            num_sensor_types=base.num_sensor_types,
            num_sensor_groups=base.num_sensor_groups,
            num_sensor_modalities=base.num_sensor_modalities,
            num_installation_environments=base.num_installation_environments,
            num_maintenance_states=base.num_maintenance_states,
            include_x=base.include_x,
            include_m=base.include_m,
            include_s=base.include_s,
            include_dynamic_residual=base.include_dynamic_residual,
            include_dynamic_threshold=base.include_dynamic_threshold,
            include_normalized_residual=base.include_normalized_residual,
            include_dynamic_feature_availability=base.include_dynamic_feature_availability,
            use_dynamic_features_for_training=base.use_dynamic_features_for_training,
            use_dynamic_features_for_prediction=base.use_dynamic_features_for_prediction,
            use_dynamic_features_for_policy=base.use_dynamic_features_for_policy,
            use_group_heads=base.use_group_heads,
            use_observed_y_in_variational_encoder=base.use_observed_y_in_variational_encoder,
            use_observation_mask_in_variational_encoder=base.use_observation_mask_in_variational_encoder,
            use_sensor_health_latent=base.use_sensor_health_latent,
            health_latent_dim=base.health_latent_dim,
            variance_floor=base.variance_floor,
            probability_floor=base.probability_floor,
            positive_class_weight=base.positive_class_weight,
            reconstruction_weight=base.reconstruction_weight,
            kl_weight=base.kl_weight,
            health_kl_weight=base.health_kl_weight,
            health_reconstruction_weight=base.health_reconstruction_weight,
            variational_logvar_clip=base.variational_logvar_clip,
        )

    @staticmethod
    def _iterate_batches(size: int, batch_size: int, *, device: torch.device) -> list[Tensor]:
        """Generate minibatch indices."""
        permutation = torch.randperm(size, device=device)
        return [permutation[start : start + batch_size] for start in range(0, size, batch_size)]

    def _resolve_inference_batch_size(self, num_rows: int, batch_size: int | None) -> int:
        """Resolve a safe inference chunk size for large-scale prediction paths."""
        if num_rows <= 0:
            raise ValueError("num_rows must be positive.")
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive when provided.")
            return min(batch_size, num_rows)
        configured = max(1, int(self.config.state_training.batch_size))
        return min(configured, num_rows)

    def fit_observation_model(
        self,
        X: Tensor,
        y: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        steps: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
    ) -> dict[str, list[float]] | None:
        """Fit the observation-link parameters used by dynamic silence detection.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Observed targets with shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            steps: Optional override for observation-link optimization steps.
            learning_rate: Optional override for observation-link learning rate.

        Returns:
            Optimization history keyed by `"loss"`, or `None` when M2 is disabled.
        """
        if not self.config.use_m2:
            return None
        X = _coerce_tensor_2d(X, name="X")
        y = _coerce_tensor_1d(y, name="y")
        z_mean, z_var = self.predict_state(
            X,
            batch_size=batch_size,
            include_observation_noise=self.config.observation.use_observation_noise,
        )
        return self.observation_model.fit_observation_link(
            y_true=y,
            z_mean=z_mean,
            z_var=z_var,
            context=context,
            sensor_metadata=sensor_metadata,
            steps=steps,
            learning_rate=learning_rate,
        )

    def fit_state_model(
        self,
        X: Tensor,
        y: Tensor,
        *,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, list[float]]:
        """Fit the sparse variational GP on observed targets.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Targets with shape `[N]`; NaNs are ignored.
            epochs: Optional override for training epochs.
            batch_size: Optional override for minibatch size.

        Returns:
            Training history keyed by `"loss"`.
        """
        X = _coerce_tensor_2d(X, name="X")
        y = _coerce_tensor_1d(y, name="y")
        observed_mask = torch.isfinite(y)
        X_obs = X[observed_mask]
        y_obs = y[observed_mask]
        if X_obs.numel() == 0:
            raise ValueError("fit_state_model requires at least one observed target.")

        if self.config.state_training.reinitialize_inducing_points:
            self.state_model.initialize_inducing_points(X_obs)

        self.state_model.train()
        self.state_likelihood.train()
        optimizer = torch.optim.Adam(
            [
                {"params": self.state_model.parameters()},
                {"params": self.state_likelihood.parameters()},
            ],
            lr=self.config.state_training.learning_rate,
            weight_decay=self.config.state_training.weight_decay,
        )
        objective = gpytorch.mlls.VariationalELBO(
            self.state_likelihood,
            self.state_model,
            num_data=int(X_obs.shape[0]),
        )

        active_epochs = self.config.state_training.epochs if epochs is None else epochs
        active_batch_size = self.config.state_training.batch_size if batch_size is None else batch_size
        history = {"loss": []}

        for epoch in range(active_epochs):
            epoch_loss = 0.0
            batches = self._iterate_batches(X_obs.shape[0], active_batch_size, device=X_obs.device)
            for batch_index in batches:
                batch_x = X_obs[batch_index]
                batch_y = y_obs[batch_index]
                optimizer.zero_grad(set_to_none=True)
                # batch_x: [B, D], batch_y: [B]
                output = self.state_model(batch_x)
                loss = -objective(output, batch_y)
                loss.backward()
                if self.config.state_training.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.state_model.parameters()) + list(self.state_likelihood.parameters()),
                        max_norm=self.config.state_training.max_grad_norm,
                    )
                optimizer.step()
                epoch_loss += float(loss.item()) * batch_x.shape[0]
            mean_epoch_loss = epoch_loss / float(X_obs.shape[0])
            history["loss"].append(mean_epoch_loss)
            LOGGER.info("State model epoch %d/%d - loss %.6f", epoch + 1, active_epochs, mean_epoch_loss)

        self.state_model.eval()
        self.state_likelihood.eval()
        return history

    def fit_joint_state_missingness(
        self,
        X: Tensor,
        y: Tensor,
        missing_indicator: Tensor,
        *,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Jointly train M1 and M3 under a coupled JVI-style objective.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Observed targets with shape `[N]`; NaNs denote unavailable values.
            missing_indicator: Structural missingness labels with shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence feature with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            dynamic_residual: Optional residual feature with shape `[N]`.
            dynamic_threshold: Optional threshold feature with shape `[N]`.
            dynamic_feature_available: Optional availability feature with shape `[N]`.
            epochs: Optional override for training epochs.
            batch_size: Optional override for minibatch size.

        Returns:
            Tuple of state-history and missingness-history dictionaries.
        """
        if not self.use_m3 or self.missingness_model is None:
            raise RuntimeError("Joint JVI training requires M3 to be enabled.")
        if self.missingness_model.config.inference_strategy != "joint_variational":
            raise RuntimeError("Joint JVI training requires M3 inference_strategy='joint_variational'.")

        X = _coerce_tensor_2d(X, name="X")
        y = _coerce_tensor_1d(y, name="y")
        missing_indicator = _coerce_tensor_1d(missing_indicator, name="missing_indicator").float()
        observed_mask = torch.isfinite(y)
        X_obs = X[observed_mask]
        y_obs = y[observed_mask]
        if X_obs.numel() == 0:
            raise ValueError("fit_joint_state_missingness requires at least one observed target.")

        if self.config.state_training.reinitialize_inducing_points:
            self.state_model.initialize_inducing_points(X_obs)

        active_epochs = self.config.state_training.epochs if epochs is None else epochs
        active_batch_size = self.config.state_training.batch_size if batch_size is None else batch_size
        self.state_model.train()
        self.state_likelihood.train()
        self.missingness_model.train()
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(self.state_model.parameters()) + list(self.state_likelihood.parameters()),
                    "lr": self.config.state_training.learning_rate,
                    "weight_decay": self.config.state_training.weight_decay,
                },
                {
                    "params": self.missingness_model.parameters(),
                    "lr": self.config.missingness_training.learning_rate,
                    "weight_decay": self.config.missingness_training.weight_decay,
                },
            ],
        )
        objective = gpytorch.mlls.VariationalELBO(
            self.state_likelihood,
            self.state_model,
            num_data=int(X_obs.shape[0]),
        )

        state_history: dict[str, list[float]] = {
            "loss": [],
            "state_loss": [],
            "joint_missingness_loss": [],
        }
        missingness_history: dict[str, list[float]] = {
            "loss": [],
            "missingness_loss": [],
            "reconstruction_loss": [],
            "kl_loss": [],
            "health_kl_loss": [],
            "health_reconstruction_loss": [],
        }

        for epoch in range(active_epochs):
            epoch_total_loss = 0.0
            epoch_state_loss = 0.0
            epoch_joint_missingness_loss = 0.0
            epoch_missingness_loss = 0.0
            epoch_reconstruction_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_health_kl_loss = 0.0
            epoch_health_reconstruction_loss = 0.0
            batches = self._iterate_batches(X.shape[0], active_batch_size, device=X.device)

            for batch_index in batches:
                batch_x = X[batch_index]
                batch_y = y[batch_index]
                batch_missing = missing_indicator[batch_index]
                batch_observed = torch.isfinite(batch_y)
                optimizer.zero_grad(set_to_none=True)

                state_loss = batch_x.new_tensor(0.0)
                if batch_observed.any():
                    output_obs = self.state_model(batch_x[batch_observed])
                    state_loss = -objective(output_obs, batch_y[batch_observed])

                output_full = self.state_model(batch_x)
                latent_mean = output_full.mean
                latent_var = torch.clamp(output_full.variance, min=self.config.state.variance_floor)
                batch_metadata = _slice_metadata(sensor_metadata, batch_index)
                batch_observation_available = batch_observed.float()
                loss_components = self.missingness_model.compute_loss_components(
                    target_missing=batch_missing,
                    z_mean=latent_mean,
                    z_var=latent_var,
                    y=batch_y,
                    observation_available=batch_observation_available,
                    context=_slice_optional(context, batch_index),
                    sensor_metadata=batch_metadata,
                    M=_slice_optional(M, batch_index),
                    S=_slice_optional(S, batch_index),
                    dynamic_residual=_slice_optional(dynamic_residual, batch_index),
                    dynamic_threshold=_slice_optional(dynamic_threshold, batch_index),
                    dynamic_feature_available=_slice_optional(dynamic_feature_available, batch_index),
                    X=batch_x if self.missingness_model.config.include_x else None,
                )
                joint_missingness_loss = self.config.state_training.joint_missingness_weight * loss_components["total_loss"]
                total_loss = state_loss + joint_missingness_loss
                total_loss.backward()

                parameter_groups: list[Tensor] = list(self.state_model.parameters()) + list(self.state_likelihood.parameters())
                parameter_groups.extend(list(self.missingness_model.parameters()))
                max_grad_norm = self.config.state_training.max_grad_norm
                if max_grad_norm is None:
                    max_grad_norm = self.config.missingness_training.max_grad_norm
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(parameter_groups, max_norm=max_grad_norm)
                optimizer.step()

                weight = float(batch_index.numel())
                epoch_total_loss += float(total_loss.item()) * weight
                epoch_state_loss += float(state_loss.item()) * weight
                epoch_joint_missingness_loss += float(joint_missingness_loss.item()) * weight
                epoch_missingness_loss += float(loss_components["missingness_loss"].item()) * weight
                epoch_reconstruction_loss += float(loss_components["reconstruction_loss"].item()) * weight
                epoch_kl_loss += float(loss_components["kl_loss"].item()) * weight
                epoch_health_kl_loss += float(loss_components["health_kl_loss"].item()) * weight
                epoch_health_reconstruction_loss += (
                    float(loss_components["health_reconstruction_loss"].item()) * weight
                )

            num_rows = float(X.shape[0])
            state_history["loss"].append(epoch_total_loss / num_rows)
            state_history["state_loss"].append(epoch_state_loss / num_rows)
            state_history["joint_missingness_loss"].append(epoch_joint_missingness_loss / num_rows)
            missingness_history["loss"].append(epoch_joint_missingness_loss / num_rows)
            missingness_history["missingness_loss"].append(epoch_missingness_loss / num_rows)
            missingness_history["reconstruction_loss"].append(epoch_reconstruction_loss / num_rows)
            missingness_history["kl_loss"].append(epoch_kl_loss / num_rows)
            missingness_history["health_kl_loss"].append(epoch_health_kl_loss / num_rows)
            missingness_history["health_reconstruction_loss"].append(
                epoch_health_reconstruction_loss / num_rows
            )
            LOGGER.info(
                "Joint JVI epoch %d/%d - total %.6f state %.6f missingness %.6f",
                epoch + 1,
                active_epochs,
                state_history["loss"][-1],
                state_history["state_loss"][-1],
                state_history["joint_missingness_loss"][-1],
            )

        self.state_model.eval()
        self.state_likelihood.eval()
        self.missingness_model.eval()
        return state_history, missingness_history

    def predict_state(
        self,
        X: Tensor,
        *,
        batch_size: int | None = None,
        include_observation_noise: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Predict latent state posterior summaries.

        Args:
            X: Input coordinates with shape `[N, D]`.
            batch_size: Optional batched inference size.
            include_observation_noise: Whether to apply the Gaussian likelihood
                and return observation-space predictive variance.

        Returns:
            Predictive mean and variance, each with shape `[N]`.
        """
        X = _coerce_tensor_2d(X, name="X")
        active_batch_size = self._resolve_inference_batch_size(X.shape[0], batch_size)
        likelihood = self.state_likelihood if include_observation_noise else None
        means: list[Tensor] = []
        variances: list[Tensor] = []
        for start in range(0, X.shape[0], active_batch_size):
            chunk = X[start : start + active_batch_size]
            mean_chunk, var_chunk = self.state_model.posterior_summary(chunk, likelihood=likelihood)
            means.append(mean_chunk)
            variances.append(var_chunk)
        return torch.cat(means, dim=0), torch.cat(variances, dim=0)

    def _dynamic_feature_mode_enabled(self, mode: str) -> bool:
        """Return whether post-observation dynamic features are allowed in one pipeline mode."""
        if mode == "training":
            return self.config.missingness.use_dynamic_features_for_training
        if mode == "prediction":
            return self.config.missingness.use_dynamic_features_for_prediction
        if mode == "policy":
            return self.config.missingness.use_dynamic_features_for_policy
        raise ValueError(f"Unknown dynamic feature mode: {mode}")

    def _resolve_dynamic_feature_bundle(
        self,
        X: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        S: Tensor | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        batch_size: int | None = None,
        mode: str = "prediction",
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor]:
        """Resolve dynamic-silence-derived features under training, prediction, or policy mode."""
        X = _coerce_tensor_2d(X, name="X")
        num_rows = X.shape[0]
        device = X.device
        dtype = X.dtype
        zero_available = torch.zeros(num_rows, device=device, dtype=dtype)

        if not self.config.use_m2 or not self._dynamic_feature_mode_enabled(mode):
            return None, None, None, zero_available

        effective_s = S
        effective_residual = dynamic_residual
        effective_threshold = dynamic_threshold
        effective_available = (
            None
            if dynamic_feature_available is None
            else _coerce_tensor_1d(dynamic_feature_available, name="dynamic_feature_available").to(
                device=device,
                dtype=dtype,
            )
        )

        need_derived_features = (
            y is not None
            and (
                effective_s is None
                or effective_residual is None
                or effective_threshold is None
                or effective_available is None
            )
        )
        if need_derived_features:
            silence = self.detect_silence(
                X,
                y,
                context=context,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
            )
            if effective_s is None:
                effective_s = silence["dynamic_silence"].float()
            if effective_residual is None:
                effective_residual = silence["residuals"]
            if effective_threshold is None:
                effective_threshold = silence["threshold"]
            if effective_available is None:
                effective_available = silence["available"].to(device=device, dtype=dtype)

        if effective_available is None:
            inferred_available = float(
                any(value is not None for value in (effective_s, effective_residual, effective_threshold))
            )
            effective_available = torch.full((num_rows,), inferred_available, device=device, dtype=dtype)

        return effective_s, effective_residual, effective_threshold, effective_available

    def _prior_mean_value(self, reference: Tensor) -> Tensor:
        """Return a prior mean baseline for shrinkage under high missingness risk."""
        constant = getattr(self.state_model.mean_module, "constant", None)
        if constant is None:
            return torch.zeros((), device=reference.device, dtype=reference.dtype)
        return constant.detach().to(device=reference.device, dtype=reference.dtype).view(())

    def missingness_aware_state_summary(
        self,
        X: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        batch_size: int | None = None,
        logit_scale: float = 1.0,
        mean_shrinkage: float = 0.25,
        variance_weight: float = 1.0,
        include_observation_noise: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Produce state summaries adjusted by structural missingness risk.

        The baseline path uses plug-in GP summaries. When `M3` is configured
        with `joint_variational`, the missingness module first refines those
        summaries with an amortized latent encoder conditioned on observed
        values, availability, and heterogeneous context. High predicted
        missingness then inflates variance and shrinks the predictive mean
        toward the GP prior mean.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Optional observations with shape `[N]` for deriving dynamic silence
                residual features before M3 inference.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            batch_size: Optional inference batch size.
            logit_scale: Optional MNAR-strength multiplier.
            mean_shrinkage: Shrinkage factor applied to mean reversion.
            variance_weight: Strength of variance inflation from missingness risk.

        Returns:
            Tuple of adjusted mean, adjusted variance, and missingness probability.
        """
        mean, var = self.predict_state(
            X,
            batch_size=batch_size,
            include_observation_noise=include_observation_noise,
        )
        effective_s, effective_residual, effective_threshold, effective_available = (
            self._resolve_dynamic_feature_bundle(
                X,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                S=S,
                dynamic_residual=None,
                dynamic_threshold=None,
                dynamic_feature_available=None,
                batch_size=batch_size,
                mode="prediction",
            )
        )
        if self.use_m3 and self.missingness_model is not None:
            mean, var, _ = self.missingness_model.infer_latent_posterior(
                z_mean=mean,
                z_var=var,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                M=M,
                S=effective_s,
                dynamic_residual=effective_residual,
                dynamic_threshold=effective_threshold,
                dynamic_feature_available=effective_available,
                X=X if self.missingness_model.config.include_x else None,
            )
        p_miss = self.predict_missingness(
            X,
            y=y,
            context=context,
            M=M,
            S=effective_s,
            sensor_metadata=sensor_metadata,
            dynamic_residual=effective_residual,
            dynamic_threshold=effective_threshold,
            dynamic_feature_available=effective_available,
            batch_size=batch_size,
            logit_scale=logit_scale,
            feature_mode="prediction",
        )
        if p_miss is None:
            return mean, var, None

        safe_keep = torch.clamp(1.0 - p_miss, min=self.config.missingness.probability_floor)
        variance_multiplier = 1.0 + variance_weight * (p_miss / safe_keep)
        adjusted_var = torch.clamp(
            var * variance_multiplier,
            min=self.config.state.variance_floor,
        )
        prior_mean = self._prior_mean_value(mean)
        adjusted_mean = (1.0 - mean_shrinkage * p_miss) * mean + (mean_shrinkage * p_miss) * prior_mean
        return adjusted_mean, adjusted_var, p_miss

    def calibrate_dynamic_silence(
        self,
        X: Tensor,
        y: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        batch_size: int | None = None,
    ) -> Tensor:
        """Calibrate the observation model quantile threshold."""
        if not self.config.use_m2:
            raise RuntimeError("M2 is disabled; dynamic silence calibration is unavailable.")
        z_mean, z_var = self.predict_state(
            X,
            batch_size=batch_size,
            include_observation_noise=self.config.observation.use_observation_noise,
        )
        y = _coerce_tensor_1d(y, name="y")
        observed_mask = torch.isfinite(y)
        if not observed_mask.any():
            raise ValueError("calibrate_dynamic_silence requires at least one observed target.")
        return self.observation_model.calibrate(
            y_true=y[observed_mask],
            z_mean=z_mean[observed_mask],
            z_var=z_var[observed_mask],
            context=None if context is None else context[observed_mask],
            sensor_metadata=_slice_metadata(sensor_metadata, observed_mask.nonzero(as_tuple=False).squeeze(-1)),
        )

    def detect_silence(
        self,
        X: Tensor,
        y: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        calibration_residuals: Tensor | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Tensor]:
        """Detect dynamic silence from residual inconsistency.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Observed targets with shape `[N]`.
            calibration_residuals: Optional residuals for quantile thresholding.

        Returns:
            Dictionary with residuals, thresholds, and boolean silence flags.
        """
        z_mean, z_var = self.predict_state(
            X,
            batch_size=batch_size,
            include_observation_noise=self.config.observation.use_observation_noise,
        )
        if not self.config.use_m2:
            return {
                "residuals": torch.zeros_like(z_mean),
                "threshold": torch.zeros_like(z_mean),
                "diagnostic_score": torch.zeros_like(z_mean),
                "diagnosis_embedding": torch.zeros(
                    z_mean.shape[0],
                    self.config.observation.diagnosis_embedding_dim,
                    device=z_mean.device,
                    dtype=z_mean.dtype,
                ),
                "dynamic_silence": torch.zeros_like(z_mean, dtype=torch.bool),
                "sensor_state_probs": torch.zeros(
                    z_mean.shape[0],
                    self.config.observation.dbn_num_states,
                    device=z_mean.device,
                    dtype=z_mean.dtype,
                ),
                "sensor_state_labels": torch.full(
                    (z_mean.shape[0],),
                    -1,
                    device=z_mean.device,
                    dtype=torch.long,
                ),
                "available": torch.zeros_like(z_mean, dtype=torch.bool),
            }
        y = _coerce_tensor_1d(y, name="y")
        observed_mask = torch.isfinite(y)
        residuals = torch.zeros_like(z_mean)
        threshold = torch.zeros_like(z_mean)
        diagnostic_score = torch.zeros_like(z_mean)
        diagnosis_embedding = torch.zeros(
            z_mean.shape[0],
            self.config.observation.diagnosis_embedding_dim,
            device=z_mean.device,
            dtype=z_mean.dtype,
        )
        sensor_state_probs = torch.zeros(
            z_mean.shape[0],
            self.config.observation.dbn_num_states,
            device=z_mean.device,
            dtype=z_mean.dtype,
        )
        sensor_state_labels = torch.full(
            (z_mean.shape[0],),
            -1,
            device=z_mean.device,
            dtype=torch.long,
        )
        flags = torch.zeros_like(z_mean, dtype=torch.bool)
        if observed_mask.any():
            with torch.no_grad():
                res_obs, thr_obs, flag_obs = self.observation_model.detect_dynamic_silence(
                    y_true=y[observed_mask],
                    z_mean=z_mean[observed_mask],
                    z_var=z_var[observed_mask],
                    context=None if context is None else context[observed_mask],
                    sensor_metadata=_slice_metadata(
                        sensor_metadata,
                        observed_mask.nonzero(as_tuple=False).squeeze(-1),
                    ),
                    calibration_residuals=calibration_residuals,
                )
                score_obs = self.observation_model.diagnostic_score(
                    y_true=y[observed_mask],
                    z_mean=z_mean[observed_mask],
                    z_var=z_var[observed_mask],
                    context=None if context is None else context[observed_mask],
                    sensor_metadata=_slice_metadata(
                        sensor_metadata,
                        observed_mask.nonzero(as_tuple=False).squeeze(-1),
                    ),
                    calibration_residuals=calibration_residuals,
                )
                embedding_obs = self.observation_model.diagnosis_embedding(
                    y_true=y[observed_mask],
                    z_mean=z_mean[observed_mask],
                    z_var=z_var[observed_mask],
                    context=None if context is None else context[observed_mask],
                    sensor_metadata=_slice_metadata(
                        sensor_metadata,
                        observed_mask.nonzero(as_tuple=False).squeeze(-1),
                    ),
                    calibration_residuals=calibration_residuals,
                )
                state_probs_obs = self.observation_model.infer_sensor_state_probs(
                    y_true=y[observed_mask],
                    z_mean=z_mean[observed_mask],
                    z_var=z_var[observed_mask],
                    context=None if context is None else context[observed_mask],
                    sensor_metadata=_slice_metadata(
                        sensor_metadata,
                        observed_mask.nonzero(as_tuple=False).squeeze(-1),
                    ),
                    calibration_residuals=calibration_residuals,
                )
            residuals[observed_mask] = res_obs
            threshold[observed_mask] = thr_obs
            diagnostic_score[observed_mask] = score_obs
            diagnosis_embedding[observed_mask] = embedding_obs
            sensor_state_probs[observed_mask] = state_probs_obs
            sensor_state_labels[observed_mask] = torch.argmax(state_probs_obs, dim=-1)
            flags[observed_mask] = flag_obs
        return {
            "residuals": residuals,
            "threshold": threshold,
            "diagnostic_score": diagnostic_score,
            "diagnosis_embedding": diagnosis_embedding,
            "dynamic_silence": flags,
            "sensor_state_probs": sensor_state_probs,
            "sensor_state_labels": sensor_state_labels,
            "available": observed_mask,
        }

    def fit_missingness_model(
        self,
        X: Tensor,
        missing_indicator: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, list[float]]:
        """Fit the structural missingness mechanism.

        Args:
            X: Input coordinates with shape `[N, D]`.
            missing_indicator: Structural missingness labels with shape `[N]`.
            y: Optional observed targets with shape `[N]` for deriving dynamic
                silence residual features.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            dynamic_residual: Optional precomputed residual features with shape `[N]`.
            dynamic_threshold: Optional precomputed threshold features with shape `[N]`.
            epochs: Optional training epoch override.
            batch_size: Optional minibatch size override.

        Returns:
            Training history keyed by `"loss"`.
        """
        if not self.use_m3 or self.missingness_model is None:
            raise RuntimeError("M3 is disabled; missingness modeling is unavailable.")

        X = _coerce_tensor_2d(X, name="X")
        missing_indicator = _coerce_tensor_1d(missing_indicator, name="missing_indicator").float()
        active_epochs = self.config.missingness_training.epochs if epochs is None else epochs
        active_batch_size = self.config.missingness_training.batch_size if batch_size is None else batch_size
        z_mean, z_var = self.predict_state(X, batch_size=active_batch_size)
        effective_s, effective_residual, effective_threshold, effective_available = (
            self._resolve_dynamic_feature_bundle(
                X,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                S=S,
                dynamic_residual=dynamic_residual,
                dynamic_threshold=dynamic_threshold,
                dynamic_feature_available=dynamic_feature_available,
                batch_size=active_batch_size,
                mode="training",
            )
        )
        optimizer = torch.optim.Adam(
            self.missingness_model.parameters(),
            lr=self.config.missingness_training.learning_rate,
            weight_decay=self.config.missingness_training.weight_decay,
        )
        self.missingness_model.train()
        history = {"loss": []}
        if self.missingness_model.config.inference_strategy == "joint_variational":
            history["missingness_loss"] = []
            history["reconstruction_loss"] = []
            history["kl_loss"] = []
            if self.missingness_model.config.use_sensor_health_latent:
                history["health_kl_loss"] = []
                history["health_reconstruction_loss"] = []

        for epoch in range(active_epochs):
            epoch_loss = 0.0
            epoch_missingness_loss = 0.0
            epoch_reconstruction_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_health_kl_loss = 0.0
            epoch_health_reconstruction_loss = 0.0
            batches = self._iterate_batches(X.shape[0], active_batch_size, device=X.device)
            for batch_index in batches:
                optimizer.zero_grad(set_to_none=True)
                batch_metadata = _slice_metadata(sensor_metadata, batch_index)
                loss_components = self.missingness_model.compute_loss_components(
                    target_missing=missing_indicator[batch_index],
                    z_mean=z_mean[batch_index].detach(),
                    z_var=z_var[batch_index].detach(),
                    y=_slice_optional(y, batch_index),
                    context=_slice_optional(context, batch_index),
                    sensor_metadata=batch_metadata,
                    M=_slice_optional(M, batch_index),
                    S=_slice_optional(effective_s, batch_index),
                    dynamic_residual=_slice_optional(effective_residual, batch_index),
                    dynamic_threshold=_slice_optional(effective_threshold, batch_index),
                    dynamic_feature_available=_slice_optional(effective_available, batch_index),
                    X=X[batch_index] if self.missingness_model.config.include_x else None,
                )
                loss = loss_components["total_loss"]
                loss.backward()
                if self.config.missingness_training.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.missingness_model.parameters(),
                        max_norm=self.config.missingness_training.max_grad_norm,
                    )
                optimizer.step()
                epoch_loss += float(loss.item()) * batch_index.numel()
                epoch_missingness_loss += float(loss_components["missingness_loss"].item()) * batch_index.numel()
                epoch_reconstruction_loss += float(loss_components["reconstruction_loss"].item()) * batch_index.numel()
                epoch_kl_loss += float(loss_components["kl_loss"].item()) * batch_index.numel()
                epoch_health_kl_loss += float(loss_components["health_kl_loss"].item()) * batch_index.numel()
                epoch_health_reconstruction_loss += (
                    float(loss_components["health_reconstruction_loss"].item()) * batch_index.numel()
                )
            mean_epoch_loss = epoch_loss / float(X.shape[0])
            history["loss"].append(mean_epoch_loss)
            if self.missingness_model.config.inference_strategy == "joint_variational":
                history["missingness_loss"].append(epoch_missingness_loss / float(X.shape[0]))
                history["reconstruction_loss"].append(epoch_reconstruction_loss / float(X.shape[0]))
                history["kl_loss"].append(epoch_kl_loss / float(X.shape[0]))
                if self.missingness_model.config.use_sensor_health_latent:
                    history["health_kl_loss"].append(epoch_health_kl_loss / float(X.shape[0]))
                    history["health_reconstruction_loss"].append(
                        epoch_health_reconstruction_loss / float(X.shape[0])
                    )
            LOGGER.info(
                "Missingness model epoch %d/%d - loss %.6f",
                epoch + 1,
                active_epochs,
                mean_epoch_loss,
            )

        self.missingness_model.eval()
        return history

    def _relational_node_features(
        self,
        X: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        silence_features: dict[str, Tensor] | None = None,
        batch_size: int | None = None,
    ) -> Tensor:
        """Build relational features for CoRel-style adaptive conformal updates."""
        X = _coerce_tensor_2d(X, name="X")
        if silence_features is not None and "diagnosis_embedding" in silence_features:
            return silence_features["diagnosis_embedding"]
        if y is not None and self.config.use_m2:
            computed = self.detect_silence(
                X,
                y,
                context=context,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
            )
            return computed["diagnosis_embedding"]
        if context is not None:
            return _coerce_tensor_2d(context, name="context")
        return X

    def predict_missingness(
        self,
        X: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        dynamic_residual: Tensor | None = None,
        dynamic_threshold: Tensor | None = None,
        dynamic_feature_available: Tensor | None = None,
        batch_size: int | None = None,
        logit_scale: float = 1.0,
        feature_mode: str = "prediction",
    ) -> Tensor | None:
        """Predict structural missingness probabilities.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Optional observed targets with shape `[N]` for deriving dynamic
                silence residual features.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            dynamic_residual: Optional precomputed residual features with shape `[N]`.
            dynamic_threshold: Optional precomputed threshold features with shape `[N]`.
            batch_size: Optional inference batch size.
            logit_scale: Optional MNAR-strength multiplier for sensitivity analysis.

        Returns:
            Missingness probabilities with shape `[N]`, or `None` if M3 is disabled.
        """
        if not self.use_m3 or self.missingness_model is None:
            return None
        effective_s, effective_residual, effective_threshold, effective_available = (
            self._resolve_dynamic_feature_bundle(
                X,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                S=S,
                dynamic_residual=dynamic_residual,
                dynamic_threshold=dynamic_threshold,
                dynamic_feature_available=dynamic_feature_available,
                batch_size=batch_size,
                mode=feature_mode,
            )
        )
        z_mean, z_var = self.predict_state(X, batch_size=batch_size)
        with torch.no_grad():
            return self.missingness_model.predict_proba(
                z_mean=z_mean,
                z_var=z_var,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                M=M,
                S=effective_s,
                dynamic_residual=effective_residual,
                dynamic_threshold=effective_threshold,
                dynamic_feature_available=effective_available,
                X=X if self.missingness_model.config.include_x else None,
                logit_scale=logit_scale,
            )

    def select_sensors(
        self,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        *,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        already_selected: list[int] | None = None,
        budget: float | None = None,
        max_selections: int | None = None,
        missingness_logit_scale: float = 1.0,
        batch_size: int | None = None,
    ) -> dict[str, Tensor | list[int] | float]:
        """Select sensors under a lazy-greedy information policy.

        Args:
            candidate_x: Candidate coordinates/features with shape `[N, D]`.
            candidate_cost: Candidate costs with shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            already_selected: Optional pre-selected indices.
            budget: Optional sensing budget.
            max_selections: Optional selection cardinality limit.
            missingness_logit_scale: Optional MNAR-strength multiplier for policy sensitivity.

        Returns:
            Selection dictionary containing indices, coordinates, utility trace,
            and total accumulated cost.
        """
        candidate_x = _coerce_tensor_2d(candidate_x, name="candidate_x")
        candidate_cost = _coerce_tensor_1d(candidate_cost, name="candidate_cost")
        _, candidate_var = self.predict_state(candidate_x, batch_size=batch_size)
        p_miss = self.predict_missingness(
            candidate_x,
            context=context,
            M=M,
            S=S,
            sensor_metadata=sensor_metadata,
            batch_size=batch_size,
            logit_scale=missingness_logit_scale,
            feature_mode="policy",
        )
        availability = None if p_miss is None else 1.0 - p_miss
        return self.policy_optimizer.select(
            candidate_x=candidate_x,
            candidate_cost=candidate_cost,
            candidate_variance=candidate_var,
            availability=availability,
            candidate_context=context,
            already_selected=already_selected,
            budget=budget,
            max_selections=max_selections,
        )

    def calibrate_reliability(
        self,
        X_cal: Tensor,
        y_cal: Tensor,
        *,
        context: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        batch_size: int | None = None,
    ) -> Tensor:
        """Calibrate the conformal predictor on a held-out set."""
        if not self.use_m5 or self.reliability_model is None:
            raise RuntimeError("M5 is disabled; reliability calibration is unavailable.")
        mu_cal, var_cal = self.predict_state(
            X_cal,
            batch_size=batch_size,
            include_observation_noise=self.config.reliability.prediction_target == "observation",
        )
        y_cal = _coerce_tensor_1d(y_cal, name="y_cal")
        observed_mask = torch.isfinite(y_cal)
        node_features = None
        if self.config.reliability.mode in {"relational_adaptive", "graph_corel"}:
            node_features = self._relational_node_features(
                X_cal,
                y=y_cal,
                context=context,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
            )[observed_mask]
        q_hat = self.reliability_model.fit(
            mu_cal=mu_cal[observed_mask],
            var_cal=var_cal[observed_mask],
            y_cal=y_cal[observed_mask],
            node_features=node_features,
        )
        self.reliability_model.reset_adaptation()
        return q_hat

    def fit(
        self,
        X_train: Tensor,
        y_train: Tensor,
        *,
        missing_indicator_train: Tensor | None = None,
        context_train: Tensor | None = None,
        M_train: Tensor | None = None,
        S_train: Tensor | None = None,
        sensor_metadata_train: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        X_cal: Tensor | None = None,
        y_cal: Tensor | None = None,
        context_cal: Tensor | None = None,
        M_cal: Tensor | None = None,
        S_cal: Tensor | None = None,
        sensor_metadata_cal: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
    ) -> PipelineFitSummary:
        """Fit the configured modules end-to-end.

        Args:
            X_train: Training inputs with shape `[N, D]`.
            y_train: Training targets with shape `[N]`.
            missing_indicator_train: Optional structural missingness labels with shape `[N]`.
            context_train: Optional training context matrix with shape `[N, C]`.
            M_train: Optional training scalar feature with shape `[N]`.
            S_train: Optional training dynamic silence feature with shape `[N]`.
            sensor_metadata_train: Optional training heterogeneous sensor metadata.
            X_cal: Optional calibration inputs with shape `[M, D]`.
            y_cal: Optional calibration targets with shape `[M]`.
            context_cal: Optional calibration context matrix with shape `[M, C]`.
            M_cal: Optional calibration scalar feature with shape `[M]`.
            S_cal: Optional calibration dynamic silence feature with shape `[M]`.
            sensor_metadata_cal: Optional calibration heterogeneous sensor metadata.

        Returns:
            Structured training summary including calibration artifacts.
        """
        joint_jvi_training = (
            self.config.state_training.training_strategy == "joint_variational"
            and self.use_m3
            and self.missingness_model is not None
            and missing_indicator_train is not None
            and self.missingness_model.config.inference_strategy == "joint_variational"
        )
        missingness_history: dict[str, list[float]] | None = None
        if joint_jvi_training:
            state_history, missingness_history = self.fit_joint_state_missingness(
                X_train,
                y_train,
                missing_indicator_train,
                context=context_train,
                M=M_train,
                S=S_train,
                sensor_metadata=sensor_metadata_train,
            )
        else:
            state_history = self.fit_state_model(X_train, y_train)
        observation_history = self.fit_observation_model(
            X_train,
            y_train,
            context=context_train,
            sensor_metadata=sensor_metadata_train,
            batch_size=self.config.state_training.batch_size,
        )

        dynamic_threshold: float | None = None
        if self.config.use_m2 and self.config.observation.threshold_mode == "quantile":
            if (
                self.config.observation.require_calibration_split_for_quantile
                and (X_cal is None or y_cal is None)
            ):
                raise ValueError(
                    "Quantile-based dynamic silence calibration requires X_cal and y_cal in research mode."
                )
            calibration_x = X_cal if X_cal is not None else X_train
            calibration_y = y_cal if y_cal is not None else y_train
            dynamic_threshold = float(
                self.calibrate_dynamic_silence(
                    calibration_x,
                    calibration_y,
                    context=context_cal if X_cal is not None else context_train,
                    sensor_metadata=sensor_metadata_cal if X_cal is not None else sensor_metadata_train,
                    batch_size=self.config.state_training.batch_size,
                ).item()
            )

        if (
            (not joint_jvi_training)
            and self.use_m3
            and self.missingness_model is not None
            and missing_indicator_train is not None
        ):
            fit_s = S_train
            fit_dynamic_residual: Tensor | None = None
            fit_dynamic_threshold: Tensor | None = None
            fit_dynamic_available: Tensor | None = None
            if self.config.use_m2:
                silence_features = self.detect_silence(
                    X_train,
                    y_train,
                    context=context_train,
                    sensor_metadata=sensor_metadata_train,
                    batch_size=self.config.state_training.batch_size,
                )
                if fit_s is None:
                    fit_s = silence_features["dynamic_silence"].float()
                fit_dynamic_residual = silence_features["residuals"]
                fit_dynamic_threshold = silence_features["threshold"]
                fit_dynamic_available = silence_features["available"].float()
            missingness_history = self.fit_missingness_model(
                X_train,
                missing_indicator_train,
                y=y_train,
                context=context_train,
                M=M_train,
                S=fit_s,
                sensor_metadata=sensor_metadata_train,
                dynamic_residual=fit_dynamic_residual,
                dynamic_threshold=fit_dynamic_threshold,
                dynamic_feature_available=fit_dynamic_available,
            )

        conformal_q_hat: float | None = None
        if self.use_m5 and self.reliability_model is not None:
            if X_cal is None or y_cal is None:
                raise ValueError("M5 calibration requires X_cal and y_cal.")
            if self.use_m3:
                cal_s = S_cal
                if cal_s is None and self.config.use_m2:
                    cal_s = self.detect_silence(
                        X_cal,
                        y_cal,
                        context=context_cal,
                        sensor_metadata=sensor_metadata_cal,
                        batch_size=self.config.state_training.batch_size,
                    )["dynamic_silence"].float()
                mu_cal, var_cal, _ = self.missingness_aware_state_summary(
                    X_cal,
                    y=y_cal,
                    context=context_cal,
                    M=M_cal,
                    S=cal_s,
                    sensor_metadata=sensor_metadata_cal,
                    batch_size=self.config.state_training.batch_size,
                    include_observation_noise=self.config.reliability.prediction_target == "observation",
                )
                y_cal_vector = _coerce_tensor_1d(y_cal, name="y_cal")
                observed_mask = torch.isfinite(y_cal_vector)
                conformal_q_hat = float(
                    self.reliability_model.fit(
                        mu_cal=mu_cal[observed_mask],
                        var_cal=var_cal[observed_mask],
                        y_cal=y_cal_vector[observed_mask],
                        node_features=(
                            self._relational_node_features(
                                X_cal,
                                y=y_cal,
                                context=context_cal,
                                sensor_metadata=sensor_metadata_cal,
                                batch_size=self.config.state_training.batch_size,
                            )[observed_mask]
                            if self.config.reliability.mode in {"relational_adaptive", "graph_corel"}
                            else None
                        ),
                    ).item()
                )
                self.reliability_model.reset_adaptation()
            else:
                conformal_q_hat = float(
                    self.calibrate_reliability(
                        X_cal,
                        y_cal,
                        context=context_cal,
                        sensor_metadata=sensor_metadata_cal,
                        batch_size=self.config.state_training.batch_size,
                    ).item()
                )

        return PipelineFitSummary(
            state_history=state_history,
            observation_history=observation_history,
            dynamic_silence_threshold=dynamic_threshold,
            missingness_history=missingness_history,
            conformal_quantile=conformal_q_hat,
        )

    def predict_interval(
        self,
        X: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        integrate_missingness: bool = False,
        logit_scale: float = 1.0,
        batch_size: int | None = None,
        y_true: Tensor | None = None,
        neighbor_errors: Tensor | None = None,
        node_features: Tensor | None = None,
        adaptive: bool | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]] | None:
        """Predict conformal intervals for query inputs."""
        if not self.use_m5 or self.reliability_model is None:
            return None
        if integrate_missingness and self.use_m3:
            mu, var, _ = self.missingness_aware_state_summary(
                X,
                y=y,
                context=context,
                M=M,
                S=S,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
                logit_scale=logit_scale,
                include_observation_noise=self.config.reliability.prediction_target == "observation",
            )
        else:
            mu, var = self.predict_state(
                X,
                batch_size=batch_size,
                include_observation_noise=self.config.reliability.prediction_target == "observation",
            )
        if node_features is None and self.config.reliability.mode in {"relational_adaptive", "graph_corel"}:
            node_features = self._relational_node_features(
                X,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
            )
        use_adaptive = (
            self.config.reliability.mode in {"adaptive", "relational_adaptive", "graph_corel"}
            if adaptive is None
            else adaptive
        )
        if use_adaptive:
            if y_true is not None:
                return self.reliability_model.predict_interval_adaptive(
                    mu_test=mu,
                    var_test=var,
                    y_true=y_true,
                    neighbor_errors=neighbor_errors,
                    node_features=node_features,
                )
            return self.reliability_model.predict_interval(
                mu_test=mu,
                var_test=var,
                node_features=node_features,
                epsilon=self.reliability_model.adaptive_epsilon,
            )
        return self.reliability_model.predict_interval(
            mu_test=mu,
            var_test=var,
            node_features=node_features,
        )

    def evaluate_predictions(
        self,
        X: Tensor,
        y: Tensor,
        *,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        integrate_missingness: bool = True,
        logit_scale: float = 1.0,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Evaluate accuracy, proper scores, and calibrated interval quality.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Targets with shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            integrate_missingness: Whether to inflate predictive uncertainty using M3.
            logit_scale: Optional MNAR-strength multiplier.

        Returns:
            Flat metric dictionary for experiment reporting.
        """
        scorer = self.reliability_model or ConformalPredictor(self.config.reliability)
        X = _coerce_tensor_2d(X, name="X")
        y = _coerce_tensor_1d(y, name="y")
        observed_mask = torch.isfinite(y)
        if not observed_mask.any():
            raise ValueError("evaluate_predictions requires at least one observed target.")

        if integrate_missingness and self.use_m3:
            mu, var, p_miss = self.missingness_aware_state_summary(
                X,
                y=y,
                context=context,
                M=M,
                S=S,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
                logit_scale=logit_scale,
                include_observation_noise=self.config.reliability.prediction_target == "observation",
            )
        else:
            mu, var = self.predict_state(
                X,
                batch_size=batch_size,
                include_observation_noise=self.config.reliability.prediction_target == "observation",
            )
            p_miss = None
        silence_features: dict[str, Tensor] | None = None
        if self.config.use_m2 and (
            S is None or (self.use_m5 and self.config.reliability.mode in {"relational_adaptive", "graph_corel"})
        ):
            silence_features = self.detect_silence(
                X,
                y,
                context=context,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
            )
            if S is None:
                S = silence_features["dynamic_silence"].float()
        lower: Tensor | None = None
        upper: Tensor | None = None
        interval_metadata: dict[str, float] | None = None
        if self.use_m5 and self.reliability_model is not None and self.reliability_model.is_calibrated:
            if self.config.reliability.mode == "adaptive":
                self.reliability_model.reset_adaptation()
                lower_obs, upper_obs, interval_metadata = self.reliability_model.predict_interval_adaptive(
                    mu_test=mu[observed_mask],
                    var_test=var[observed_mask],
                    y_true=y[observed_mask],
                )
                lower = torch.zeros_like(mu)
                upper = torch.zeros_like(mu)
                lower[observed_mask] = lower_obs
                upper[observed_mask] = upper_obs
            elif self.config.reliability.mode in {"relational_adaptive", "graph_corel"}:
                self.reliability_model.reset_adaptation()
                node_features = self._relational_node_features(
                    X,
                    y=y,
                    context=context,
                    sensor_metadata=sensor_metadata,
                    silence_features=silence_features,
                    batch_size=batch_size,
                )
                lower_obs, upper_obs, interval_metadata = self.reliability_model.predict_interval_adaptive(
                    mu_test=mu[observed_mask],
                    var_test=var[observed_mask],
                    y_true=y[observed_mask],
                    node_features=node_features[observed_mask],
                )
                lower = torch.zeros_like(mu)
                upper = torch.zeros_like(mu)
                lower[observed_mask] = lower_obs
                upper[observed_mask] = upper_obs
            else:
                lower, upper, interval_metadata = self.reliability_model.predict_interval(
                    mu_test=mu,
                    var_test=var,
                )

        metrics = scorer.evaluate_gaussian_predictions(
            mu=mu[observed_mask],
            var=var[observed_mask],
            y_true=y[observed_mask],
            lower=None if lower is None else lower[observed_mask],
            upper=None if upper is None else upper[observed_mask],
        )
        output = {
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "crps": metrics.crps,
            "log_score": metrics.log_score,
        }
        if metrics.coverage is not None:
            output["coverage"] = metrics.coverage
        if metrics.interval_width is not None:
            output["interval_width"] = metrics.interval_width
        if p_miss is not None:
            output["mean_missingness_proba"] = float(p_miss[observed_mask].mean().item())
        if interval_metadata is not None and "final_epsilon" in interval_metadata:
            output["final_adaptive_epsilon"] = float(interval_metadata["final_epsilon"])
        if interval_metadata is not None and "mean_neighbor_error" in interval_metadata:
            output["mean_neighbor_error"] = float(interval_metadata["mean_neighbor_error"])
        if interval_metadata is not None and "mean_graph_quantile" in interval_metadata:
            output["mean_graph_quantile"] = float(interval_metadata["mean_graph_quantile"])
        return output

    def infer_sensor_health(
        self,
        X: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Tensor] | None:
        """Infer latent sensor-health summaries from the configured M3 module."""
        if not self.use_m3 or self.missingness_model is None:
            return None
        z_mean, z_var = self.predict_state(X, batch_size=batch_size)
        effective_s, effective_residual, effective_threshold, effective_available = (
            self._resolve_dynamic_feature_bundle(
                X,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                S=S,
                dynamic_residual=None,
                dynamic_threshold=None,
                dynamic_feature_available=None,
                batch_size=batch_size,
                mode="prediction",
            )
        )
        with torch.no_grad():
            return self.missingness_model.infer_health_posterior(
                z_mean=z_mean,
                z_var=z_var,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                M=M,
                S=effective_s,
                dynamic_residual=effective_residual,
                dynamic_threshold=effective_threshold,
                dynamic_feature_available=effective_available,
                X=X if self.missingness_model.config.include_x else None,
            )

    def missingness_sensitivity_sweep(
        self,
        X: Tensor,
        *,
        logit_scales: list[float],
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Tensor]:
        """Sweep MNAR-strength multipliers for sensitivity analysis.

        Args:
            X: Input coordinates with shape `[N, D]`.
            logit_scales: Multipliers applied to structural missingness logits.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.
            batch_size: Optional inference batch size.

        Returns:
            Mapping from scale labels to missingness probabilities.
        """
        if not self.use_m3 or self.missingness_model is None:
            raise RuntimeError("M3 is disabled; missingness sensitivity analysis is unavailable.")
        outputs: dict[str, Tensor] = {}
        for scale in logit_scales:
            outputs[f"logit_scale_{scale:.3f}"] = self.predict_missingness(
                X,
                context=context,
                M=M,
                S=S,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
                logit_scale=scale,
            )
        return outputs

    def run_missingness_sensitivity_analysis(
        self,
        X: Tensor,
        y: Tensor,
        *,
        logit_scales: list[float],
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate predictive metrics under varying MNAR-strength assumptions.

        Args:
            X: Evaluation inputs with shape `[N, D]`.
            y: Evaluation targets with shape `[N]`.
            logit_scales: Multipliers applied to structural missingness logits.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.

        Returns:
            Mapping from sensitivity setting to predictive metrics.
        """
        results: dict[str, dict[str, float]] = {}
        for scale in logit_scales:
            key = f"logit_scale_{scale:.3f}"
            results[key] = self.evaluate_predictions(
                X,
                y,
                context=context,
                M=M,
                S=S,
                sensor_metadata=sensor_metadata,
                integrate_missingness=self.use_m3,
                logit_scale=scale,
                batch_size=batch_size,
            )
        return results

    def build_ablation_configs(self) -> dict[str, SilenceAwareIDSConfig]:
        """Build experiment-ready ablation variants from the current config."""
        base = replace(self.config)
        plugin_missingness = replace(base.missingness, inference_strategy="plug_in")
        joint_missingness = replace(base.missingness, inference_strategy="joint_variational")
        sequential_state = replace(base.state_training, training_strategy="sequential")
        joint_state = replace(base.state_training, training_strategy="joint_variational")
        adaptive_reliability = replace(base.reliability, mode="adaptive")
        relational_reliability = replace(base.reliability, mode="relational_adaptive")
        graph_reliability = replace(base.reliability, mode="graph_corel")
        myopic_policy = replace(base.policy, planning_strategy="lazy_greedy")
        rollout_policy = replace(base.policy, planning_strategy="non_myopic_rollout")
        ppo_warm_policy = replace(base.policy, planning_strategy="ppo_warmstart")
        ppo_online_policy = replace(base.policy, planning_strategy="ppo_online")
        return {
            "base_gp_only": replace(
                base,
                use_m2=False,
                use_m3=False,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=False,
                policy=myopic_policy,
                state_training=sequential_state,
            ),
            "gp_plus_dynamic_silence": replace(
                base,
                use_m2=True,
                use_m3=False,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=False,
                policy=myopic_policy,
                state_training=sequential_state,
            ),
            "gp_plus_homogeneous_missingness": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=True,
                sensor_conditional_missingness=False,
                missingness=replace(plugin_missingness, mode="selection"),
                policy=myopic_policy,
                state_training=sequential_state,
            ),
            "gp_plus_sensor_conditional_missingness": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(plugin_missingness, mode="selection"),
                policy=myopic_policy,
                state_training=sequential_state,
            ),
            "gp_plus_joint_variational_missingness": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(joint_missingness, mode="selection"),
                policy=myopic_policy,
                state_training=sequential_state,
            ),
            "gp_plus_joint_jvi_training": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(joint_missingness, mode="selection"),
                policy=myopic_policy,
                state_training=joint_state,
            ),
            "gp_plus_conformal_reliability": replace(
                base,
                use_m2=True,
                use_m3=False,
                use_m5=True,
                homogeneous_missingness=False,
                sensor_conditional_missingness=False,
                policy=myopic_policy,
                state_training=sequential_state,
                reliability=adaptive_reliability,
            ),
            "relational_reliability_baseline": replace(
                base,
                use_m2=True,
                use_m3=False,
                use_m5=True,
                homogeneous_missingness=False,
                sensor_conditional_missingness=False,
                policy=myopic_policy,
                state_training=sequential_state,
                reliability=relational_reliability,
            ),
            "ppo_warmstart_baseline": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=True,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(joint_missingness, mode="selection"),
                policy=ppo_warm_policy,
                state_training=joint_state,
                reliability=graph_reliability,
            ),
            "full_model": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=True,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(joint_missingness, mode="selection"),
                policy=ppo_online_policy,
                state_training=joint_state,
                reliability=graph_reliability,
            ),
            "gp_plus_pattern_mixture_missingness": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=False,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(plugin_missingness, mode="pattern_mixture"),
                policy=myopic_policy,
                state_training=sequential_state,
            ),
            "myopic_policy_baseline": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=True,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(joint_missingness, mode="selection"),
                policy=myopic_policy,
                state_training=joint_state,
                reliability=graph_reliability,
            ),
            "rollout_policy_baseline": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=True,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(joint_missingness, mode="selection"),
                policy=rollout_policy,
                state_training=joint_state,
                reliability=graph_reliability,
            ),
            "variance_policy_baseline": replace(
                base,
                use_m2=True,
                use_m3=True,
                use_m5=True,
                homogeneous_missingness=False,
                sensor_conditional_missingness=True,
                missingness=replace(joint_missingness, mode="selection"),
                policy=replace(myopic_policy, utility_surrogate="variance"),
                state_training=joint_state,
                reliability=graph_reliability,
            ),
        }

    def run_ablation_suite(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_eval: Tensor,
        y_eval: Tensor,
        *,
        missing_indicator_train: Tensor | None = None,
        context_train: Tensor | None = None,
        M_train: Tensor | None = None,
        S_train: Tensor | None = None,
        sensor_metadata_train: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        context_eval: Tensor | None = None,
        M_eval: Tensor | None = None,
        S_eval: Tensor | None = None,
        sensor_metadata_eval: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        X_cal: Tensor | None = None,
        y_cal: Tensor | None = None,
        context_cal: Tensor | None = None,
        M_cal: Tensor | None = None,
        S_cal: Tensor | None = None,
        sensor_metadata_cal: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        candidate_x: Tensor | None = None,
        candidate_cost: Tensor | None = None,
        candidate_context: Tensor | None = None,
        candidate_M: Tensor | None = None,
        candidate_S: Tensor | None = None,
        candidate_sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        budget: float | None = None,
        max_selections: int | None = None,
        variant_names: list[str] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, AblationOutcome]:
        """Run a research-ready ablation suite across configured module variants.

        Args:
            X_train: Training inputs with shape `[N, D]`.
            y_train: Training targets with shape `[N]`.
            X_eval: Evaluation inputs with shape `[M, D]`.
            y_eval: Evaluation targets with shape `[M]`.
            missing_indicator_train: Optional structural missingness labels `[N]`.
            context_train: Optional training context `[N, C]`.
            M_train: Optional training scalar feature `[N]`.
            S_train: Optional training dynamic silence feature `[N]`.
            sensor_metadata_train: Optional training metadata.
            context_eval: Optional evaluation context `[M, C]`.
            M_eval: Optional evaluation scalar feature `[M]`.
            S_eval: Optional evaluation dynamic silence feature `[M]`.
            sensor_metadata_eval: Optional evaluation metadata.
            X_cal: Optional calibration inputs `[K, D]`.
            y_cal: Optional calibration targets `[K]`.
            context_cal: Optional calibration context `[K, C]`.
            M_cal: Optional calibration scalar feature `[K]`.
            S_cal: Optional calibration dynamic silence feature `[K]`.
            sensor_metadata_cal: Optional calibration metadata.
            candidate_x: Optional candidate inputs for policy validation `[P, D]`.
            candidate_cost: Optional candidate sensing costs `[P]`.
            candidate_context: Optional candidate context `[P, C]`.
            candidate_M: Optional candidate scalar feature `[P]`.
            candidate_S: Optional candidate dynamic silence feature `[P]`.
            candidate_sensor_metadata: Optional candidate metadata.
            budget: Optional policy budget.
            max_selections: Optional policy cardinality limit.
            variant_names: Optional subset of ablation variants.

        Returns:
            Mapping from ablation name to structured ablation outcomes.
        """
        variants = self.build_ablation_configs()
        active_names = variant_names or list(variants.keys())
        results: dict[str, AblationOutcome] = {}

        for variant_name in active_names:
            if variant_name not in variants:
                raise KeyError(f"Unknown ablation variant: {variant_name}")
            variant = SilenceAwareIDS(variants[variant_name])
            fit_summary = variant.fit(
                X_train,
                y_train,
                missing_indicator_train=missing_indicator_train,
                context_train=context_train,
                M_train=M_train,
                S_train=S_train,
                sensor_metadata_train=sensor_metadata_train,
                X_cal=X_cal,
                y_cal=y_cal,
                context_cal=context_cal,
                M_cal=M_cal,
                S_cal=S_cal,
                sensor_metadata_cal=sensor_metadata_cal,
            )
            eval_s = S_eval
            if eval_s is None and variant.config.use_m2:
                eval_s = variant.detect_silence(
                    X_eval,
                    y_eval,
                    context=context_eval,
                    sensor_metadata=sensor_metadata_eval,
                    batch_size=batch_size,
                )["dynamic_silence"].float()
            metrics = variant.evaluate_predictions(
                X_eval,
                y_eval,
                context=context_eval,
                M=M_eval,
                S=eval_s,
                sensor_metadata=sensor_metadata_eval,
                integrate_missingness=variant.use_m3,
                batch_size=batch_size,
            )

            selection: dict[str, Tensor | list[int] | float] | None = None
            if candidate_x is not None and candidate_cost is not None:
                selection = variant.select_sensors(
                    candidate_x,
                    candidate_cost,
                    context=candidate_context,
                    M=candidate_M,
                    S=candidate_S,
                    sensor_metadata=candidate_sensor_metadata,
                    budget=budget,
                    max_selections=max_selections,
                    batch_size=batch_size,
                )

            report = variant.ablation_report()
            report["fit_summary"] = {
                "state_epochs": len(fit_summary.state_history.get("loss", [])),
                "observation_steps": (
                    None
                    if fit_summary.observation_history is None
                    else len(fit_summary.observation_history.get("loss", []))
                ),
                "dynamic_silence_threshold": fit_summary.dynamic_silence_threshold,
                "missingness_epochs": (
                    None
                    if fit_summary.missingness_history is None
                    else len(fit_summary.missingness_history.get("loss", []))
                ),
                "conformal_quantile": fit_summary.conformal_quantile,
            }
            results[variant_name] = AblationOutcome(
                variant_name=variant_name,
                metrics=metrics,
                report=report,
                selection=selection,
            )

        return results

    def spawn_ablation_variant(self, variant_name: str) -> "SilenceAwareIDS":
        """Instantiate one named ablation variant."""
        variants = self.build_ablation_configs()
        if variant_name not in variants:
            raise KeyError(f"Unknown ablation variant: {variant_name}")
        return SilenceAwareIDS(variants[variant_name])

    def forward(
        self,
        X: Tensor,
        *,
        y: Tensor | None = None,
        context: Tensor | None = None,
        M: Tensor | None = None,
        S: Tensor | None = None,
        sensor_metadata: SensorMetadataBatch | Mapping[str, Tensor] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Run the end-to-end prediction pipeline.

        Args:
            X: Input coordinates with shape `[N, D]`.
            y: Optional observations with shape `[N]`.
            context: Optional context matrix with shape `[N, C]`.
            M: Optional scalar feature with shape `[N]`.
            S: Optional dynamic silence signal with shape `[N]`.
            sensor_metadata: Optional heterogeneous sensor metadata.

        Returns:
            Structured dictionary with explicit latent and observation-space
            summaries, silence flags, optional missingness probabilities, and
            optional conformal intervals.
        """
        X = _coerce_tensor_2d(X, name="X")
        state_mean, state_var = self.predict_state(X, batch_size=batch_size)
        observation_mean: Tensor | None = None
        observation_var: Tensor | None = None
        needs_observation_summary = self.config.use_m2 or (
            self.use_m5 and self.config.reliability.prediction_target == "observation"
        )
        if needs_observation_summary:
            observation_mean, observation_var = self.predict_state(
                X,
                batch_size=batch_size,
                include_observation_noise=True,
            )
        output: dict[str, Any] = {
            "state_mean": state_mean,
            "state_var": state_var,
            "state_target": "latent",
            "observation_mean": observation_mean,
            "observation_var": observation_var,
            "observation_target": "observation",
            "dynamic_silence": None,
            "diagnosis_embedding": None,
            "sensor_state_probs": None,
            "sensor_state_labels": None,
            "missingness_proba": None,
            "health_latent_mean": None,
            "health_latent_var": None,
            "conformal_interval": None,
            "conformal_interval_target": None,
        }
        silence_features: dict[str, Tensor] | None = None

        if y is not None and self.config.use_m2:
            silence_features = self.detect_silence(
                X,
                y,
                context=context,
                sensor_metadata=sensor_metadata,
                batch_size=batch_size,
            )
            output["dynamic_silence"] = silence_features["dynamic_silence"]
            output["diagnosis_embedding"] = silence_features["diagnosis_embedding"]
            output["sensor_state_probs"] = silence_features["sensor_state_probs"]
            output["sensor_state_labels"] = silence_features["sensor_state_labels"]
            if S is None:
                S = silence_features["dynamic_silence"].float()

        if self.use_m3 and self.missingness_model is not None:
            effective_s, effective_residual, effective_threshold, effective_available = (
                self._resolve_dynamic_feature_bundle(
                    X,
                    y=y,
                    context=context,
                    sensor_metadata=sensor_metadata,
                    S=S,
                    dynamic_residual=None if silence_features is None else silence_features["residuals"],
                    dynamic_threshold=None if silence_features is None else silence_features["threshold"],
                    dynamic_feature_available=None if silence_features is None else silence_features["available"],
                    batch_size=batch_size,
                    mode="prediction",
                )
            )
            output["missingness_proba"] = self.missingness_model.predict_proba(
                z_mean=state_mean,
                z_var=state_var,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                M=M,
                S=effective_s,
                dynamic_residual=effective_residual,
                dynamic_threshold=effective_threshold,
                dynamic_feature_available=effective_available,
                X=X if self.missingness_model.config.include_x else None,
            )
            health_stats = self.missingness_model.infer_health_posterior(
                z_mean=state_mean,
                z_var=state_var,
                y=y,
                context=context,
                sensor_metadata=sensor_metadata,
                M=M,
                S=effective_s,
                dynamic_residual=effective_residual,
                dynamic_threshold=effective_threshold,
                dynamic_feature_available=effective_available,
                X=X if self.missingness_model.config.include_x else None,
            )
            output["health_latent_mean"] = health_stats["health_mean"]
            output["health_latent_var"] = health_stats["health_var"]

        if self.use_m5 and self.reliability_model is not None and self.reliability_model.is_calibrated:
            interval = self.predict_interval(
                X,
                y=y,
                context=context,
                M=M,
                S=S,
                sensor_metadata=sensor_metadata,
                integrate_missingness=self.use_m3,
                batch_size=batch_size,
            )
            if interval is not None:
                lower, upper, metadata = interval
                output["conformal_interval"] = {"lower": lower, "upper": upper, "metadata": metadata}
                output["conformal_interval_target"] = metadata.get("prediction_target")

        return output

    def ablation_report(self) -> dict[str, Any]:
        """Return the current ablation setting in experiment-friendly form."""
        missingness_assumption = "disabled"
        if self.use_m3:
            missingness_assumption = (
                "homogeneous" if self.config.homogeneous_missingness else "sensor_conditional"
            )
        base_gp_only = (not self.config.use_m2) and (not self.use_m3) and (not self.use_m5)
        gp_plus_dynamic_silence = self.config.use_m2 and (not self.use_m3) and (not self.use_m5)
        gp_plus_homogeneous_missingness = (
            self.config.use_m2
            and self.use_m3
            and (not self.use_m5)
            and self.config.homogeneous_missingness
            and self.config.missingness.mode == "selection"
            and self.config.policy.utility_surrogate != "variance"
        )
        gp_plus_sensor_conditional_missingness = (
            self.config.use_m2
            and self.use_m3
            and (not self.use_m5)
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "plug_in"
            and self.config.policy.utility_surrogate != "variance"
        )
        gp_plus_joint_variational_missingness = (
            self.config.use_m2
            and self.use_m3
            and (not self.use_m5)
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "joint_variational"
            and self.config.state_training.training_strategy == "sequential"
            and self.config.policy.utility_surrogate != "variance"
        )
        gp_plus_joint_jvi_training = (
            self.config.use_m2
            and self.use_m3
            and (not self.use_m5)
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "joint_variational"
            and self.config.state_training.training_strategy == "joint_variational"
            and self.config.policy.utility_surrogate != "variance"
        )
        gp_plus_pattern_mixture_missingness = (
            self.config.use_m2
            and self.use_m3
            and (not self.use_m5)
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "pattern_mixture"
            and self.config.missingness.inference_strategy == "plug_in"
            and self.config.policy.utility_surrogate != "variance"
        )
        gp_plus_conformal_reliability = (
            self.config.use_m2
            and (not self.use_m3)
            and self.use_m5
            and self.config.reliability.mode == "adaptive"
        )
        relational_reliability_baseline = (
            self.config.use_m2
            and (not self.use_m3)
            and self.use_m5
            and self.config.reliability.mode == "relational_adaptive"
        )
        myopic_policy_baseline = (
            self.config.use_m2
            and self.use_m3
            and self.use_m5
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "joint_variational"
            and self.config.state_training.training_strategy == "joint_variational"
            and self.config.policy.utility_surrogate != "variance"
            and self.config.policy.planning_strategy == "lazy_greedy"
        )
        ppo_warmstart_baseline = (
            self.config.use_m2
            and self.use_m3
            and self.use_m5
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "joint_variational"
            and self.config.state_training.training_strategy == "joint_variational"
            and self.config.policy.utility_surrogate != "variance"
            and self.config.policy.planning_strategy == "ppo_warmstart"
        )
        rollout_policy_baseline = (
            self.config.use_m2
            and self.use_m3
            and self.use_m5
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "joint_variational"
            and self.config.state_training.training_strategy == "joint_variational"
            and self.config.policy.utility_surrogate != "variance"
            and self.config.policy.planning_strategy == "non_myopic_rollout"
        )
        variance_policy_baseline = (
            self.config.use_m2
            and self.use_m3
            and self.use_m5
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "joint_variational"
            and self.config.state_training.training_strategy == "joint_variational"
            and self.config.policy.utility_surrogate == "variance"
            and self.config.policy.planning_strategy == "lazy_greedy"
        )
        full_model = (
            self.config.use_m2
            and self.use_m3
            and self.use_m5
            and self.config.sensor_conditional_missingness
            and self.config.missingness.mode == "selection"
            and self.config.missingness.inference_strategy == "joint_variational"
            and self.config.state_training.training_strategy == "joint_variational"
            and self.config.policy.utility_surrogate != "variance"
            and self.config.policy.planning_strategy == "ppo_online"
        )
        return {
            "use_m2": self.config.use_m2,
            "use_m3": self.use_m3,
            "use_m5": self.use_m5,
            "dynamic_silence": self.config.use_m2,
            "missingness_assumption": missingness_assumption,
            "missingness_mode": self.config.missingness.mode,
            "missingness_inference_strategy": self.config.missingness.inference_strategy,
            "diagnosis_mode": self.config.observation.diagnosis_mode,
            "policy_surrogate": self.config.policy.utility_surrogate,
            "policy_planning_strategy": self.config.policy.planning_strategy,
            "state_training_strategy": self.config.state_training.training_strategy,
            "policy_objective": (
                "environment_trained_ppo_actor_critic"
                if self.config.policy.planning_strategy == "ppo_online"
                else (
                    "ppo_warmstarted_actor_critic_surrogate"
                    if self.config.policy.planning_strategy == "ppo_warmstart"
                    else (
                        "discounted_non_myopic_rollout_surrogate"
                        if self.config.policy.planning_strategy == "non_myopic_rollout"
                        else "myopic_gaussian_information_surrogate"
                    )
                )
            ),
            "missingness_integration": (
                "joint_elbo_sparse_gp_with_latent_adapter"
                if (
                    self.config.missingness.inference_strategy == "joint_variational"
                    and self.config.state_training.training_strategy == "joint_variational"
                )
                else (
                    "joint_variational_latent_adapter"
                    if self.config.missingness.inference_strategy == "joint_variational"
                    else "plug_in_latent_summary"
                )
            ),
            "diagnosis_representation": "pi_ssd_embedding" if self.config.observation.use_pi_ssd else "threshold_only",
            "diagnosis_temporal_model": "dbn" if self.config.observation.use_dbn else "heuristic_sequence_score",
            "diagnosis_curriculum": (
                "linear_corruption_curriculum" if self.config.observation.use_pi_ssd else "disabled"
            ),
            "diagnosis_latent_dynamics": (
                "latent_ode_lite" if self.config.observation.use_latent_ode else "disabled"
            ),
            "observation_uncertainty_for_silence": self.config.observation.use_observation_noise,
            "missingness_sensor_health_latent": self.config.missingness.use_sensor_health_latent,
            "reliability_prediction_target": self.config.reliability.prediction_target,
            "reliability_mode": self.config.reliability.mode,
            "reliability_relational": self.config.reliability.mode == "relational_adaptive",
            "reliability_graph_corel": self.config.reliability.mode == "graph_corel",
            "reliability_graph_message_passing_steps": self.config.reliability.graph_message_passing_steps,
            "dynamic_feature_policy_mode": {
                "training": self.config.missingness.use_dynamic_features_for_training,
                "prediction": self.config.missingness.use_dynamic_features_for_prediction,
                "policy": self.config.missingness.use_dynamic_features_for_policy,
            },
            "comparison_grid": {
                "base_gp_only": base_gp_only,
                "gp_plus_dynamic_silence": gp_plus_dynamic_silence,
                "gp_plus_homogeneous_missingness": gp_plus_homogeneous_missingness,
                "gp_plus_sensor_conditional_missingness": gp_plus_sensor_conditional_missingness,
                "gp_plus_joint_variational_missingness": gp_plus_joint_variational_missingness,
                "gp_plus_joint_jvi_training": gp_plus_joint_jvi_training,
                "gp_plus_pattern_mixture_missingness": gp_plus_pattern_mixture_missingness,
                "pattern_mixture_missingness": self.use_m3 and self.config.missingness.mode == "pattern_mixture",
                "gp_plus_conformal_reliability": gp_plus_conformal_reliability,
                "relational_reliability_baseline": relational_reliability_baseline,
                "myopic_policy_baseline": myopic_policy_baseline,
                "ppo_warmstart_baseline": ppo_warmstart_baseline,
                "rollout_policy_baseline": rollout_policy_baseline,
                "variance_policy_baseline": variance_policy_baseline,
                "full_model": full_model,
            },
        }
