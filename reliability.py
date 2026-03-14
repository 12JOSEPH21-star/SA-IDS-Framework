from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ConformalConfig:
    """Configuration for normalized conformal prediction."""

    epsilon: float = 0.1
    delta: float = 1e-3
    variance_floor: float = 1e-6
    prediction_target: str = "observation"
    mode: str = "graph_corel"
    adaptation_rate: float = 0.05
    min_epsilon: float = 0.01
    max_epsilon: float = 0.5
    relational_neighbor_weight: float = 0.0
    relational_temperature: float = 1.0
    relational_decay: float = 0.95
    max_relational_memory: int = 2048
    graph_k_neighbors: int = 8
    graph_hidden_dim: int = 32
    graph_learning_rate: float = 1e-3
    graph_training_steps: int = 100
    graph_temperature: float = 1.0
    graph_score_weight: float = 0.5
    graph_covariance_weight: float = 0.5
    graph_message_passing_steps: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must lie in (0, 1).")
        if self.delta <= 0.0:
            raise ValueError("delta must be positive.")
        if self.prediction_target not in {"latent", "observation"}:
            raise ValueError("prediction_target must be either 'latent' or 'observation'.")
        if self.mode not in {"split", "adaptive", "relational_adaptive", "graph_corel"}:
            raise ValueError("mode must be 'split', 'adaptive', 'relational_adaptive', or 'graph_corel'.")
        if self.adaptation_rate < 0.0:
            raise ValueError("adaptation_rate must be non-negative.")
        if not 0.0 < self.min_epsilon < 1.0:
            raise ValueError("min_epsilon must lie in (0, 1).")
        if not 0.0 < self.max_epsilon < 1.0:
            raise ValueError("max_epsilon must lie in (0, 1).")
        if self.min_epsilon > self.max_epsilon:
            raise ValueError("min_epsilon must not exceed max_epsilon.")
        if not 0.0 <= self.relational_neighbor_weight <= 1.0:
            raise ValueError("relational_neighbor_weight must lie in [0, 1].")
        if self.relational_temperature <= 0.0:
            raise ValueError("relational_temperature must be positive.")
        if not 0.0 < self.relational_decay <= 1.0:
            raise ValueError("relational_decay must lie in (0, 1].")
        if self.max_relational_memory <= 0:
            raise ValueError("max_relational_memory must be positive.")
        if self.graph_k_neighbors <= 0:
            raise ValueError("graph_k_neighbors must be positive.")
        if self.graph_hidden_dim <= 0:
            raise ValueError("graph_hidden_dim must be positive.")
        if self.graph_learning_rate <= 0.0:
            raise ValueError("graph_learning_rate must be positive.")
        if self.graph_training_steps <= 0:
            raise ValueError("graph_training_steps must be positive.")
        if self.graph_temperature <= 0.0:
            raise ValueError("graph_temperature must be positive.")
        if not 0.0 <= self.graph_score_weight <= 1.0:
            raise ValueError("graph_score_weight must lie in [0, 1].")
        if not 0.0 <= self.graph_covariance_weight <= 1.0:
            raise ValueError("graph_covariance_weight must lie in [0, 1].")
        if self.graph_message_passing_steps < 0:
            raise ValueError("graph_message_passing_steps must be non-negative.")


@dataclass
class PredictiveMetrics:
    """Summary of predictive accuracy, sharpness, and coverage."""

    rmse: float
    mae: float
    crps: float
    log_score: float
    coverage: float | None = None
    interval_width: float | None = None


class _GraphCoRelAdapter(nn.Module):
    """STGNN-lite adapter with iterative graph message passing for local quantiles."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        k_neighbors: int,
        temperature: float,
        message_passing_steps: int,
    ) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.temperature = temperature
        self.message_passing_steps = message_passing_steps
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.message_update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _message_pass(self, node_features: Tensor, hidden: Tensor) -> Tensor:
        """Run one weighted kNN message-passing step over features `[N, F]`."""
        if node_features.shape[0] <= 1 or self.message_passing_steps == 0:
            return hidden
        distances = torch.cdist(node_features, node_features, p=2)
        diagonal = torch.arange(node_features.shape[0], device=node_features.device)
        distances[diagonal, diagonal] = float("inf")
        k = min(self.k_neighbors, node_features.shape[0] - 1)
        if k <= 0:
            return hidden
        neighbor_distance, neighbor_index = torch.topk(distances, k=k, dim=-1, largest=False)
        weights = torch.softmax(-neighbor_distance / self.temperature, dim=-1)
        neighbor_hidden = hidden[neighbor_index.reshape(-1)].view(
            node_features.shape[0],
            k,
            hidden.shape[-1],
        )
        aggregated_hidden = torch.sum(weights.unsqueeze(-1) * neighbor_hidden, dim=1)
        update_input = torch.cat([hidden, aggregated_hidden, node_features], dim=-1)
        return self.layer_norm(hidden + self.message_update(update_input))

    def forward(self, x: Tensor) -> Tensor:
        """Return positive local quantile estimates for features `[N, F]`."""
        hidden = self.input_projection(x)
        for _ in range(self.message_passing_steps):
            hidden = self._message_pass(x, hidden)
        return F.softplus(self.readout(torch.cat([hidden, x], dim=-1)).squeeze(-1))


class ConformalPredictor:
    """Variance-normalized conformal predictor for finite-sample reliability."""

    def __init__(self, config: ConformalConfig | None = None) -> None:
        self.config = config or ConformalConfig()
        self._q_hat: Tensor | None = None
        self._n_calibration: int = 0
        self._calibration_scores: Tensor | None = None
        self._adaptive_epsilon: float = self.config.epsilon
        self._relational_feature_bank: list[Tensor] = []
        self._relational_error_bank: list[float] = []
        self._graph_adapter: _GraphCoRelAdapter | None = None
        self._graph_optimizer: torch.optim.Optimizer | None = None
        self._graph_feature_dim: int | None = None
        self._graph_calibration_features: Tensor | None = None
        self._graph_calibration_scores: Tensor | None = None

    @property
    def is_calibrated(self) -> bool:
        """Return whether the conformal predictor has been calibrated."""
        return self._q_hat is not None

    @property
    def adaptive_epsilon(self) -> float:
        """Return the current adaptive miscoverage budget."""
        return float(self._adaptive_epsilon)

    def nonconformity_scores(self, mu: Tensor, var: Tensor, y_true: Tensor) -> Tensor:
        """Compute normalized nonconformity scores with shape `[N]`."""
        mu = mu.view(-1)
        var = torch.clamp(var.view(-1), min=self.config.variance_floor)
        y_true = y_true.view(-1)
        scale = torch.sqrt(var) + self.config.delta
        return torch.abs(y_true - mu) / scale

    def quantile(self, scores: Tensor, epsilon: float | None = None) -> Tensor:
        """Compute the finite-sample conformal quantile."""
        scores = scores.view(-1)
        if scores.numel() == 0:
            raise ValueError("scores must contain at least one value.")
        target_epsilon = self.config.epsilon if epsilon is None else epsilon
        sorted_scores, _ = torch.sort(scores)
        rank = math.ceil((sorted_scores.numel() + 1) * (1.0 - target_epsilon))
        rank = max(1, min(rank, sorted_scores.numel()))
        return sorted_scores[rank - 1]

    def fit(
        self,
        mu_cal: Tensor,
        var_cal: Tensor,
        y_cal: Tensor,
        *,
        node_features: Tensor | None = None,
    ) -> Tensor:
        """Fit the conformal predictor on a calibration split."""
        scores = self.nonconformity_scores(mu=mu_cal, var=var_cal, y_true=y_cal)
        self._calibration_scores = scores.detach()
        self._q_hat = self.quantile(scores)
        self._n_calibration = int(scores.numel())
        if self.config.mode == "graph_corel":
            if node_features is None:
                raise ValueError("graph_corel mode requires node_features during calibration.")
            self._fit_graph_corel(node_features.detach(), scores.detach())
        self.reset_adaptation()
        return self._q_hat

    def reset_adaptation(self, epsilon: float | None = None) -> float:
        """Reset the adaptive miscoverage level used by ACI-style updates."""
        value = self.config.epsilon if epsilon is None else epsilon
        clamped = max(self.config.min_epsilon, min(self.config.max_epsilon, float(value)))
        self._adaptive_epsilon = clamped
        self._relational_feature_bank = []
        self._relational_error_bank = []
        return clamped

    def _ensure_graph_adapter(self, input_dim: int, *, device: torch.device, dtype: torch.dtype) -> None:
        """Create or reinitialize the graph quantile adapter for the current feature width."""
        if self._graph_adapter is not None and self._graph_feature_dim == input_dim:
            self._graph_adapter = self._graph_adapter.to(device=device, dtype=dtype)
            return
        self._graph_adapter = _GraphCoRelAdapter(
            input_dim=input_dim,
            hidden_dim=self.config.graph_hidden_dim,
            k_neighbors=self.config.graph_k_neighbors,
            temperature=self.config.graph_temperature,
            message_passing_steps=self.config.graph_message_passing_steps,
        ).to(
            device=device,
            dtype=dtype,
        )
        self._graph_optimizer = torch.optim.Adam(
            self._graph_adapter.parameters(),
            lr=self.config.graph_learning_rate,
        )
        self._graph_feature_dim = input_dim

    def _graph_neighbor_indices(
        self,
        query_features: Tensor,
        reference_features: Tensor,
        *,
        exclude_self: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return top-k neighbor indices and attention weights."""
        distances = torch.cdist(query_features, reference_features, p=2)
        if exclude_self and query_features.shape[0] == reference_features.shape[0]:
            diagonal = torch.arange(query_features.shape[0], device=query_features.device)
            distances[diagonal, diagonal] = float("inf")
        k = min(self.config.graph_k_neighbors, reference_features.shape[0])
        neighbor_distance, neighbor_index = torch.topk(distances, k=k, dim=-1, largest=False)
        weights = torch.softmax(-neighbor_distance / self.config.graph_temperature, dim=-1)
        return neighbor_index, weights

    def _graph_aggregate_features(
        self,
        query_features: Tensor,
        reference_features: Tensor,
        reference_scores: Tensor,
        *,
        exclude_self: bool = False,
    ) -> Tensor:
        """Build graph-aware node features for local quantile prediction."""
        neighbor_index, weights = self._graph_neighbor_indices(
            query_features,
            reference_features,
            exclude_self=exclude_self,
        )
        flat_index = neighbor_index.reshape(-1)
        neighbor_features = reference_features[flat_index].view(
            query_features.shape[0],
            neighbor_index.shape[1],
            reference_features.shape[-1],
        )
        neighbor_scores = reference_scores[flat_index].view(query_features.shape[0], neighbor_index.shape[1])
        weighted_neighbor_features = torch.sum(weights.unsqueeze(-1) * neighbor_features, dim=1)
        weighted_neighbor_score = torch.sum(weights * neighbor_scores, dim=1, keepdim=True)
        score_variance = torch.sum(
            weights * (neighbor_scores - weighted_neighbor_score).pow(2),
            dim=1,
            keepdim=True,
        )
        covariance_proxy = torch.exp(-torch.sum((query_features - weighted_neighbor_features).pow(2), dim=-1, keepdim=True))
        return torch.cat(
            [
                query_features,
                weighted_neighbor_features,
                self.config.graph_score_weight * weighted_neighbor_score,
                self.config.graph_score_weight * score_variance,
                self.config.graph_covariance_weight * covariance_proxy,
            ],
            dim=-1,
        )

    def _fit_graph_corel(self, node_features: Tensor, scores: Tensor) -> None:
        """Fit the STGNN-lite adapter on calibration nonconformity scores."""
        if node_features.ndim != 2:
            raise ValueError(f"node_features must have shape [N, F], received {tuple(node_features.shape)}.")
        if node_features.shape[0] != scores.shape[0]:
            raise ValueError("node_features and scores must share the same leading dimension.")
        graph_features = self._graph_aggregate_features(
            node_features,
            node_features,
            scores,
            exclude_self=True,
        )
        self._ensure_graph_adapter(graph_features.shape[-1], device=graph_features.device, dtype=graph_features.dtype)
        if self._graph_adapter is None or self._graph_optimizer is None:
            raise RuntimeError("Graph CoRel adapter initialization failed.")
        self._graph_adapter.train()
        for _ in range(self.config.graph_training_steps):
            prediction = self._graph_adapter(graph_features)
            loss = F.huber_loss(prediction, scores, reduction="mean")
            self._graph_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._graph_adapter.parameters(), max_norm=10.0)
            self._graph_optimizer.step()
        self._graph_adapter.eval()
        self._graph_calibration_features = node_features.detach()
        self._graph_calibration_scores = scores.detach()

    def _predict_graph_quantiles(
        self,
        node_features: Tensor,
        *,
        epsilon: float,
    ) -> Tensor:
        """Predict node-specific CoRel quantiles for features `[N, F]`."""
        if (
            self._graph_adapter is None
            or self._graph_calibration_features is None
            or self._graph_calibration_scores is None
        ):
            raise RuntimeError("Graph CoRel adapter is unavailable; call fit() with node_features first.")
        base_quantile = self.quantile(self._calibration_scores, epsilon=epsilon)
        graph_features = self._graph_aggregate_features(
            node_features,
            self._graph_calibration_features.to(device=node_features.device, dtype=node_features.dtype),
            self._graph_calibration_scores.to(device=node_features.device, dtype=node_features.dtype),
            exclude_self=False,
        )
        self._graph_adapter.eval()
        with torch.no_grad():
            local_quantile = self._graph_adapter(graph_features)
        return (1.0 - self.config.graph_score_weight) * base_quantile + self.config.graph_score_weight * local_quantile

    def _relational_neighbor_error(self, node_feature: Tensor | None) -> float | None:
        """Estimate neighbor error from stored relational memory."""
        if (
            node_feature is None
            or not self._relational_feature_bank
            or self.config.relational_neighbor_weight <= 0.0
        ):
            return None
        query = node_feature.view(-1)
        bank = torch.stack(self._relational_feature_bank).to(device=query.device, dtype=query.dtype)
        distances = torch.cdist(query.unsqueeze(0), bank, p=2).squeeze(0)
        weights = torch.softmax(-distances / self.config.relational_temperature, dim=0)
        error_bank = torch.tensor(
            self._relational_error_bank,
            device=query.device,
            dtype=query.dtype,
        )
        return float(torch.sum(weights * error_bank).item())

    def _update_relational_memory(self, node_feature: Tensor | None, error: float) -> None:
        """Append one node/error observation to the relational memory bank."""
        if node_feature is None:
            return
        decayed = [value * self.config.relational_decay for value in self._relational_error_bank]
        self._relational_error_bank = decayed
        self._relational_feature_bank.append(node_feature.detach().view(-1).cpu())
        self._relational_error_bank.append(float(error))
        if len(self._relational_feature_bank) > self.config.max_relational_memory:
            overflow = len(self._relational_feature_bank) - self.config.max_relational_memory
            self._relational_feature_bank = self._relational_feature_bank[overflow:]
            self._relational_error_bank = self._relational_error_bank[overflow:]

    def update_adaptive_epsilon(
        self,
        error: Tensor | float,
        *,
        neighbor_error: Tensor | float | None = None,
    ) -> float:
        """Update the adaptive miscoverage budget using online errors."""
        error_value = float(error.item()) if isinstance(error, Tensor) else float(error)
        blended_error = error_value
        if neighbor_error is not None:
            neighbor_value = (
                float(neighbor_error.item()) if isinstance(neighbor_error, Tensor) else float(neighbor_error)
            )
            weight = self.config.relational_neighbor_weight
            blended_error = (1.0 - weight) * error_value + weight * neighbor_value
        updated = self._adaptive_epsilon + self.config.adaptation_rate * (
            blended_error - self._adaptive_epsilon
        )
        self._adaptive_epsilon = max(self.config.min_epsilon, min(self.config.max_epsilon, updated))
        return self._adaptive_epsilon

    def predict_interval(
        self,
        mu_test: Tensor,
        var_test: Tensor,
        *,
        node_features: Tensor | None = None,
        q_hat: Tensor | None = None,
        epsilon: float | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Predict calibrated intervals with shape `[N]`."""
        mu_test = mu_test.view(-1)
        var_test = torch.clamp(var_test.view(-1), min=self.config.variance_floor)
        if self.config.mode == "graph_corel":
            if node_features is None:
                raise ValueError("graph_corel mode requires node_features for predict_interval.")
            target_epsilon = self.config.epsilon if epsilon is None else epsilon
            quantile = self._predict_graph_quantiles(
                node_features.view(mu_test.shape[0], -1),
                epsilon=target_epsilon,
            )
            q_hat_value = float(torch.mean(quantile).item())
        else:
            if q_hat is not None:
                quantile = q_hat
            elif epsilon is not None:
                if self._calibration_scores is None:
                    raise RuntimeError("Calibration scores are unavailable; call fit() before predict_interval.")
                quantile = self.quantile(self._calibration_scores, epsilon=epsilon)
            else:
                quantile = self._q_hat
            if quantile is None:
                raise RuntimeError("ConformalPredictor must be calibrated before calling predict_interval.")
            q_hat_value = float(quantile.item())

        scale = torch.sqrt(var_test) + self.config.delta
        radius = quantile * scale
        lower = mu_test - radius
        upper = mu_test + radius
        metadata = {
            "q_hat": q_hat_value,
            "epsilon": float(self.config.epsilon if epsilon is None else epsilon),
            "n_calibration": float(self._n_calibration),
            "prediction_target": self.config.prediction_target,
            "mode": self.config.mode,
            "graph_message_passing_steps": float(self.config.graph_message_passing_steps),
        }
        return lower, upper, metadata

    def predict_interval_adaptive(
        self,
        mu_test: Tensor,
        var_test: Tensor,
        *,
        y_true: Tensor | None = None,
        neighbor_errors: Tensor | None = None,
        node_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Predict intervals with ACI-style online epsilon updates.

        If `y_true` is provided, the interval sequence is generated one step at a
        time and `epsilon` is updated after each realized error. Without `y_true`,
        the current adaptive epsilon is used without updating state.
        """
        mu_test = mu_test.view(-1)
        var_test = var_test.view(-1)
        if y_true is None:
            return self.predict_interval(
                mu_test=mu_test,
                var_test=var_test,
                node_features=node_features,
                epsilon=self._adaptive_epsilon,
            )
        if self._calibration_scores is None:
            raise RuntimeError("Calibration scores are unavailable; call fit() before predict_interval_adaptive.")

        y_true = y_true.view(-1)
        lower = torch.empty_like(mu_test)
        upper = torch.empty_like(mu_test)
        initial_epsilon = self._adaptive_epsilon
        epsilon_values: list[float] = []
        relational_values: list[float] = []
        graph_quantile_values: list[float] = []
        for index in range(mu_test.numel()):
            epsilon = self._adaptive_epsilon
            step_lower, step_upper, step_metadata = self.predict_interval(
                mu_test=mu_test[index : index + 1],
                var_test=var_test[index : index + 1],
                node_features=None if node_features is None else node_features[index : index + 1],
                epsilon=epsilon,
            )
            lower[index] = step_lower[0]
            upper[index] = step_upper[0]
            epsilon_values.append(epsilon)
            if self.config.mode == "graph_corel":
                graph_quantile_values.append(float(step_metadata["q_hat"]))
            covered = bool(
                ((y_true[index] >= lower[index]) & (y_true[index] <= upper[index])).item()
            )
            error = float(not covered)
            neighbor_error = None
            if neighbor_errors is not None:
                neighbor_error = neighbor_errors.view(-1)[index]
            elif self.config.mode == "relational_adaptive":
                feature = None if node_features is None else node_features[index]
                neighbor_error = self._relational_neighbor_error(feature)
                if neighbor_error is not None:
                    relational_values.append(float(neighbor_error))
            self.update_adaptive_epsilon(error, neighbor_error=neighbor_error)
            if self.config.mode == "relational_adaptive":
                feature = None if node_features is None else node_features[index]
                self._update_relational_memory(feature, error)
        metadata = {
            "initial_epsilon": float(initial_epsilon),
            "final_epsilon": float(self._adaptive_epsilon),
            "mean_epsilon": float(sum(epsilon_values) / max(len(epsilon_values), 1)),
            "mean_neighbor_error": float(sum(relational_values) / max(len(relational_values), 1))
            if relational_values
            else 0.0,
            "mean_graph_quantile": float(sum(graph_quantile_values) / max(len(graph_quantile_values), 1))
            if self.config.mode == "graph_corel" and graph_quantile_values
            else 0.0,
            "n_calibration": float(self._n_calibration),
            "prediction_target": self.config.prediction_target,
            "mode": self.config.mode,
            "graph_message_passing_steps": float(self.config.graph_message_passing_steps),
        }
        return lower, upper, metadata

    def empirical_coverage(self, y_true: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
        """Compute empirical interval coverage."""
        y_true = y_true.view(-1)
        lower = lower.view(-1)
        upper = upper.view(-1)
        return ((y_true >= lower) & (y_true <= upper)).float().mean()

    def gaussian_log_score(self, mu: Tensor, var: Tensor, y_true: Tensor) -> Tensor:
        """Compute Gaussian log-score values with shape `[N]`."""
        mu = mu.view(-1)
        var = torch.clamp(var.view(-1), min=self.config.variance_floor)
        y_true = y_true.view(-1)
        squared_error = (y_true - mu).pow(2)
        normalizer = var.new_tensor(2.0 * math.pi)
        return 0.5 * (torch.log(normalizer * var) + squared_error / var)

    def gaussian_crps(self, mu: Tensor, var: Tensor, y_true: Tensor) -> Tensor:
        """Compute Gaussian CRPS values with shape `[N]`."""
        mu = mu.view(-1)
        var = torch.clamp(var.view(-1), min=self.config.variance_floor)
        y_true = y_true.view(-1)
        sigma = torch.sqrt(var)
        standardized = (y_true - mu) / sigma
        normal_pdf = torch.exp(-0.5 * standardized.pow(2)) / math.sqrt(2.0 * math.pi)
        normal_cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        return sigma * (
            standardized * (2.0 * normal_cdf - 1.0)
            + 2.0 * normal_pdf
            - 1.0 / math.sqrt(math.pi)
        )

    def evaluate_gaussian_predictions(
        self,
        mu: Tensor,
        var: Tensor,
        y_true: Tensor,
        *,
        lower: Tensor | None = None,
        upper: Tensor | None = None,
    ) -> PredictiveMetrics:
        """Evaluate Gaussian predictions using proper scoring rules and coverage.

        Args:
            mu: Predictive means with shape `[N]`.
            var: Predictive variances with shape `[N]`.
            y_true: Targets with shape `[N]`.
            lower: Optional lower interval bounds with shape `[N]`.
            upper: Optional upper interval bounds with shape `[N]`.

        Returns:
            Aggregated predictive metrics.
        """
        mu = mu.view(-1)
        var = torch.clamp(var.view(-1), min=self.config.variance_floor)
        y_true = y_true.view(-1)
        rmse = torch.sqrt(torch.mean((y_true - mu).pow(2)))
        mae = torch.mean(torch.abs(y_true - mu))
        crps = torch.mean(self.gaussian_crps(mu=mu, var=var, y_true=y_true))
        log_score = torch.mean(self.gaussian_log_score(mu=mu, var=var, y_true=y_true))

        coverage_value: float | None = None
        width_value: float | None = None
        if lower is not None and upper is not None:
            lower = lower.view(-1)
            upper = upper.view(-1)
            coverage_value = float(self.empirical_coverage(y_true=y_true, lower=lower, upper=upper).item())
            width_value = float(torch.mean(upper - lower).item())

        return PredictiveMetrics(
            rmse=float(rmse.item()),
            mae=float(mae.item()),
            crps=float(crps.item()),
            log_score=float(log_score.item()),
            coverage=coverage_value,
            interval_width=width_value,
        )
