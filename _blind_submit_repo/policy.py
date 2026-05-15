from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LOGGER = logging.getLogger(__name__)


@dataclass
class InformationPolicyConfig:
    """Configuration for scalable information-driven sensing policies."""

    utility_surrogate: str = "mi_proxy"
    planning_strategy: str = "ppo_warmstart"
    selection_mode: str = "budget"
    budget: float | None = None
    cost_weight: float = 0.0
    lengthscale: float = 1.0
    spatial_distance_dims: tuple[int, ...] = ()
    observation_noise: float = 1e-2
    redundancy_strength: float = 0.95
    variance_floor: float = 1e-6
    novelty_floor: float = 1e-3
    ratio_epsilon: float = 1e-6
    mutual_information_scale: float = 0.5
    route_distance_weight: float = 0.0
    planning_horizon: int = 3
    future_discount: float = 0.8
    lookahead_strength: float = 0.5
    future_context_index: int | None = None
    future_volatility_weight: float = 0.25
    ppo_hidden_dim: int = 64
    ppo_epochs: int = 20
    ppo_learning_rate: float = 1e-3
    ppo_clip_epsilon: float = 0.2
    ppo_entropy_weight: float = 1e-3
    ppo_value_weight: float = 0.5
    ppo_warmstart_weight: float = 0.5
    ppo_policy_weight: float = 0.25
    ppo_temperature: float = 1.0
    ppo_max_candidates: int = 2048
    ppo_rollout_episodes: int = 8
    max_selections: int | None = None

    def __post_init__(self) -> None:
        self.spatial_distance_dims = tuple(int(dim) for dim in self.spatial_distance_dims)
        valid_surrogates = {"mi_proxy", "variance"}
        valid_planners = {"lazy_greedy", "non_myopic_rollout", "ppo_warmstart", "ppo_online"}
        valid_modes = {"budget", "penalized", "ratio"}
        if self.utility_surrogate not in valid_surrogates:
            raise ValueError(f"utility_surrogate must be one of {sorted(valid_surrogates)}.")
        if self.planning_strategy not in valid_planners:
            raise ValueError(f"planning_strategy must be one of {sorted(valid_planners)}.")
        if self.selection_mode not in valid_modes:
            raise ValueError(f"selection_mode must be one of {sorted(valid_modes)}.")
        if self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive.")
        if self.observation_noise <= 0.0:
            raise ValueError("observation_noise must be positive.")
        if not 0.0 < self.redundancy_strength <= 1.0:
            raise ValueError("redundancy_strength must lie in (0, 1].")
        if self.mutual_information_scale <= 0.0:
            raise ValueError("mutual_information_scale must be positive.")
        if self.route_distance_weight < 0.0:
            raise ValueError("route_distance_weight must be non-negative.")
        if self.planning_horizon <= 0:
            raise ValueError("planning_horizon must be positive.")
        if not 0.0 < self.future_discount <= 1.0:
            raise ValueError("future_discount must lie in (0, 1].")
        if self.lookahead_strength < 0.0:
            raise ValueError("lookahead_strength must be non-negative.")
        if self.future_context_index is not None and self.future_context_index < 0:
            raise ValueError("future_context_index must be non-negative when provided.")
        if self.future_volatility_weight < 0.0:
            raise ValueError("future_volatility_weight must be non-negative.")
        if self.ppo_hidden_dim <= 0:
            raise ValueError("ppo_hidden_dim must be positive.")
        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive.")
        if self.ppo_learning_rate <= 0.0:
            raise ValueError("ppo_learning_rate must be positive.")
        if self.ppo_clip_epsilon <= 0.0:
            raise ValueError("ppo_clip_epsilon must be positive.")
        if self.ppo_entropy_weight < 0.0:
            raise ValueError("ppo_entropy_weight must be non-negative.")
        if self.ppo_value_weight < 0.0:
            raise ValueError("ppo_value_weight must be non-negative.")
        if self.ppo_warmstart_weight < 0.0:
            raise ValueError("ppo_warmstart_weight must be non-negative.")
        if self.ppo_policy_weight < 0.0:
            raise ValueError("ppo_policy_weight must be non-negative.")
        if self.ppo_temperature <= 0.0:
            raise ValueError("ppo_temperature must be positive.")
        if self.ppo_max_candidates <= 0:
            raise ValueError("ppo_max_candidates must be positive.")
        if self.ppo_rollout_episodes <= 0:
            raise ValueError("ppo_rollout_episodes must be positive.")
        if len(set(self.spatial_distance_dims)) != len(self.spatial_distance_dims):
            raise ValueError("spatial_distance_dims must not contain duplicates.")
        if any(dim < 0 for dim in self.spatial_distance_dims):
            raise ValueError("spatial_distance_dims must contain non-negative indices only.")


@dataclass
class LazyGreedyCache:
    """Cache for lazy-greedy upper bounds and redundancy state."""

    max_similarity: Tensor
    log_conditional_variance_factor: Tensor
    selected_mask: Tensor
    current_cost: float = 0.0


class _PolicyBackbone(nn.Module):
    """Small actor-critic network for PPO-style warm-start."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return actor logits and value estimates for features `[N, F]`."""
        hidden = self.backbone(x)
        return self.actor_head(hidden).squeeze(-1), self.value_head(hidden).squeeze(-1)


class LazyGreedyOptimizer:
    """Scalable active-sensing optimizer with myopic and rollout planners.

    The utility is an approximate Gaussian information surrogate, not an exact
    MI/EIG computation. The `lazy_greedy` planner is the original myopic
    Minoux-style optimizer. The `non_myopic_rollout` planner augments the same
    heap-based search with a discounted rollout surrogate over a short horizon.
    The `ppo_warmstart` planner adds an actor-critic prior trained against
    greedy surrogate rewards. The `ppo_online` planner further fine-tunes that
    prior on short environment rollouts using realized marginal-gain rewards
    while preserving the outer lazy-greedy heap for scalable selection.
    """

    def __init__(self, config: InformationPolicyConfig | None = None) -> None:
        self.config = config or InformationPolicyConfig()
        self.policy_network: _PolicyBackbone | None = None
        self.policy_optimizer: torch.optim.Optimizer | None = None
        self._policy_input_dim: int | None = None
        self._last_policy_stats: dict[str, float] = {}

    def _resolve_distance_dims(self, feature_dim: int) -> list[int]:
        """Resolve which feature dimensions define physical placement distance."""
        if self.config.spatial_distance_dims:
            dims = [dim for dim in self.config.spatial_distance_dims if dim < feature_dim]
            if not dims:
                raise ValueError(
                    "spatial_distance_dims does not overlap the provided candidate feature dimension."
                )
            return dims
        return list(range(min(2, feature_dim)))

    def _similarity_to_anchor(self, candidate_x: Tensor, anchor_x: Tensor) -> Tensor:
        """Compute an RBF-style similarity using only physical placement dimensions."""
        if candidate_x.ndim != 2:
            raise ValueError(
                f"candidate_x must have shape [N, D], received {tuple(candidate_x.shape)}."
            )
        if anchor_x.ndim != 1:
            raise ValueError(f"anchor_x must have shape [D], received {tuple(anchor_x.shape)}.")
        distance_dims = self._resolve_distance_dims(candidate_x.shape[-1])
        diff = candidate_x[:, distance_dims] - anchor_x[distance_dims].unsqueeze(0)
        squared_distance = diff.pow(2).sum(dim=-1)
        return torch.exp(-0.5 * squared_distance / (self.config.lengthscale ** 2))

    def _base_information_gain(
        self,
        effective_variance: Tensor,
        *,
        availability: Tensor | None = None,
    ) -> Tensor:
        """Return the one-step information-gain surrogate before cost handling."""
        effective_variance = torch.clamp(effective_variance, min=self.config.variance_floor)
        if self.config.utility_surrogate == "variance":
            gain = effective_variance
        else:
            signal_to_noise = effective_variance / (
                self.config.observation_noise + self.config.variance_floor
            )
            gain = self.config.mutual_information_scale * torch.log1p(signal_to_noise)
        if availability is not None:
            gain = gain * torch.clamp(availability, min=0.0, max=1.0)
        return gain

    def _contextual_volatility_scale(self, context_features: Tensor | None) -> Tensor:
        """Map context features to a bounded volatility multiplier."""
        if context_features is None:
            return torch.ones((), dtype=torch.float32)
        if context_features.ndim == 1:
            if self.config.future_context_index is not None:
                if self.config.future_context_index >= context_features.shape[0]:
                    return torch.ones((), device=context_features.device, dtype=context_features.dtype)
                raw = context_features[self.config.future_context_index]
            else:
                raw = context_features.norm(p=2)
            return 1.0 + self.config.future_volatility_weight * torch.log1p(torch.abs(raw))
        if context_features.ndim == 2:
            if context_features.shape[0] == 0:
                return torch.ones(0, device=context_features.device, dtype=context_features.dtype)
            if self.config.future_context_index is not None:
                if self.config.future_context_index >= context_features.shape[-1]:
                    return torch.ones(context_features.shape[0], device=context_features.device, dtype=context_features.dtype)
                raw = context_features[:, self.config.future_context_index]
            else:
                raw = context_features.norm(p=2, dim=-1)
            return 1.0 + self.config.future_volatility_weight * torch.log1p(torch.abs(raw))
        raise ValueError(
            f"context_features must have shape [C] or [N, C], received {tuple(context_features.shape)}."
        )

    def _rollout_gain(
        self,
        effective_variance: Tensor,
        *,
        availability: Tensor | None = None,
        novelty: Tensor | None = None,
        context_features: Tensor | None = None,
    ) -> Tensor:
        """Approximate a discounted non-myopic utility over a short horizon."""
        one_step_gain = self._base_information_gain(effective_variance, availability=availability)
        if novelty is None:
            novelty_factor = torch.ones_like(one_step_gain)
        elif torch.is_tensor(novelty):
            novelty_factor = torch.clamp(
                novelty.to(device=one_step_gain.device, dtype=one_step_gain.dtype),
                min=self.config.novelty_floor,
                max=1.0,
            )
        else:
            novelty_factor = torch.full_like(
                one_step_gain,
                float(max(min(novelty, 1.0), self.config.novelty_floor)),
            )
        if self.config.planning_strategy == "lazy_greedy" or self.config.planning_horizon == 1:
            return one_step_gain * novelty_factor

        volatility = self._contextual_volatility_scale(context_features).to(
            device=effective_variance.device,
            dtype=effective_variance.dtype,
        )
        propagated_variance = effective_variance
        propagated_novelty = novelty_factor
        total_gain = one_step_gain * propagated_novelty
        growth = torch.clamp(
            self.config.future_discount + self.config.future_volatility_weight * (volatility - 1.0),
            min=self.config.novelty_floor,
        )
        for step in range(1, self.config.planning_horizon):
            propagated_variance = torch.clamp(
                propagated_variance * growth,
                min=self.config.variance_floor,
            )
            propagated_novelty = torch.clamp(
                propagated_novelty * self.config.future_discount,
                min=self.config.novelty_floor,
                max=1.0,
            )
            future_gain = self._base_information_gain(propagated_variance, availability=availability)
            total_gain = total_gain + (
                self.config.lookahead_strength
                * (self.config.future_discount ** step)
                * volatility
                * future_gain
                * propagated_novelty
            )
        return total_gain

    def _policy_feature_stack(
        self,
        candidate_x: Tensor,
        candidate_variance: Tensor,
        candidate_cost: Tensor,
        *,
        availability: Tensor | None = None,
        candidate_context: Tensor | None = None,
        cache: LazyGreedyCache | None = None,
        active_budget: float | None = None,
    ) -> Tensor:
        """Build actor-critic features with shape `[N, F]`."""
        features = [
            candidate_x,
            torch.log(torch.clamp(candidate_variance.view(-1, 1), min=self.config.variance_floor)),
            candidate_cost.view(-1, 1),
        ]
        if availability is None:
            features.append(torch.ones(candidate_x.shape[0], 1, device=candidate_x.device, dtype=candidate_x.dtype))
        else:
            features.append(
                torch.clamp(availability.view(-1, 1), min=0.0, max=1.0).to(
                    device=candidate_x.device,
                    dtype=candidate_x.dtype,
                )
        )
        if candidate_context is not None:
            features.append(candidate_context.to(device=candidate_x.device, dtype=candidate_x.dtype))
        if cache is not None:
            features.extend(
                [
                    cache.selected_mask.view(-1, 1).to(device=candidate_x.device, dtype=candidate_x.dtype),
                    cache.max_similarity.view(-1, 1).to(device=candidate_x.device, dtype=candidate_x.dtype),
                    cache.log_conditional_variance_factor.view(-1, 1).to(
                        device=candidate_x.device,
                        dtype=candidate_x.dtype,
                    ),
                    self._route_distance_vector(candidate_x, cache).view(-1, 1).to(
                        device=candidate_x.device,
                        dtype=candidate_x.dtype,
                    ),
                ]
            )
            if active_budget is not None:
                remaining_budget = max(float(active_budget) - float(cache.current_cost), 0.0)
                normalized_remaining = remaining_budget / max(float(active_budget), self.config.ratio_epsilon)
                features.append(
                    torch.full(
                        (candidate_x.shape[0], 1),
                        normalized_remaining,
                        device=candidate_x.device,
                        dtype=candidate_x.dtype,
                    )
            )
        return torch.cat(features, dim=-1)

    def _route_distance_vector(self, candidate_x: Tensor, cache: LazyGreedyCache) -> Tensor:
        """Return nearest-route distances for every candidate with shape `[N]`."""
        if candidate_x.ndim != 2:
            raise ValueError(f"candidate_x must have shape [N, D], received {tuple(candidate_x.shape)}.")
        route_distance = torch.zeros(candidate_x.shape[0], device=candidate_x.device, dtype=candidate_x.dtype)
        if not torch.any(cache.selected_mask):
            return route_distance
        distance_dims = self._resolve_distance_dims(candidate_x.shape[-1])
        selected_points = candidate_x[cache.selected_mask][:, distance_dims]
        candidate_points = candidate_x[:, distance_dims]
        pairwise = torch.cdist(candidate_points, selected_points, p=2)
        route_distance = pairwise.amin(dim=-1)
        route_distance[cache.selected_mask] = 0.0
        return route_distance

    def _effective_operational_cost(
        self,
        cost: Tensor,
        *,
        route_distance: Tensor | None = None,
    ) -> Tensor:
        """Combine sensing and route costs into one effective operational cost."""
        effective_cost = cost
        if route_distance is not None:
            effective_cost = effective_cost + self.config.route_distance_weight * torch.clamp(
                route_distance,
                min=0.0,
            )
        return torch.clamp(effective_cost, min=self.config.ratio_epsilon)

    def _route_distance_for_candidate(
        self,
        index: int,
        candidate_x: Tensor | None,
        cache: LazyGreedyCache,
    ) -> Tensor | None:
        """Return the incremental route distance for one candidate."""
        if (
            candidate_x is None
            or self.config.route_distance_weight <= 0.0
            or not torch.any(cache.selected_mask)
        ):
            return None
        distance_dims = self._resolve_distance_dims(candidate_x.shape[-1])
        selected_points = candidate_x[cache.selected_mask][:, distance_dims]
        candidate_point = candidate_x[index, distance_dims].unsqueeze(0)
        return torch.cdist(candidate_point, selected_points, p=2).amin()

    def _ensure_policy_network(self, input_dim: int, *, device: torch.device, dtype: torch.dtype) -> None:
        """Create or reinitialize the PPO warm-start network for the current feature width."""
        if self.policy_network is not None and self._policy_input_dim == input_dim:
            self.policy_network = self.policy_network.to(device=device, dtype=dtype)
            return
        self.policy_network = _PolicyBackbone(input_dim=input_dim, hidden_dim=self.config.ppo_hidden_dim).to(
            device=device,
            dtype=dtype,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.config.ppo_learning_rate,
        )
        self._policy_input_dim = input_dim
        self._last_policy_stats = {}

    def _resolve_policy_training_subset(
        self,
        teacher_gain: Tensor,
        *,
        max_selections: int | None,
    ) -> Tensor:
        """Choose a bounded subset of candidates for PPO warm-start updates."""
        num_candidates = teacher_gain.shape[0]
        limit = min(num_candidates, self.config.ppo_max_candidates)
        if limit >= num_candidates:
            return torch.arange(num_candidates, device=teacher_gain.device)

        top_count = max(1, min(limit // 2, num_candidates))
        top_index = torch.topk(teacher_gain, k=top_count).indices
        remaining_mask = torch.ones(num_candidates, device=teacher_gain.device, dtype=torch.bool)
        remaining_mask[top_index] = False
        remaining_index = remaining_mask.nonzero(as_tuple=False).squeeze(-1)
        random_count = max(0, limit - top_count)
        if random_count == 0 or remaining_index.numel() == 0:
            return top_index
        random_perm = torch.randperm(remaining_index.numel(), device=teacher_gain.device)[:random_count]
        sampled = remaining_index[random_perm]
        combined = torch.cat([top_index, sampled], dim=0)
        if max_selections is not None and max_selections > 0 and combined.numel() < max_selections:
            extra_top = torch.topk(teacher_gain, k=min(max_selections, num_candidates)).indices
            combined = torch.unique(torch.cat([combined, extra_top], dim=0))
        return combined[:limit]

    def _initialize_cache(self, candidate_x: Tensor) -> LazyGreedyCache:
        """Create a fresh lazy-greedy cache for one candidate pool."""
        return LazyGreedyCache(
            max_similarity=torch.zeros(candidate_x.shape[0], device=candidate_x.device, dtype=candidate_x.dtype),
            log_conditional_variance_factor=torch.zeros(
                candidate_x.shape[0],
                device=candidate_x.device,
                dtype=candidate_x.dtype,
            ),
            selected_mask=torch.zeros(candidate_x.shape[0], device=candidate_x.device, dtype=torch.bool),
            current_cost=0.0,
        )

    def _invalid_action_mask(
        self,
        cache: LazyGreedyCache,
        candidate_cost: Tensor,
        *,
        active_budget: float | None,
    ) -> Tensor:
        """Return a mask for candidates that cannot be selected in the current state."""
        invalid = cache.selected_mask.clone()
        if active_budget is not None:
            invalid = invalid | ((cache.current_cost + candidate_cost) > active_budget)
        return invalid

    def _state_value(self, candidate_values: Tensor, invalid_mask: Tensor) -> Tensor:
        """Aggregate candidate-wise value estimates into one state-value scalar."""
        valid_values = candidate_values[~invalid_mask]
        if valid_values.numel() == 0:
            return candidate_values.new_tensor(0.0)
        return valid_values.mean()

    def _policy_action_distribution(
        self,
        logits: Tensor,
        invalid_mask: Tensor,
    ) -> torch.distributions.Categorical:
        """Return a categorical policy over currently valid actions."""
        masked_logits = logits.masked_fill(invalid_mask, -1e9)
        probabilities = torch.softmax(masked_logits / self.config.ppo_temperature, dim=-1)
        return torch.distributions.Categorical(probs=probabilities)

    def warm_start_policy(
        self,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        candidate_variance: Tensor,
        *,
        availability: Tensor | None = None,
        candidate_context: Tensor | None = None,
        cache: LazyGreedyCache | None = None,
        active_budget: float | None = None,
        max_selections: int | None = None,
    ) -> dict[str, float]:
        """Train a PPO-style actor-critic prior from greedy surrogate rewards.

        This is a bounded, candidate-wise warm-start step rather than a full
        environment-trained PPO loop. It is designed to remain scalable on large
        candidate pools by training on a capped subset while still exposing a
        DRL-flavored policy ablation in the unified pipeline.
        """
        features = self._policy_feature_stack(
            candidate_x,
            candidate_variance,
            candidate_cost,
            availability=availability,
            candidate_context=candidate_context,
            cache=cache,
            active_budget=active_budget,
        )
        self._ensure_policy_network(features.shape[-1], device=features.device, dtype=features.dtype)
        if self.policy_network is None or self.policy_optimizer is None:
            raise RuntimeError("Failed to initialize PPO warm-start policy network.")

        with torch.no_grad():
            teacher_gain = self.score_candidate(
                candidate_variance,
                candidate_cost,
                availability=availability,
                conditional_variance=candidate_variance,
                context_features=candidate_context,
            )
            train_index = self._resolve_policy_training_subset(
                teacher_gain,
                max_selections=max_selections,
            )
            train_features = features[train_index]
            train_gain = teacher_gain[train_index]
            selection_limit = max_selections if max_selections is not None else self.config.max_selections
            if selection_limit is None:
                selection_limit = max(1, min(5, train_index.numel() // 10 or 1))
            selection_limit = max(1, min(int(selection_limit), train_index.numel()))
            teacher_actions = torch.zeros(train_index.numel(), device=features.device, dtype=features.dtype)
            topk = torch.topk(train_gain, k=selection_limit).indices
            teacher_actions[topk] = 1.0
            advantage = train_gain - train_gain.mean()
            advantage = advantage / torch.clamp(
                train_gain.std(unbiased=False),
                min=self.config.variance_floor,
            )
            reward_target = train_gain / torch.clamp(
                train_gain.abs().mean(),
                min=self.config.variance_floor,
            )
            old_logits, _ = self.policy_network(train_features)
            old_probs = torch.sigmoid(old_logits).detach()

        self.policy_network.train()
        final_loss = 0.0
        final_policy_loss = 0.0
        final_value_loss = 0.0
        final_entropy = 0.0
        for _ in range(self.config.ppo_epochs):
            logits, values = self.policy_network(train_features)
            probs = torch.sigmoid(logits)
            action_prob = torch.where(teacher_actions > 0.5, probs, 1.0 - probs)
            old_action_prob = torch.where(teacher_actions > 0.5, old_probs, 1.0 - old_probs)
            old_action_prob = torch.clamp(old_action_prob, min=self.config.variance_floor)
            ratio = action_prob / old_action_prob
            clipped_ratio = torch.clamp(
                ratio,
                min=1.0 - self.config.ppo_clip_epsilon,
                max=1.0 + self.config.ppo_clip_epsilon,
            )
            surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
            policy_loss = -surrogate.mean()
            warmstart_loss = F.binary_cross_entropy_with_logits(logits, teacher_actions)
            value_loss = F.mse_loss(values, reward_target, reduction="mean")
            entropy = -(
                probs * torch.log(torch.clamp(probs, min=self.config.variance_floor))
                + (1.0 - probs) * torch.log(torch.clamp(1.0 - probs, min=self.config.variance_floor))
            ).mean()
            loss = (
                policy_loss
                + self.config.ppo_warmstart_weight * warmstart_loss
                + self.config.ppo_value_weight * value_loss
                - self.config.ppo_entropy_weight * entropy
            )
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
            self.policy_optimizer.step()
            final_loss = float(loss.item())
            final_policy_loss = float(policy_loss.item())
            final_value_loss = float(value_loss.item())
            final_entropy = float(entropy.item())

        self.policy_network.eval()
        self._last_policy_stats = {
            "policy_training_loss": final_loss,
            "policy_surrogate_loss": final_policy_loss,
            "policy_value_loss": final_value_loss,
            "policy_entropy": final_entropy,
            "policy_training_steps": float(self.config.ppo_epochs),
            "policy_training_candidates": float(train_index.numel()),
        }
        return dict(self._last_policy_stats)

    def online_train_policy(
        self,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        candidate_variance: Tensor,
        *,
        availability: Tensor | None = None,
        candidate_context: Tensor | None = None,
        budget: float | None = None,
        max_selections: int | None = None,
    ) -> dict[str, float]:
        """Fine-tune the actor-critic on short simulated selection rollouts."""
        active_budget = self.config.budget if budget is None else budget
        selection_limit = self.config.max_selections if max_selections is None else max_selections
        if selection_limit is None:
            selection_limit = max(1, min(5, candidate_x.shape[0]))
        selection_limit = max(1, min(int(selection_limit), candidate_x.shape[0]))

        with torch.no_grad():
            teacher_gain = self.score_candidate(
                candidate_variance,
                candidate_cost,
                availability=availability,
                conditional_variance=candidate_variance,
                context_features=candidate_context,
            )
            train_index = self._resolve_policy_training_subset(
                teacher_gain,
                max_selections=selection_limit,
            )
        subset_x = candidate_x[train_index]
        subset_cost = candidate_cost[train_index]
        subset_variance = candidate_variance[train_index]
        subset_availability = None if availability is None else availability[train_index]
        subset_context = None if candidate_context is None else candidate_context[train_index]
        initial_cache = self._initialize_cache(subset_x)

        warm_stats = self.warm_start_policy(
            subset_x,
            subset_cost,
            subset_variance,
            availability=subset_availability,
            candidate_context=subset_context,
            cache=initial_cache,
            active_budget=active_budget,
            max_selections=selection_limit,
        )
        base_features = self._policy_feature_stack(
            subset_x,
            subset_variance,
            subset_cost,
            availability=subset_availability,
            candidate_context=subset_context,
            cache=initial_cache,
            active_budget=active_budget,
        )
        self._ensure_policy_network(base_features.shape[-1], device=base_features.device, dtype=base_features.dtype)
        if self.policy_network is None or self.policy_optimizer is None:
            raise RuntimeError("Failed to initialize PPO online policy network.")

        self.policy_network.train()
        final_loss = 0.0
        final_policy_loss = 0.0
        final_value_loss = 0.0
        final_entropy = 0.0
        mean_reward = 0.0
        total_steps = 0
        for _ in range(self.config.ppo_epochs):
            episode_rewards: list[float] = []
            batch_state_features: list[Tensor] = []
            batch_invalid_masks: list[Tensor] = []
            batch_actions: list[Tensor] = []
            batch_old_log_probs: list[Tensor] = []
            batch_returns: list[Tensor] = []
            batch_advantages: list[Tensor] = []
            for _episode in range(self.config.ppo_rollout_episodes):
                cache = self._initialize_cache(subset_x)
                rewards: list[Tensor] = []
                state_values: list[Tensor] = []
                episode_features: list[Tensor] = []
                episode_invalid_masks: list[Tensor] = []
                episode_actions: list[Tensor] = []
                episode_old_log_probs: list[Tensor] = []
                for _step in range(selection_limit):
                    state_features = self._policy_feature_stack(
                        subset_x,
                        subset_variance,
                        subset_cost,
                        availability=subset_availability,
                        candidate_context=subset_context,
                        cache=cache,
                        active_budget=active_budget,
                    )
                    logits, candidate_values = self.policy_network(state_features)
                    invalid_mask = self._invalid_action_mask(cache, subset_cost, active_budget=active_budget)
                    if bool(invalid_mask.all().item()):
                        break
                    distribution = self._policy_action_distribution(logits, invalid_mask)
                    action = distribution.sample()
                    old_log_prob = distribution.log_prob(action).detach()
                    reward = self.marginal_gain(
                        int(action.item()),
                        subset_variance,
                        subset_cost,
                        cache,
                        availability=subset_availability,
                        context_features=subset_context,
                    )
                    episode_features.append(state_features.detach())
                    episode_invalid_masks.append(invalid_mask.detach())
                    episode_actions.append(action.detach())
                    episode_old_log_probs.append(old_log_prob)
                    state_values.append(self._state_value(candidate_values, invalid_mask))
                    rewards.append(reward.detach())
                    cache.current_cost += float(subset_cost[action].item())
                    self.update_cache(
                        cache=cache,
                        candidate_x=subset_x,
                        candidate_variance=subset_variance,
                        selected_index=int(action.item()),
                    )
                if not rewards:
                    continue
                returns: list[Tensor] = []
                running = rewards[0].new_tensor(0.0)
                for reward in reversed(rewards):
                    running = reward + self.config.future_discount * running
                    returns.append(running)
                returns.reverse()
                returns_tensor = torch.stack(returns)
                values_tensor = torch.stack(state_values)
                advantages = returns_tensor - values_tensor.detach()
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / torch.clamp(
                        advantages.std(unbiased=False),
                        min=self.config.variance_floor,
                    )
                batch_state_features.extend(episode_features)
                batch_invalid_masks.extend(episode_invalid_masks)
                batch_actions.extend(episode_actions)
                batch_old_log_probs.extend(episode_old_log_probs)
                batch_returns.extend([value.detach() for value in returns_tensor])
                batch_advantages.extend([value.detach() for value in advantages])
                episode_rewards.append(float(torch.stack(rewards).sum().item()))
                total_steps += len(rewards)
            if not batch_state_features:
                continue
            policy_losses: list[Tensor] = []
            value_losses: list[Tensor] = []
            entropies: list[Tensor] = []
            for state_features, invalid_mask, action, old_log_prob, target_return, advantage in zip(
                batch_state_features,
                batch_invalid_masks,
                batch_actions,
                batch_old_log_probs,
                batch_returns,
                batch_advantages,
            ):
                logits, candidate_values = self.policy_network(state_features)
                distribution = self._policy_action_distribution(logits, invalid_mask)
                new_log_prob = distribution.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped_ratio = torch.clamp(
                    ratio,
                    min=1.0 - self.config.ppo_clip_epsilon,
                    max=1.0 + self.config.ppo_clip_epsilon,
                )
                surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
                policy_losses.append(-surrogate)
                value_estimate = self._state_value(candidate_values, invalid_mask)
                value_losses.append(F.mse_loss(value_estimate, target_return, reduction="mean"))
                entropies.append(distribution.entropy())
            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy = torch.stack(entropies).mean()
            loss = (
                policy_loss
                + self.config.ppo_value_weight * value_loss
                - self.config.ppo_entropy_weight * entropy
            )
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=10.0)
            self.policy_optimizer.step()
            final_loss = float(loss.item())
            final_policy_loss = float(policy_loss.item())
            final_value_loss = float(value_loss.item())
            final_entropy = float(entropy.item())
            mean_reward = float(sum(episode_rewards) / max(len(episode_rewards), 1))

        self.policy_network.eval()
        self._last_policy_stats = {
            **warm_stats,
            "policy_training_loss": final_loss if total_steps > 0 else warm_stats.get("policy_training_loss", 0.0),
            "policy_surrogate_loss": final_policy_loss if total_steps > 0 else warm_stats.get("policy_surrogate_loss", 0.0),
            "policy_value_loss": final_value_loss if total_steps > 0 else warm_stats.get("policy_value_loss", 0.0),
            "policy_entropy": final_entropy if total_steps > 0 else warm_stats.get("policy_entropy", 0.0),
            "policy_rollout_episodes": float(self.config.ppo_rollout_episodes),
            "policy_rollout_mean_reward": mean_reward,
            "policy_rollout_steps": float(total_steps),
            "policy_training_candidates": float(train_index.numel()),
        }
        return dict(self._last_policy_stats)

    def _policy_prior(
        self,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        candidate_variance: Tensor,
        *,
        availability: Tensor | None = None,
        candidate_context: Tensor | None = None,
        cache: LazyGreedyCache | None = None,
        active_budget: float | None = None,
    ) -> Tensor:
        """Return PPO policy prior probabilities with shape `[N]`."""
        features = self._policy_feature_stack(
            candidate_x,
            candidate_variance,
            candidate_cost,
            availability=availability,
            candidate_context=candidate_context,
            cache=cache,
            active_budget=active_budget,
        )
        self._ensure_policy_network(features.shape[-1], device=features.device, dtype=features.dtype)
        if self.policy_network is None:
            raise RuntimeError("PPO warm-start policy network is unavailable.")
        self.policy_network.eval()
        with torch.no_grad():
            logits, _ = self.policy_network(features)
        return torch.sigmoid(logits / self.config.ppo_temperature)

    def _policy_prior_bonus(
        self,
        policy_prior: Tensor,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        *,
        cache: LazyGreedyCache | None = None,
    ) -> Tensor:
        """Convert policy probabilities into route-aware upper-bound bonuses."""
        if cache is None:
            route_distance = torch.zeros_like(candidate_cost)
        else:
            route_distance = self._route_distance_vector(candidate_x, cache)
        effective_cost = self._effective_operational_cost(candidate_cost, route_distance=route_distance)
        return policy_prior / effective_cost

    def _screen_with_policy_prior(
        self,
        upper_bounds: Tensor,
        policy_prior: Tensor | None,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        *,
        cache: LazyGreedyCache,
        active_limit: int | None = None,
    ) -> Tensor:
        """Use PPO priors only to shortlist very large candidate pools."""
        if policy_prior is None or upper_bounds.numel() <= self.config.ppo_max_candidates:
            return upper_bounds

        valid_mask = ~cache.selected_mask
        valid_count = int(valid_mask.sum().item())
        if valid_count <= self.config.ppo_max_candidates:
            return upper_bounds

        shortlist_size = min(self.config.ppo_max_candidates, valid_count)
        reserve = max(1, int(active_limit) if active_limit is not None else self.config.planning_horizon)
        probe_size = min(valid_count, max(shortlist_size // 2, reserve * 8))
        neg_inf = upper_bounds.new_tensor(float("-inf"))
        valid_upper = upper_bounds.masked_fill(~valid_mask, neg_inf)
        valid_bonus = self._policy_prior_bonus(
            policy_prior,
            candidate_x,
            candidate_cost,
            cache=cache,
        ).masked_fill(~valid_mask, neg_inf)

        upper_index = torch.topk(valid_upper, k=probe_size).indices
        bonus_index = torch.topk(valid_bonus, k=probe_size).indices
        shortlisted = torch.unique(torch.cat([upper_index, bonus_index], dim=0))
        if shortlisted.numel() > shortlist_size:
            combined = valid_upper[shortlisted]
            finite_bonus = torch.clamp(valid_bonus[shortlisted], min=0.0)
            if torch.any(finite_bonus > 0.0):
                finite_bonus = finite_bonus / torch.clamp(
                    finite_bonus.max(),
                    min=self.config.ratio_epsilon,
                )
                combined = combined + self.config.ppo_policy_weight * finite_bonus
            keep = torch.topk(combined, k=shortlist_size).indices
            shortlisted = shortlisted[keep]

        screened = upper_bounds.new_full(upper_bounds.shape, float("-inf"))
        screened[shortlisted] = upper_bounds[shortlisted]
        screened[cache.selected_mask] = float("-inf")
        self._last_policy_stats["policy_screened_candidates"] = float(shortlisted.numel())
        return screened

    def _current_upper_bounds(
        self,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        candidate_variance: Tensor,
        *,
        cache: LazyGreedyCache,
        availability: Tensor | None = None,
        candidate_context: Tensor | None = None,
    ) -> Tensor:
        """Recompute cached upper bounds under the current selection state."""
        conditional_variance = torch.clamp(
            candidate_variance * torch.exp(cache.log_conditional_variance_factor),
            min=self.config.variance_floor,
        )
        novelty = 1.0 - torch.clamp(cache.max_similarity, min=0.0, max=1.0)
        route_distance = (
            self._route_distance_vector(candidate_x, cache)
            if self.config.route_distance_weight > 0.0
            else None
        )
        upper_bounds = self.score_candidate(
            candidate_variance,
            candidate_cost,
            availability=availability,
            conditional_variance=conditional_variance,
            novelty=novelty,
            context_features=candidate_context,
            route_distance=route_distance,
        )
        upper_bounds = upper_bounds.clone()
        upper_bounds[cache.selected_mask] = float("-inf")
        return upper_bounds

    def score_candidate(
        self,
        variance: Tensor,
        cost: Tensor,
        *,
        availability: Tensor | None = None,
        conditional_variance: Tensor | None = None,
        novelty: Tensor | None = None,
        context_features: Tensor | None = None,
        route_distance: Tensor | None = None,
    ) -> Tensor:
        """Score a candidate using a scalable approximate information surrogate."""
        variance = torch.clamp(variance, min=self.config.variance_floor)
        effective_variance = variance if conditional_variance is None else torch.clamp(
            conditional_variance,
            min=self.config.variance_floor,
        )
        base_gain = self._rollout_gain(
            effective_variance,
            availability=availability,
            novelty=novelty,
            context_features=context_features,
        )
        effective_cost = self._effective_operational_cost(cost, route_distance=route_distance)

        if self.config.selection_mode == "penalized":
            return base_gain - self.config.cost_weight * effective_cost
        if self.config.selection_mode == "ratio":
            return base_gain / effective_cost
        if route_distance is not None and self.config.route_distance_weight > 0.0:
            return base_gain - self.config.route_distance_weight * route_distance
        return base_gain

    def marginal_gain(
        self,
        index: int,
        candidate_variance: Tensor,
        candidate_cost: Tensor,
        cache: LazyGreedyCache,
        *,
        candidate_x: Tensor | None = None,
        availability: Tensor | None = None,
        context_features: Tensor | None = None,
    ) -> Tensor:
        """Compute the lazy-greedy marginal gain for one candidate."""
        conditional_variance = torch.clamp(
            candidate_variance[index] * torch.exp(cache.log_conditional_variance_factor[index]),
            min=self.config.variance_floor,
        )
        novelty = 1.0 - torch.clamp(cache.max_similarity[index], min=0.0, max=1.0)
        route_distance = self._route_distance_for_candidate(index, candidate_x, cache)
        gain = self.score_candidate(
            candidate_variance[index],
            candidate_cost[index],
            availability=None if availability is None else availability[index],
            conditional_variance=conditional_variance,
            novelty=novelty,
            context_features=None if context_features is None else context_features[index],
            route_distance=route_distance,
        )
        return gain

    def _routing_cost(self, selected_x: Tensor) -> float:
        """Compute ordered routing cost over selected physical coordinates."""
        if selected_x.shape[0] <= 1:
            return 0.0
        distance_dims = self._resolve_distance_dims(selected_x.shape[-1])
        diffs = selected_x[1:, distance_dims] - selected_x[:-1, distance_dims]
        return float(torch.sqrt(torch.clamp(diffs.pow(2).sum(dim=-1), min=1e-6)).sum().item())

    def _optimize_route_order(self, selected_x: Tensor) -> Tensor:
        """Reorder selected points using a nearest-neighbor route heuristic."""
        if selected_x.shape[0] <= 2:
            return torch.arange(selected_x.shape[0], device=selected_x.device, dtype=torch.long)
        distance_dims = self._resolve_distance_dims(selected_x.shape[-1])
        points = selected_x[:, distance_dims]
        pairwise = torch.cdist(points, points, p=2)
        best_order = torch.arange(selected_x.shape[0], device=selected_x.device, dtype=torch.long)
        best_cost = float("inf")
        for start in range(selected_x.shape[0]):
            remaining = set(range(selected_x.shape[0]))
            remaining.remove(start)
            order = [start]
            current = start
            while remaining:
                next_index = min(remaining, key=lambda idx: float(pairwise[current, idx].item()))
                order.append(next_index)
                remaining.remove(next_index)
                current = next_index
            order_tensor = torch.tensor(order, device=selected_x.device, dtype=torch.long)
            route_cost = self._routing_cost(selected_x[order_tensor])
            if route_cost < best_cost:
                best_cost = route_cost
                best_order = order_tensor
        return best_order

    def update_cache(
        self,
        cache: LazyGreedyCache,
        candidate_x: Tensor,
        candidate_variance: Tensor,
        selected_index: int,
    ) -> LazyGreedyCache:
        """Update cached conditional-variance factors after selecting one candidate."""
        anchor = candidate_x[selected_index]
        similarity = self._similarity_to_anchor(candidate_x, anchor)
        selected_variance = torch.clamp(candidate_variance[selected_index], min=self.config.variance_floor)
        reduction = self.config.redundancy_strength * similarity.pow(2) * (
            selected_variance / (selected_variance + self.config.observation_noise)
        )
        residual_factor = torch.clamp(
            1.0 - reduction,
            min=self.config.novelty_floor,
            max=1.0,
        )
        cache.log_conditional_variance_factor = (
            cache.log_conditional_variance_factor + torch.log(residual_factor)
        )
        cache.max_similarity = torch.maximum(cache.max_similarity, similarity)
        cache.max_similarity[selected_index] = torch.tensor(
            1.0,
            device=cache.max_similarity.device,
            dtype=cache.max_similarity.dtype,
        )
        cache.log_conditional_variance_factor[selected_index] = torch.log(
            candidate_variance.new_tensor(self.config.novelty_floor)
        )
        cache.selected_mask[selected_index] = True
        return cache

    def select(
        self,
        candidate_x: Tensor,
        candidate_cost: Tensor,
        candidate_variance: Tensor,
        *,
        availability: Tensor | None = None,
        candidate_context: Tensor | None = None,
        already_selected: list[int] | None = None,
        budget: float | None = None,
        max_selections: int | None = None,
    ) -> dict[str, Tensor | list[int] | float | dict[str, float]]:
        """Select candidates using lazy greedy heap-based reevaluation.

        Args:
            candidate_x: Candidate features with shape `[N, D]`.
            candidate_cost: Candidate costs with shape `[N]`.
            candidate_variance: Posterior marginal variances with shape `[N]`.
            availability: Optional availability weights with shape `[N]`.
            already_selected: Optional pre-selected candidate indices.
            budget: Optional cost budget overriding the config.
            max_selections: Optional cardinality limit overriding the config.

        Returns:
            Dictionary containing selected indices, selected coordinates, utility
            trace, and total cost.
        """
        if candidate_x.ndim != 2:
            raise ValueError(f"candidate_x must have shape [N, D], received {tuple(candidate_x.shape)}.")
        candidate_cost = candidate_cost.view(-1)
        candidate_variance = candidate_variance.view(-1)
        if candidate_x.shape[0] != candidate_cost.shape[0] or candidate_x.shape[0] != candidate_variance.shape[0]:
            raise ValueError("candidate_x, candidate_cost, and candidate_variance must share the same length.")

        if availability is not None:
            availability = availability.view(-1)
            if availability.shape[0] != candidate_x.shape[0]:
                raise ValueError("availability must have the same length as candidate_x.")
        if candidate_context is not None:
            if candidate_context.ndim != 2:
                raise ValueError(
                    f"candidate_context must have shape [N, C], received {tuple(candidate_context.shape)}."
                )
            if candidate_context.shape[0] != candidate_x.shape[0]:
                raise ValueError("candidate_context must have the same length as candidate_x.")

        active_budget = self.config.budget if budget is None else budget
        active_limit = self.config.max_selections if max_selections is None else max_selections
        selected: list[int] = []
        utility_trace: list[float] = []

        cache = self._initialize_cache(candidate_x)

        seeded = already_selected or []
        for index in seeded:
            selected.append(index)
            cache.current_cost += float(candidate_cost[index].item())
            self.update_cache(
                cache=cache,
                candidate_x=candidate_x,
                candidate_variance=candidate_variance,
                selected_index=index,
            )

        upper_bounds = self._current_upper_bounds(
            candidate_x,
            candidate_cost,
            candidate_variance,
            cache=cache,
            availability=availability,
            candidate_context=candidate_context,
        )
        policy_prior: Tensor | None = None
        if self.config.planning_strategy in {"ppo_warmstart", "ppo_online"}:
            if self.config.planning_strategy == "ppo_online":
                self.online_train_policy(
                    candidate_x,
                    candidate_cost,
                    candidate_variance,
                    availability=availability,
                    candidate_context=candidate_context,
                    budget=active_budget,
                    max_selections=active_limit,
                )
            else:
                self.warm_start_policy(
                    candidate_x,
                    candidate_cost,
                    candidate_variance,
                    availability=availability,
                    candidate_context=candidate_context,
                    max_selections=active_limit,
                )
            policy_prior = self._policy_prior(
                candidate_x,
                candidate_cost,
                candidate_variance,
                availability=availability,
                candidate_context=candidate_context,
                cache=cache if self.config.planning_strategy == "ppo_online" else None,
                active_budget=active_budget if self.config.planning_strategy == "ppo_online" else None,
            )
            upper_bounds = self._current_upper_bounds(
                candidate_x,
                candidate_cost,
                candidate_variance,
                cache=cache,
                availability=availability,
                candidate_context=candidate_context,
            )
            upper_bounds = self._screen_with_policy_prior(
                upper_bounds,
                policy_prior,
                candidate_x,
                candidate_cost,
                cache=cache,
                active_limit=active_limit,
            )
        else:
            self._last_policy_stats = {}
        heap: list[tuple[float, int, int]] = []
        current_round = len(selected)
        for index, bound in enumerate(upper_bounds.tolist()):
            if cache.selected_mask[index]:
                continue
            heapq.heappush(heap, (-bound, index, -1))

        while heap:
            if active_limit is not None and len(selected) >= active_limit:
                break

            _, index, evaluated_round = heapq.heappop(heap)
            if cache.selected_mask[index]:
                continue

            incremental_cost = float(candidate_cost[index].item())
            if active_budget is not None and cache.current_cost + incremental_cost > active_budget:
                continue

            exact_gain = float(
                self.marginal_gain(
                    index=index,
                    candidate_variance=candidate_variance,
                    candidate_cost=candidate_cost,
                    availability=availability,
                    candidate_x=candidate_x,
                    context_features=candidate_context,
                    cache=cache,
                ).item()
            )
            next_upper = -heap[0][0] if heap else float("-inf")

            if exact_gain <= 0.0 and (evaluated_round == current_round or exact_gain >= next_upper):
                break

            if evaluated_round == current_round or exact_gain >= next_upper:
                selected.append(index)
                utility_trace.append(exact_gain)
                cache.current_cost += incremental_cost
                self.update_cache(
                    cache=cache,
                    candidate_x=candidate_x,
                    candidate_variance=candidate_variance,
                    selected_index=index,
                )
                if policy_prior is not None and self.config.planning_strategy == "ppo_online":
                    policy_prior = self._policy_prior(
                        candidate_x,
                        candidate_cost,
                        candidate_variance,
                        availability=availability,
                        candidate_context=candidate_context,
                        cache=cache,
                        active_budget=active_budget,
                    )
                upper_bounds = self._current_upper_bounds(
                    candidate_x,
                    candidate_cost,
                    candidate_variance,
                    cache=cache,
                    availability=availability,
                    candidate_context=candidate_context,
                )
                upper_bounds = self._screen_with_policy_prior(
                    upper_bounds,
                    policy_prior,
                    candidate_x,
                    candidate_cost,
                    cache=cache,
                    active_limit=active_limit,
                )
                heap = []
                for next_index, bound in enumerate(upper_bounds.tolist()):
                    if cache.selected_mask[next_index]:
                        continue
                    heapq.heappush(heap, (-bound, next_index, current_round))
                current_round += 1
                continue

            heapq.heappush(heap, (-exact_gain, index, current_round))

        selected_tensor = torch.tensor(selected, device=candidate_x.device, dtype=torch.long)
        selected_x = candidate_x[selected_tensor] if selected else candidate_x.new_zeros((0, candidate_x.shape[-1]))
        if selected:
            route_order = self._optimize_route_order(selected_x)
            selected_tensor = selected_tensor[route_order]
            selected_x = selected_x[route_order]
            selected = selected_tensor.tolist()
        utility_tensor = torch.tensor(utility_trace, device=candidate_x.device, dtype=candidate_x.dtype)
        routing_cost = self._routing_cost(selected_x)
        return {
            "selected_indices": selected,
            "selected_x": selected_x,
            "utility_trace": utility_tensor,
            "total_cost": cache.current_cost,
            "routing_cost": routing_cost,
            "total_operational_cost": cache.current_cost + routing_cost,
            "planning_strategy": self.config.planning_strategy,
            "policy_stats": dict(self._last_policy_stats),
        }
