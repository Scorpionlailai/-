from pathlib import Path
p=Path('src/malr/models/e23/model.py'); s=p.read_text()
s=s.replace('from .iterative import IterationDiagnostics, IterativeTVHController','from .iterative import IterationDiagnostics, IterativeTVHController\nfrom .historical_candidates import HistoricalCandidateProvider\nfrom .stacked import StackLayerDiagnostics, StackedTVHController')
s=s.replace('    iterations: tuple[IterationDiagnostics, ...] | None = None','    iterations: tuple[IterationDiagnostics, ...] | None = None\n    stack_layers: tuple[StackLayerDiagnostics, ...] | None = None')
s=s.replace('        recovery_rounds: int = 1,','        recovery_rounds: int = 1,\n        recovery_layers: int = 1, expert_groups: int = 1, activation_checkpointing: bool = False,')
s=s.replace('("parallel_v1", "iterative_tv_h_v1")','("parallel_v1", "iterative_tv_h_v1", "stacked_parallel_tv_h_v1")',1)
s=s.replace('        if contextual and (','        stacked = recovery_schedule == "stacked_parallel_tv_h_v1"\n        if type(recovery_layers) is not int or type(expert_groups) is not int or min(recovery_layers, expert_groups) < 1 or type(activation_checkpointing) is not bool:\n            raise ValueError("invalid stacked recovery geometry")\n        if not stacked and (recovery_layers != 1 or expert_groups != 1 or activation_checkpointing):\n            raise ValueError("stack parameters require stacked recovery")\n        if (contextual or stacked) and (',1)
s=s.replace('        self.recovery_rounds = recovery_rounds','        self.recovery_rounds = recovery_rounds\n        self.recovery_layers = recovery_layers\n        self.expert_groups = expert_groups\n        self.activation_checkpointing = activation_checkpointing',1)
a=s.index('        self.historical = HistoricalExpert('); b=s.index('        self.history_adapter =',a)
old=s[a:b]
new='''        self.stacked = None
        if stacked:
            self.historical = HistoricalCandidateProvider(patch_length,d_model,retrieval_dim,variable_count,
                topk=historical_topk,similarity_threshold=historical_threshold,
                retrieval_strategy=historical_retrieval)
            self.stacked = StackedTVHController(variable_count,patch_length,d_model,layers=recovery_layers,
                groups=expert_groups,activation_checkpointing=activation_checkpointing,
                variable_topk=variable_topk,variable_offsets=variable_offsets,
                temporal_local_radius=temporal_local_radius,temporal_boundaries=temporal_boundaries)
        else:
'''+''.join('    '+line+'\n' for line in old.splitlines())
s=s[:a]+new+s[b:]
s=s.replace('        disabled_experts: frozenset[str] = frozenset(),','        disabled_experts: frozenset[str] = frozenset(),\n        disabled_groups: frozenset[tuple[int,int]] = frozenset(),',1)
s=s.replace('        if self.iterative is not None:\n            patch_values','        if self.iterative is not None or self.stacked is not None:\n            patch_values',1)
s=s.replace('        iterations = None\n        if self.iterative is not None:', '''        iterations = None; stack_layers = None
        if self.stacked is not None:
            result = self.stacked(inputs,historical=self.historical,epoch=epoch,explore_variable=explore_variable,
                specialist_exposure=specialist_exposure,specialist_seed=specialist_seed,
                disabled_experts=disabled_experts,disabled_groups=disabled_groups)
            routed,evidence = result.router,dict(result.evidence)
            selected,stack_layers = result.specialist_selected,result.layers
        elif self.iterative is not None:''',1)
s=s.replace('return E23RepairOutput(explicit, routed, evidence, base, selected, iterations)','return E23RepairOutput(explicit, routed, evidence, base, selected, iterations, stack_layers)')
s=s.replace('        self.forecast_input = forecast_input','        if repair.stacked is not None and (forecast_input != "explicit_v1" or predictor.variable_mixer is not None):\n            raise ValueError("stacked recovery requires the original explicit predictor")\n        self.forecast_input = forecast_input',1)
p.write_text(s)
p=Path('integrations/e23/config.py'); s=p.read_text()
s=s.replace('    recovery_rounds: int = 1','    recovery_rounds: int = 1\n    recovery_layers: int = 1\n    expert_groups: int = 1\n    activation_checkpointing: bool = False',1)
s=s.replace('_TRAINING = frozenset({','_MODEL_STACKED_WINDOW = _MODEL_ITERATIVE_WINDOW | {"recovery_layers", "expert_groups", "activation_checkpointing"}\n_TRAINING = frozenset({',1)
s=s.replace('        _MODEL_ITERATIVE_WINDOW,','        _MODEL_ITERATIVE_WINDOW, _MODEL_STACKED_WINDOW,',1)
s=s.replace('        recovery_rounds=_integer(model.get("recovery_rounds", 1), "recovery_rounds"),','        recovery_rounds=_integer(model.get("recovery_rounds", 1), "recovery_rounds"),\n        recovery_layers=_integer(model.get("recovery_layers", 1), "recovery_layers"),\n        expert_groups=_integer(model.get("expert_groups", 1), "expert_groups"),\n        activation_checkpointing=model.get("activation_checkpointing", False),',1)
s=s.replace('("parallel_v1", "iterative_tv_h_v1")','("parallel_v1", "iterative_tv_h_v1", "stacked_parallel_tv_h_v1")',1)
marker='    if settings.recovery_rounds not in (1, 3):'
s=s.replace(marker,'''    stacked = settings.recovery_schedule == "stacked_parallel_tv_h_v1"
    if type(settings.activation_checkpointing) is not bool:
        raise ValueError("activation_checkpointing must be boolean")
    if stacked and (settings.recovery_layers not in (1,4) or settings.expert_groups not in (1,8)
                    or settings.recovery_rounds != 1):
        raise ValueError("unsupported stacked layer/group geometry")
    if not stacked and (settings.recovery_layers != 1 or settings.expert_groups != 1 or settings.activation_checkpointing):
        raise ValueError("stacked settings require stacked schedule")
'''+marker,1)
s=s.replace('if settings.recovery_schedule == "iterative_tv_h_v1" and (','if settings.recovery_schedule in ("iterative_tv_h_v1", "stacked_parallel_tv_h_v1") and (',1)
p.write_text(s)
p=Path('integrations/e23/train.py'); s=p.read_text()
s=s.replace('    return identity\n\n\ndef _verify_trial_payload','''    if settings.recovery_schedule == "stacked_parallel_tv_h_v1":
        identity.update(recovery_layers=settings.recovery_layers,expert_groups=settings.expert_groups,
                        group_width=settings.d_model,group_fusion="gated_concat_v1",
                        activation_checkpointing=settings.activation_checkpointing)
    return identity


def _verify_trial_payload''',1)
s=s.replace('raise ValueError("iterative recovery cannot use a history adapter")','raise ValueError(f"{settings.recovery_schedule} recovery cannot use a history adapter")')
s=s.replace('        recovery_rounds=settings.recovery_rounds,','        recovery_rounds=settings.recovery_rounds,\n        recovery_layers=settings.recovery_layers,expert_groups=settings.expert_groups,\n        activation_checkpointing=settings.activation_checkpointing,',1)
marker='    if forecast_mixer not in ("none", "patch_attention_v1"):'
# Build forecast and train forecast both reject changed interface before side effects.
s=s.replace(marker,'''    if settings.recovery_schedule == "stacked_parallel_tv_h_v1" and (forecast_mixer != "none" or forecast_input != "explicit_v1" or history_adapter != "none"):
        raise ValueError("stacked recovery requires original explicit predictor and no adapter")
'''+marker,1)
a=s.index('def train_forecast('); pre=s[:a]; rest=s[a:]
rest=rest.replace('    if not 0 < epochs <= settings.stage2_epochs:', '''    if settings.recovery_schedule == "stacked_parallel_tv_h_v1" and (selection_metric != "mae" or forecast_loss != "mse_plus_0p2_mae" or forecast_mixer != "none" or forecast_input != "explicit_v1"):
        raise ValueError("stacked Stage 2 requires MAE selection and the fixed mixed loss/explicit predictor")
    if not 0 < epochs <= settings.stage2_epochs:''',1)
s=pre+rest
p.write_text(s)
p=Path('src/malr/models/e23/stacked.py'); s=p.read_text().replace('from torch import Tensor, nn','from torch import Tensor, nn\nfrom torch.utils.checkpoint import checkpoint')
a=s.index('            routed=grouped_tv_h(current,'); b=s.index('            diagnostics.append(',a)
s=s[:a]+'''            def run_layer(state_tensor, *, _layer=layer, _cache=caches[index], _allowed=allowed[...,index,:,:]):
                current_inputs=replace(inputs,base_latents=state_tensor)
                output=grouped_tv_h(current_inputs,_layer.groups,h_cache,_cache,_allowed,source_context_allowed)
                fusion=_layer.fusion(state_tensor,output.latents-state_tensor[...,None,:],output.weights,
                    inputs.rho,output.available[...,1:].any(-1))
                return output,fusion
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                routed,fused=checkpoint(run_layer,state,use_reentrant=False)
            else:
                routed,fused=run_layer(state)
'''+s[b:]
p.write_text(s)
