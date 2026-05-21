# Phase 6 15s Scaling Configs

Configs para medir e reduzir custo de treino MADDPG em datasets de 15 segundos.

Objetivo destes ficheiros: diagnosticar performance, nao tirar conclusoes finais
de KPI. Todos usam profiling runtime em `tracking.runtime_profiling_enabled`.
Os logs/diagnostics estao deliberadamente raros para nao distorcer runtime:
`runtime_profiling_interval=1024`, MLflow off, action/reward diagnostics off.

## Perfis

- `remote_20260521_profile_15s_v2g_maddpg_v48_update20_amp_small_seed123.yaml`
  - V2G ativo;
  - update MADDPG a cada 20 steps de ambiente;
  - actor update a cada 4 updates;
  - AMP ligado;
  - actor `256-128`, critic `512-256`.

- `remote_20260521_profile_15s_v2g_maddpg_v48_update60_amp_small_seed123.yaml`
  - igual ao anterior, mas update MADDPG a cada 60 steps;
  - usado para medir limite agressivo de reducao de custo.

- `remote_20260521_profile_15s_no_v2g_maddpg_v48_update20_amp_small_seed123.yaml`
  - mesma receita update20, mas dataset sem V2G.

## Metricas A Observar

No export/local metrics procurar:

- `Runtime/env_step_seconds`;
- `Runtime/observation_encoding_seconds`;
- `Runtime/reward_shaping_seconds`;
- `Runtime/agent_update_seconds`;
- `Runtime/diagnostics_build_seconds`;
- `MADDPG/runtime_replay_sample_seconds`;
- `MADDPG/runtime_tensor_transfer_seconds`;
- `MADDPG/runtime_target_compute_seconds`;
- `MADDPG/runtime_critic_update_seconds`;
- `MADDPG/runtime_actor_update_seconds`;
- `MADDPG/runtime_metrics_build_seconds`.

Estas metricas dizem se o bottleneck esta no simulador/wrapper, no replay,
na transferencia GPU, no critic/actor, ou no logging.
