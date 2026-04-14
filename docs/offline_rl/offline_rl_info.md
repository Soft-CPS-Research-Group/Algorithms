# Offline Reinforcement Learning — Info

## 1. Observations

**Common** - ```month;day_type;hour;carbon_intensity;non_shiftable_load;solar_generation;electrical_storage_soc;net_electricity_consumption;electricity_pricing;electricity_pricing_predicted_1;electricity_pricing_predicted_2;electricity_pricing_predicted_3```

**Charger** - ```electric_vehicle_charger_charger_<id>_connected_state;connected_electric_vehicle_at_charger_charger_<id>_departure_time;connected_electric_vehicle_at_charger_charger_<id>_required_soc_departure;connected_electric_vehicle_at_charger_charger_<id>_soc;connected_electric_vehicle_at_charger_charger_<id>_battery_capacity;electric_vehicle_charger_charger_<id>_incoming_state;incoming_electric_vehicle_at_charger_charger_<id>_estimated_arrival_time```

**Washing Machine** - ```washing_machine_1_start_time_step;washing_machine_1_end_time_step```

### Total

- Agent 0 - 37 (base + 1 charger + washing machine)
- Agent 1, 2, 5, 7, 8, 10, 12, 13, 15, 16 - 28 (base)
- Agent 3, 4, 6, 9, 11 - 35 (base + 1 charger)
- Agent 14 - 42 (base + 2 chargers)
- Total of 534 scalar slots per timestamp when adding all the agents

Note: Agent's charger identifier is equal to the agent's identifier plus one.