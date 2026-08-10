import jax
import jax.numpy as jnp
from utopia.core.state import SimState
from utopia.core.config import SimulationConfig

def _logistics_step(state: SimState, config: SimulationConfig) -> SimState:
    """
    Phase 1.3: Real-Time Telematics & Logistics Bottlenecks.
    Handles the movement of goods in transit and calculates dynamic port dwell times.
    """
    # 1. Advance the shift register: goods move from index i to i-1
    # Goods at index 0 arrive today and get added to inventory.
    # Goods at indices 1..max-1 shift down.
    # The new max-1 index becomes 0 (it will be filled by new purchases in the market step if delayed that long).
    
    in_transit = state.firms.in_transit_inventory  # Shape: (num_firms, num_goods, max_delay)
    
    # Arrivals (what reaches index 0)
    arriving_goods = in_transit[:, :, 0]
    
    # Add arrivals to firm inventory at the freshest index
    new_inventory = state.firms.inventory.at[:, :, -1].add(arriving_goods)
    
    # Shift the register: we take slice from 1 to max, and append zeros at the end
    shifted_in_transit = jnp.concatenate([
        in_transit[:, :, 1:],
        jnp.zeros((config.num_firms, config.num_goods, 1), dtype=jnp.float32)
    ], axis=2)
    
    # 2. Update Port Dynamics & Dwell Time
    # We map ports 1:1 with regions. We sum all goods currently in transit per firm.
    firm_transit_volumes = jnp.sum(shifted_in_transit, axis=(1, 2))
    port_backlog = jax.ops.segment_sum(firm_transit_volumes, state.firms.region_id, num_segments=config.num_regions)
    
    # Effective port capacity adjusted by telematics_multiplier (e.g. storms reduce capacity)
    effective_capacity = state.logistics.port_capacity * state.logistics.telematics_multiplier
    
    # Queue is simply the backlog minus capacity (bounded to 0)
    new_queue = jnp.maximum(0.0, port_backlog - effective_capacity)
    
    # Dwell time calculation based on queue size relative to capacity
    # If queue is large, delay increases.
    # 1.0 means exactly capacity, 2.0 means double capacity.
    # We map this to discrete ticks of delay.
    congestion_ratio = new_queue / (effective_capacity + 1e-5)
    
    # Base delay is 1 tick. Add 1 tick for every full capacity of backlog.
    calculated_delay = 1 + jnp.floor(congestion_ratio).astype(jnp.int32)
    
    # Bound the dwell time to our shift register limits (0 to max_delay - 1)
    # Note: Delay of 0 means immediate arrival next tick (inserted at index 0 of shifted array, 
    # which is the slot that arrives *next* tick because it just shifted).
    # Delay of 1 means it will arrive in 2 ticks. 
    # Let's map dwell_time directly to the index in the shift register.
    # Index 0: Arrives next tick.
    # Index max_delay-1: Arrives in max_delay ticks.
    new_dwell_time = jnp.clip(calculated_delay - 1, 0, config.max_transit_delay - 1)
    
    # Update state
    new_firms = state.firms._replace(
        inventory=new_inventory,
        in_transit_inventory=shifted_in_transit
    )
    
    new_logistics = state.logistics._replace(
        port_queue=new_queue,
        dwell_time=new_dwell_time
    )
    
    return state._replace(firms=new_firms, logistics=new_logistics)
