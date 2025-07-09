def calculate_reward(self, action: int) -> float:
    """
    Simplified reward function focused on clear, stable learning signals.
    
    :param action: int = The action made by the agent for the current candle.
    :return: float = the reward to give to the agent for current step
    """
    # First, penalize if the action is not valid
    if not self._is_valid(action):
        return -1.0
    
    # Get current PnL
    pnl = self.get_unrealized_profit()
    
    # Ensure previous pnl is 0 at start
    if self._last_trade_tick is None:
        self._previous_pnl = 0.0
    
    # Get trade duration for time-based penalties
    if self._last_trade_tick is not None:
        trade_duration = self._current_tick - self._last_trade_tick
    else:
        trade_duration = 0
    
    max_trade_duration = self.rl_config.get("max_trade_duration_candles", 100)
    
    # Calculate PnL change since last tick
    if trade_duration <= 1:
        self._previous_pnl = 0.0
        pnl_change = 0
    else:
        pnl_change = pnl - self._previous_pnl
        self._previous_pnl = pnl
    
    # Initialize reward
    reward = 0.0
    
    # === SIMPLIFIED ACTION-SPECIFIC REWARDS ===
    
    # Buying when neutral (entering position)
    if action == Actions.Buy.value and self._position == Positions.Neutral:
        reward = 0.0  # Neutral - let the holding rewards do the work
        
    # Holding a long position - this is where most learning happens
    elif action == Actions.Neutral.value and self._position == Positions.Long:
        # Simple reward based on PnL change
        if pnl_change > 0:
            reward = 1.0  # Fixed positive reward for gains
        elif pnl_change < 0:
            reward = -1.0  # Fixed negative reward for losses
        else:
            reward = 0.0  # Neutral for no change
        
        # Small time penalty to encourage decision making
        if trade_duration > max_trade_duration * 0.8:
            reward -= 0.5
    
    # Staying neutral (not trading)
    elif action == Actions.Neutral.value and self._position == Positions.Neutral:
        reward = 0.0  # Neutral - not penalizing patience
    
    # Selling a long position (closing position) - BIG REWARD EVENT
    elif action == Actions.Sell.value and self._position == Positions.Long:
        # This is the main learning signal - final trade outcome
        if pnl > 0.01:  # Profitable trade (>1%)
            reward = 10.0  # Strong positive reward
        elif pnl > 0:  # Small profit
            reward = 5.0   # Moderate positive reward
        elif pnl > -0.01:  # Small loss (<1%)
            reward = -2.0  # Small penalty
        else:  # Large loss
            reward = -5.0  # Moderate penalty
        
        # Bonus for quick profitable trades
        if pnl > 0 and trade_duration <= max_trade_duration * 0.3:
            reward += 2.0
        
        # Extra penalty for holding losers too long
        if pnl < 0 and trade_duration > max_trade_duration * 0.7:
            reward -= 2.0
    
    return reward