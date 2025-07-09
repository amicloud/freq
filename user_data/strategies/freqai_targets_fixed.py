def set_freqai_targets(
    self, dataframe: DataFrame, metadata: dict, **kwargs
) -> DataFrame:
    """
    Version 10: 
    1. &-max-increase: maximum return over the next label_period_candles
    2. &-dd-before-max: binary flag indicating if a 5% drawdown occurs before the max increase
    """
    kernel = self.freqai_info["feature_parameters"]["label_period_candles"]
    close = dataframe["close"]

    # Target 1: max increase over next kernel candles
    # Store max values and positions for verification
    max_values = []
    max_positions = []
    
    for i in range(len(dataframe)):
        future = close.iloc[i + 1 : i + 1 + kernel]
        if future.empty:
            max_values.append(np.nan)
            max_positions.append(np.nan)
        else:
            max_val = future.max()
            max_pos = future.values.argmax()
            max_values.append(max_val)
            max_positions.append(max_pos)
    
    dataframe["&-max-increase"] = (np.array(max_values) / close - 1)

    # Target 2: 5% drawdown before max increase
    def dd_before_max(row):
        idx = row.name
        future = close.iloc[idx + 1 : idx + 1 + kernel]
        if future.empty:
            return np.nan
        
        current_price = close.iloc[idx]
        drawdown_threshold = current_price * 0.95
        max_idx = future.values.argmax()
        
        # Verify max position matches target 1
        expected_max_pos = max_positions[idx]
        if not np.isnan(expected_max_pos) and max_idx != expected_max_pos:
            raise ValueError(f"Max position mismatch at index {idx}: Target 1 found max at position {expected_max_pos}, Target 2 found max at position {max_idx}")
        
        # Verify max value matches target 1
        max_val = future.iloc[max_idx]
        expected_max_val = max_values[idx]
        if not np.isnan(expected_max_val) and not np.isclose(max_val, expected_max_val, rtol=1e-10):
            raise ValueError(f"Max value mismatch at index {idx}: Target 1 found max value {expected_max_val}, Target 2 found max value {max_val}")
        
        # Check if any price before the max position hits the drawdown threshold
        prices_before_max = future.iloc[:max_idx]  # Exclude the max position itself
        
        if len(prices_before_max) == 0:  # Max occurs at first position
            return 0
        
        return int((prices_before_max <= drawdown_threshold).any())

    dataframe["&-dd-before-max"] = dataframe.apply(dd_before_max, axis=1)

    return dataframe
