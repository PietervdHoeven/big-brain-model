# src/big_brain/training/stopping.py

class EarlyStopper:
    """
    Minimal early-stopping helper.

    Parameters
    ----------
    patience : int
        How many consecutive epochs without improvement to tolerate.
    delta : float, optional
        Minimum change in the monitored metric to qualify as an improvement.
        (Prevents stopping on tiny numerical noise.)
    mode : {"min", "max"}
        - "min"  → smaller is better   (e.g. val_loss)  
        - "max"  → larger is better   (e.g. accuracy)
    """

    def __init__(self, patience: int = 5, delta: float = 0.0, mode: str = "min") -> None:
        assert patience > 0, "patience must be > 0"
        assert mode in {"min", "max"}, "mode must be 'min' or 'max'"

        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.best_score: float | None = None
        self.wait: int = 0
        self.should_stop: bool = False

        self._direction = -1 if mode == "min" else 1  # multiply metric by this for comparison
        

    def step(self, current_score: float) -> bool:
        """
        Call once per validation epoch.

        Returns
        -------
        improved : bool
            True if the current score is a genuine improvement over `best_score`.
            When True, `wait` is reset to 0.
        """
        # normalise so that "higher is better"
        score = self._direction * current_score

        if self.best_score is None:
            # first epoch always counts as improvement
            self.best_score = score
            return True

        if score > self.best_score + self.delta:
            # improvement
            self.best_score = score
            self.wait = 0
            return True

        # no improvement
        self.wait += 1
        if self.wait >= self.patience:
            self.should_stop = True
        return False