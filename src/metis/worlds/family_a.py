class A0Environment:
    def reset(self) -> str:
        return "start"

    def step(self, action: str) -> str:
        if action == "right":
            return "success"

        if action == "left":
            return "failure"

        raise ValueError(f"Invalid action: {action}")
