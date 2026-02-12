"""Verification runner for Stage 1 Backboard lifecycle hardening."""
import os
import json
import logging
from Stage_1 import run

# Configure provider
os.environ["SPECFORGE_PROVIDER"] = "backboard"
# os.environ["BACKBOARD_API_KEY"] = "YOUR_API_KEY"
os.environ["SPECFORGE_MODEL"] = "gpt-4o"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

state = {
    "idea_raw": "A system that generates a random haiku about space."
}

class MockContext:
    def __init__(self, state):
        self._state = state
        self._logger = logging.getLogger("SpecForge_Run_Verification")
    def get(self, key):
        return self._state.get(key)
    def set(self, key, value):
        self._state[key] = value
    @property
    def logger(self):
        return self._logger

ctx = MockContext(state)
success = run(ctx)

print("\n=== Final Result ===")
print(f"Success: {success}")
if success:
    print(json.dumps(state["spec"], indent=2))
