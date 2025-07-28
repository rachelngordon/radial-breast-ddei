import numpy as np

# --- Main Evaluation Plan ---
# Increased fixed budget of 320 total spokes.
MAIN_EVALUATION_PLAN = [
    {
        "spokes_per_frame": 16,
        "num_frames": 20, # 16 * 20 = 320 total spokes
        "slice": slice(1, 21), # Wide window
        "description": "High temporal resolution"
    },
    {
        "spokes_per_frame": 20,
        "num_frames": 16, # 20 * 16 = 320 total spokes
        "slice": slice(3, 19), # Medium-wide window
        "description": "Good temporal resolution"
    },
    {
        "spokes_per_frame": 32,
        "num_frames": 10, # 32 * 10 = 320 total spokes
        "slice": slice(5, 15), # Narrow window centered on enhancement
        "description": "Standard temporal resolution"
    },
    {
        "spokes_per_frame": 40,
        "num_frames": 8,  # 40 * 8 = 320 total spokes
        "slice": slice(5, 13), # Very narrow window on peak
        "description": "Low temporal resolution"
    }
]

# --- Stress Test Plan ---
# Designed to push the limits with very few spokes per frame.
# This has a different (lower) total spoke budget.
STRESS_TEST_PLAN = [
    {
        "spokes_per_frame": 8,
        "num_frames": 22, # 8 * 22 = 176 total spokes
        "slice": slice(0, 22), # The entire 22-frame duration
        "description": "Stress test: max temporal points, min spokes"
    }
]


# --- Example Usage ---

# Assume high-res data is pre-loaded
# images_high_res = ... # Shape (width, height, 22)

# First, run the fair comparisons
print("--- Running Main Evaluation (Budget: 320 spokes) ---")
for config in MAIN_EVALUATION_PLAN:
    # ... (same logic as before: slice data, run recons, get metrics)
    spokes = config["spokes_per_frame"]
    time_slice = config["slice"]
    # images_for_test = images_high_res[:, :, time_slice]
    print(f"  Testing {spokes} spokes/frame with {images_for_test.shape[2]} frames.")


# Then, run the stress test to highlight DL model's advantage
print("\n--- Running Stress Test (Budget: 176 spokes) ---")
for config in STRESS_TEST_PLAN:
    # ... (same logic as before)
    spokes = config["spokes_per_frame"]
    time_slice = config["slice"]
    # images_for_test = images_high_res[:, :, time_slice]
    print(f"  Testing {spokes} spokes/frame with {images_for_test.shape[2]} frames.")