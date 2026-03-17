# Example Usage and Quick Start

## Running the Complete Pipeline

```bash
# 1. Activate environment
source env/bin/activate

# 2. Download datasets
python gd_pull.py

# 3. Run main pipeline
python main_pipeline.py

# 4. Check results
ls outputs/
```

## Using Individual Modules

### Dataset Pipeline

```python
from src.dataset_pipeline import DatasetLoader, GridGenerator

# Load dataset
loader = DatasetLoader('configs/dataset_config.yaml')
pets_data = loader.load_pets_dataset('./data/pets_frames/')

# Create grid
grid_gen = GridGenerator((720, 1280), grid_size=(8, 8))
```

### Density Estimation

```python
from src.density_estimation import YoloDetector, DensityCalculator

# Initialize detector
detector = YoloDetector(model_path='models/yolov5s.pt')

# Detect persons
detections = detector.detect(frame)

# Calculate density
calc = DensityCalculator(frame.shape[1], frame.shape[0])
head_count = calc.calculate_head_count(detections)
```

### Motion Analysis

```python
from src.motion_analysis import OpticalFlow, SpeedEstimator

# Calculate optical flow
flow_calc = OpticalFlow()
flow = flow_calc.calculate_farneback(prev_frame, curr_frame)

# Estimate speed
speed_est = SpeedEstimator()
avg_speed = speed_est.calculate_average_speed(flow)
```

### Risk Analysis

```python
from src.risk_analysis import SriCalculator

# Calculate SRI
sri_calc = SriCalculator()
sri = sri_calc.calculate_sri(
    density_norm=0.5,
    speed_norm=0.3,
    bottleneck_norm=0.2,
    flow_variance_norm=0.1
)
print(f"Risk Level: {sri_calc.get_risk_level(sri)}")
```

### Visualization

```python
from src.visualization import HeatmapOverlay, Dashboard

# Create overlay
overlay = HeatmapOverlay(colormap='jet')
output_frame = overlay.overlay_heatmap(frame, density_map, alpha=0.4)

# Render dashboard
dashboard = Dashboard(1920, 1080)
dashboard.render_frame(output_frame, (0, 0), (960, 540))
display = dashboard.get_display()
```

## Jupyter Notebook Example

```python
import cv2
import numpy as np
from pathlib import Path
from src.dataset_pipeline import DatasetLoader, FrameUtils
from src.density_estimation import YoloDetector, DensityCalculator
from src.visualization import HeatmapOverlay

# Load configuration
from src.utils import IoUtils
config = IoUtils.load_json('configs/dataset_config.yaml')

# Load frame
frame_path = Path('./data/pets_frames/') / 'frame_0001.jpg'
frame = cv2.imread(str(frame_path))

# Detect persons
detector = YoloDetector()
detections = detector.detect(frame)

# Calculate density
calc = DensityCalculator(frame.shape[1], frame.shape[0])
density_map = calc.generate_density_map(detections, (8, 8))

# Visualize
overlay = HeatmapOverlay()
result = overlay.overlay_heatmap(frame, density_map, alpha=0.4)

# Display
cv2.imshow('Result', result)
cv2.waitKey(0)
```

---

For more examples, see `notebooks/` directory.
