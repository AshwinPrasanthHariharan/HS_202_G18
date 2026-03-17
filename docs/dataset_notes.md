# Dataset Documentation

## Supported Datasets

This project supports three major crowd analysis datasets:

### 1. PETS Dataset (Performance Evaluation of Tracking and Surveillance)

**Source**: http://www.cvg.reading.ac.uk/PETS09/

**Characteristics**:
- High resolution video sequences
- Multiple camera angles
- Ground truth head annotations (point locations)
- Frame resolution: 768×576 pixels
- Frame rate: 7 FPS

**Contents**:
- S0: Crowd dispersal scenario
- S1-S3: Various crowd gathering scenarios
- S4-S5: Crowd ingress/egress scenarios

**File Structure**:
```
pets_frames/
├── S0_L1_01/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── S1_L1_03/
└── ...

pets_annotations.json
{
  "S0_L1_01/frame_0001": {
    "head_count": 42,
    "points": [[x1, y1], [x2, y2], ...]
  }
}
```

**Usage**:
```python
from src.dataset_pipeline import DatasetLoader
loader = DatasetLoader()
pets_data = loader.load_pets_dataset('./data/pets_frames/')
```

### 2. Mall Dataset (UCF Mall Dataset)

**Characteristics**:
- Surveillance camera footage from shopping mall
- Single overhead camera perspective
- Long-term continuous recording
- Lower resolution but realistic scenario

**File Structure**:
```
mall_frames/
├── frame_00001.jpg
├── frame_00002.jpg
└── ...
```

### 3. JHU Dataset (Johns Hopkins University)

**Source**: http://www.cs.jhu.edu/~mbanerjee/crowd/

**Characteristics**:
- Static crowd images
- Hand-annotated density maps
- Diverse crowd scenarios
- High person count annotations

**File Structure**:
```
jhu_images/
├── img_001.jpg
├── img_002.jpg
└── ...

jhu_annotations.json
{
  "img_001": {
    "head_count": 89,
    "density_map": "path/to/density_001.npy"
  }
}
```

## Annotation Format

### Head Count Annotations
```json
{
  "frame_id": {
    "head_count": 42,
    "timestamp": 1.5,
    "scene": "S0_L1_01",
    "confidence": 0.95
  }
}
```

### Point Annotations (Head Locations)
```json
{
  "frame_id": {
    "points": [
      {"x": 100.5, "y": 150.3, "conf": 0.98},
      {"x": 200.2, "y": 300.1, "conf": 0.97},
      ...
    ],
    "total_count": 42
  }
}
```

### Density Map Annotations
```json
{
  "img_id": {
    "density_map_file": "density_maps/img_001_density.npy",
    "resolution": [512, 512],
    "head_count": 89,
    "max_density": 0.45
  }
}
```

## Data Statistics

### PETS Dataset
- Total frames: ~8000 (varies by sequence)
- Resolution: 768×576
- FPS: 7
- Average crowd size: 10-50 persons
- Annotation quality: High (manual annotation)

### Mall Dataset
- Total frames: ~2000
- Resolution: 320×240
- FPS: 2.5
- Average crowd size: 1-100+ persons
- Scene: Overhead surveillance

### JHU Dataset
- Total images: ~120
- Resolution: Variable
- Average crowd size: 50-150 persons
- Annotation: Density maps + head count

## Preprocessing Steps

### Frame Normalization
```python
from src.dataset_pipeline import FrameUtils
frame = FrameUtils.normalize_frame(frame)
```

### Resizing
```python
target_size = (512, 512)
resized = FrameUtils.resize_frame(frame, target_size)
```

### Color Space Conversion
```python
gray = FrameUtils.convert_color(frame, cv2.COLOR_BGR2GRAY)
```

## Data Loading Example

```python
from src.dataset_pipeline import DatasetLoader, GridGenerator
from src.density_estimation import AnnotationReader
import cv2

# Load dataset
loader = DatasetLoader('configs/dataset_config.yaml')
pets_data = loader.load_pets_dataset('./data/pets_frames/')

# Load annotations
ann_reader = AnnotationReader('./data/annotations/pets_annotations.json')

# Process frame
frame_id = 'S0_L1_01/frame_0001'
frame = cv2.imread(f'./data/pets_frames/{frame_id}.jpg')

# Get annotations
annotations = ann_reader.get_frame_annotations(frame_id)
head_count = ann_reader.get_head_count(frame_id)
point_annotations = ann_reader.get_point_annotations(frame_id)

# Create grid
grid_gen = GridGenerator((frame.shape[0], frame.shape[1]), (8, 8))

print(f"Frame: {frame_id}")
print(f"Head count: {head_count}")
print(f"Points: {len(point_annotations)}")
```

## Custom Dataset Integration

To add a new dataset:

1. **Create loader method**:
```python
def load_custom_dataset(self, data_dir: str) -> Dict:
    """Load custom dataset."""
    # Implementation
    return dataset_dict
```

2. **Add to configuration**:
```yaml
# configs/dataset_config.yaml
datasets:
  custom:
    name: "Custom Dataset"
    paths: "./data/custom/"
```

3. **Create annotations**:
Create annotations in one of the supported formats above.

## Dataset Quality Metrics

### Annotation Accuracy
- PETS: ±1-2 pixels localization error
- JHU: Head count within ±5 persons

### Coverage
- PETS: 8000+ frames across multiple scenarios
- Mall: 2000+ frames of continuous surveillance
- JHU: 120 diverse crowd images

### Diversity
- Scenarios: Dispersal, gathering, ingress, egress, queuing
- Perspectives: Side view, overhead
- Lighting: Various (indoor/outdoor)
- Crowd density: 1-150+ persons

## Dataset Download

### Automatic Download
```bash
python gd_pull.py
```

### Manual Download
Visit the official dataset websites (links above) and organize as shown in File Structure sections.

## Troubleshooting

### Issue: Annotation file not found
**Solution**: Ensure JSON files are in `data/annotations/` directory

### Issue: Frame loading fails
**Solution**: Check frame paths match annotation keys

### Issue: Memory error with large datasets
**Solution**: Use `frame_skip` parameter in `dataset_config.yaml`

---

**Last Updated**: March 2026
**Format Version**: 1.0
