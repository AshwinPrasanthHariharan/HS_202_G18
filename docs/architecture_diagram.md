# Crowd-Stampede-Risk-System Architecture

## High-Level System Design

```
┌────────────────────────────────────────────────────────────┐
│                     VIDEO INPUT SOURCES                    │
│  (PETS, Mall, JHU Datasets / Real-time Camera Feeds)       │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │  TEAM 1: DATASET PIPELINE      │
        │  • Load frames                 │
        │  • Parse annotations           │
        │  • Create grids                │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │  TEAM 2: DENSITY ESTIMATION    │
        │  • YOLO detection              │
        │  • Density calculation         │
        │  • Heatmap generation          │
        └────────────┬───────────────────┘
                     │
              ┌──────┴──────┐
              │             │
              ↓             ↓
    ┌──────────────────┐  ┌──────────────────┐
    │  TEAM 3: MOTION  │  │  TEAM 4: RISK    │
    │  ANALYSIS        │  │  ANALYSIS        │
    │  • Optical flow  │  │  • Bottleneck    │
    │  • Speed est.    │  │  • SRI calc      │
    └────────┬─────────┘  └─────────┬────────┘
             │                      │
             └──────────┬───────────┘
                        ↓
        ┌────────────────────────────────┐
        │  TEAM 5: VISUALIZATION         │
        │  • Overlay heatmaps            │
        │  • Display alerts              │
        │  • Dashboard rendering         │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │         OUTPUT SYSTEM          │
        │  • Video files                 │
        │  • Alert logs                  │
        │  • Risk reports                │
        │  • Visualizations              │
        └────────────────────────────────┘
```

## System Components

### Core Modules
1. **dataset_pipeline**: Data ingestion and preprocessing
2. **density_estimation**: Crowd counting and mapping
3. **motion_analysis**: Movement tracking and analysis
4. **risk_analysis**: Risk assessment and SRI calculation
5. **visualization**: Real-time visualization and dashboards

### Supporting Infrastructure
- **utils**: Common utilities for all modules
- **configs**: YAML configuration files
- **outputs**: Results directory structure

## Data Types & Structures

### Frame
```python
{
    'id': str,
    'timestamp': float,
    'image': np.ndarray (H×W×3),
    'dataset': str
}
```

### Detection
```python
{
    'x': float,
    'y': float,
    'width': float,
    'height': float,
    'confidence': float,
    'class': str
}
```

### Density Map
```python
{
    'grid': np.ndarray (grid_rows × grid_cols),
    'values': List[float],
    'max_density': float,
    'timestamp': float
}
```

### Risk Assessment
```python
{
    'sri': float,
    'level': str,
    'zone_risks': Dict[str, float],
    'bottlenecks': List[Dict],
    'timestamp': float
}
```

## Processing Pipeline

### Frame Processing Flow
1. **Load**: Read frame from dataset
2. **Detect**: YOLO person detection
3. **Grid**: Map detections to spatial grid
4. **Density**: Calculate density values
5. **Flow**: Compute optical flow
6. **Speed**: Estimate motion speed
7. **Bottleneck**: Detect congestion areas
8. **Risk**: Calculate SRI
9. **Alert**: Generate alerts if needed
10. **Visualize**: Render output frame
11. **Save**: Store results

### Typical Processing Time (per frame)
- Detection: ~100ms (GPU) / ~500ms (CPU)
- Density: ~10ms
- Optical flow: ~50ms
- Risk calculation: ~5ms
- Visualization: ~20ms
- **Total: ~185ms (GPU) / ~585ms (CPU)**

## Configuration System

### grid_config.yaml
- Spatial grid dimensions
- Zone definitions
- Cell sizes
- Interpolation parameters

### risk_config.yaml
- Density thresholds
- Speed thresholds
- Bottleneck parameters
- SRI weights
- Alert settings

### dataset_config.yaml
- Dataset paths
- Processing parameters
- Batch size
- Frame skip rate

## Scalability Considerations

### Horizontal Scaling
- Multiple camera feeds → Parallel processing
- Independent zones → Process in parallel
- Batch processing → GPU efficiency

### Vertical Scaling
- Higher resolution → Better accuracy, slower processing
- More threads → Better CPU utilization
- GPU acceleration → 5-10x speedup

## Extension Architecture

### Adding New Detectors
```python
from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        pass
```

### Adding New Risk Factors
```python
def calculate_custom_risk(self, metric1, metric2) -> float:
    # Custom calculation
    return risk_value
```

### Adding New Visualizations
```python
class CustomOverlay:
    def render(self, frame, data) -> np.ndarray:
        # Custom rendering
        return output_frame
```

## Performance Optimization

### For Real-Time Processing
- Use frame skipping (process every Nth frame)
- Reduce processing resolution
- Enable GPU acceleration
- Use batch processing

### For Accuracy
- Increase grid resolution
- Use full resolution frames
- Disable frame skipping
- Use ensemble detectors

### For Storage
- Compress output videos
- Store only alerts
- Use efficient formats (HDF5)
- Archive old data

## Failure Recovery

### Handling Detection Failures
1. Use previous frame detections
2. Interpolate missing data
3. Use density from previous frame
4. Alert operator

### Handling Flow Failures
1. Use zero flow (no movement)
2. Use averaged flow from previous frames
3. Assume constant velocity
4. Log error for investigation

### Handling System Failures
1. Checkpoint processing state
2. Recovery from last checkpoint
3. Graceful degradation
4. Alternative alert method

---

See individual module documentation for more details.
