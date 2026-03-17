# System Architecture and Workflow

## System Overview

The Crowd-Stampede-Risk-System is a comprehensive video analysis platform designed to detect and predict stampede risks in crowded spaces. The system processes video feeds through multiple analytical stages to produce real-time risk assessments.

## Architecture Diagram

```
Video Input (PETS, Mall, JHU)
        ↓
┌─────────────────────────────────┐
│  Team 1: Dataset Pipeline       │
│  - Load video frames             │
│  - Create spatial grids          │
│  - Generate metadata             │
└──────────────┬──────────────────┘
                ↓
┌─────────────────────────────────┐
│  Team 2: Density Estimation     │
│  - YOLO person detection         │
│  - Density map generation        │
│  - Zone-based aggregation        │
└──────────────┬──────────────────┘
                ↓
        ┌───────┴───────┐
        ↓               ↓
┌──────────────┐  ┌──────────────┐
│  Team 3:     │  │  Team 4:     │
│  Motion      │  │  Risk        │
│  Analysis    │  │  Analysis    │
│  - Optical   │  │  - Bottle-   │
│    flow      │  │    necks     │
│  - Speed     │  │  - SRI calc  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ↓
┌─────────────────────────────────┐
│  Team 5: Visualization          │
│  - Heatmap overlays             │
│  - Risk indicators              │
│  - Dashboard & alerts           │
└─────────────────────────────────┘
        ↓
Output (Visualizations, Logs, Reports)
```

## Data Flow

### Stage 1: Data Ingestion
- Load video frames from datasets (PETS, Mall, JHU)
- Parse ground truth annotations
- Generate spatial grids for analysis

### Stage 2: Detection & Density
- Detect persons using YOLO v5
- Calculate crowd density (persons/m²)
- Create smooth density heatmaps
- Aggregate density by zones

### Stage 3: Motion Analysis
- Calculate optical flow between frames
- Estimate crowd movement speed
- Detect flow patterns and anomalies

### Stage 4: Risk Assessment
- Identify bottleneck regions
- Calculate Stampede Risk Index (SRI)
- Determine alert thresholds
- Generate alerts for high-risk situations

### Stage 5: Visualization
- Overlay heatmaps on video
- Display risk indicators
- Generate dashboard with metrics
- Log all alerts and events

## Module Dependencies

```
dataset_pipeline
    ↓
density_estimation ←─────── risk_analysis
    ↓                              ↑
motion_analysis ──────────────────┘
    ↓
visualization
```

## Configuration Hierarchy

```
configs/
├── grid_config.yaml       # Spatial grid definition
├── risk_config.yaml       # Risk thresholds and SRI weights
└── dataset_config.yaml    # Dataset paths and processing params
```

## Performance Metrics

### Key Metrics Calculated
1. **Density**: Persons per square meter
2. **Speed**: Pixels/frame → Meters/second
3. **SRI**: Composite stampede risk index (0-1)
4. **Flow Variance**: Uniformity of crowd motion

### Risk Levels
| Level | SRI Range | Description |
|-------|-----------|-------------|
| Safe | 0.0-0.3 | Normal operations |
| Elevated | 0.3-0.6 | Increased monitoring |
| Warning | 0.6-0.8 | Alert issued |
| Critical | 0.8-1.0 | Immediate action required |

## Processing Pipeline

### Frame-by-Frame Processing
```
1. Read frame from video
2. Detect persons (YOLO)
3. Calculate local density
4. Compute optical flow
5. Estimate motion speed
6. Detect bottlenecks
7. Calculate SRI
8. Generate alerts if threshold exceeded
9. Render visualizations
10. Log results
```

## Output Structure

```
outputs/
├── density_outputs/      # Density maps and statistics
├── speed_outputs/        # Motion analysis results
├── risk_outputs/         # Risk assessments and SRI values
├── alert_logs/          # Alert history and events
└── demo_videos/         # Annotated output videos
```

## Extension Points

### Adding New Datasets
1. Extend `DatasetLoader` class
2. Implement `load_<dataset_name>()` method
3. Add configuration to `dataset_config.yaml`

### Custom Detection Models
1. Implement `Detector` interface
2. Replace YOLO with alternative model
3. Update configuration

### Additional Risk Factors
1. Create new calculator in `risk_analysis/`
2. Add weight to `SriCalculator`
3. Update thresholds in configuration

## Performance Considerations

### Optimization Strategies
- Frame skipping for real-time processing
- Batch processing for efficiency
- Parallel processing for independent zones
- GPU acceleration (if available)

### Scalability
- Supports multiple camera feeds
- Distributed processing capable
- Configurable grid resolution
- Adjustable processing resolution

## Failure Modes & Recovery

| Failure | Impact | Recovery |
|---------|--------|----------|
| Detection fails | Missing density data | Use interpolation |
| Flow calculation fails | No speed data | Use previous frame data |
| Bottleneck detection fails | Missed congestion | Alert on density alone |
| Alert system fails | No notifications | Log to file |

## Testing & Validation

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### End-to-End Tests
```bash
python main_pipeline.py --test
```

## Monitoring & Logging

All components log to:
- Console (INFO level)
- File: `outputs/logs/pipeline.log`
- Alert file: `outputs/alert_logs/alerts.log`

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

---

For more details, see individual module documentation in the `docs/` directory.
