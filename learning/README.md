# Collective Learning Loop for Terra Constellata

This module implements a comprehensive Collective Learning Loop system that uses reinforcement learning to continuously improve the Terra Constellata system's agent coordination and creative capabilities.

## Overview

The Collective Learning Loop integrates multiple learning components to create a self-improving AI system:

- **Workflow Trace Capture**: Records and analyzes LangGraph workflow executions
- **Pattern Analysis**: Discovers recurring patterns in successful workflows
- **Reinforcement Learning**: Optimizes Sentinel Orchestrator decision-making
- **Reward Model Training**: Learns from user feedback (CAT scores)
- **Prompt Optimization**: Improves agent prompts based on performance data
- **Feedback Collection**: Gathers and processes user evaluations

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Workflow       │    │  Pattern        │    │  Feedback       │
│  Traces         │───▶│  Analysis       │───▶│  Collection     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  RL Environment │    │  Reward Model   │    │  Prompt         │
│  (Orchestrator) │    │  Training       │    │  Optimization   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │  Collective     │
                    │  Learning Loop  │
                    │  (Main System)  │
                    └─────────────────┘
```

## Key Components

### 1. Workflow Tracer (`workflow_tracer.py`)
Captures and analyzes LangGraph workflow executions with detailed metrics:
- Node execution times and sequences
- Agent interactions and coordination patterns
- Success rates and performance metrics
- State transitions and decision points

### 2. Pattern Analyzer (`pattern_analyzer.py`)
Discovers optimal patterns from successful workflows:
- Workflow execution patterns
- Agent combination optimization
- Decision-making patterns
- Performance correlation analysis

### 3. RL Environment (`rl_environment.py`)
Gymnasium environment for orchestrator decision-making:
- State space: system status, agent utilization, workflow progress
- Action space: scan, dispatch, coordinate, monitor, optimize
- Reward function: based on efficiency, success, and user satisfaction

### 4. Reward Model (`reward_model.py`)
Machine learning models trained on user feedback:
- Predicts workflow quality and user satisfaction
- Combines multiple ML algorithms (XGBoost, LightGBM, Random Forest)
- Feature engineering from workflow traces and feedback

### 5. Feedback Collector (`feedback_collector.py`)
Comprehensive feedback collection system:
- CAT (Creative Assessment Tool) scores (1-10 scale)
- Multi-dimensional feedback (creativity, usefulness, novelty, quality)
- User satisfaction ratings
- Automated feedback requests for high-performing workflows

### 6. Prompt Optimizer (`prompt_optimizer.py`)
Optimizes agent prompts based on performance data:
- Template-based prompt generation
- Performance tracking and analysis
- Context-aware optimization
- A/B testing capabilities

### 7. Collective Learning Loop (`collective_learning_loop.py`)
Main orchestration system that integrates all components:
- Continuous learning cycles
- Real-time optimization recommendations
- Performance monitoring and metrics
- Automated model training and deployment

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The learning system will automatically create necessary directories for data storage.

## Quick Start

```python
import asyncio
from langchain.llms.base import BaseLLM
from ..agents.sentinel.sentinel_orchestrator import SentinelOrchestrator
from .collective_learning_loop import CollectiveLearningLoop

# Initialize the orchestrator
llm = YourLLMImplementation()  # e.g., OpenAI, Anthropic, etc.
orchestrator = SentinelOrchestrator(llm)

# Create learning system
learning_config = {
    'learning_interval_minutes': 30,
    'min_samples_for_learning': 10,
    'enable_rl_training': True,
    'enable_pattern_discovery': True,
    'enable_prompt_optimization': True
}

learning_loop = CollectiveLearningLoop(orchestrator, learning_config)

# Start continuous learning
await learning_loop.start_learning()

# The system will now continuously learn from workflow executions
# and user feedback to improve performance
```

## Usage Examples

### Basic Learning Loop
```python
# Run the basic example
python -m terra_constellata.learning.example_usage
```

### Feedback Integration
```python
from .feedback_collector import FeedbackCollector, UserFeedback, CATScore

# Initialize feedback collector
collector = FeedbackCollector()

# Create and submit feedback
feedback = UserFeedback("workflow_123", "user_alice")
feedback.add_cat_score(9, {'creativity': 9, 'usefulness': 8})
feedback.set_ratings(satisfaction=0.95, usefulness=0.9)

collector.submit_feedback(feedback)
```

### Pattern Analysis
```python
from .pattern_analyzer import PatternAnalyzer
from .workflow_tracer import WorkflowTracer

# Analyze workflow patterns
tracer = WorkflowTracer()
analyzer = PatternAnalyzer(tracer)

results = analyzer.analyze_patterns()
recommendations = analyzer.get_pattern_recommendations({
    'workflow_type': 'creative_discovery'
})
```

## Configuration

The learning system can be configured through the `learning_config` dictionary:

```python
config = {
    'learning_interval_minutes': 30,        # Learning cycle frequency
    'min_samples_for_learning': 10,         # Minimum samples for training
    'max_learning_cycles_per_day': 48,      # Rate limiting
    'enable_rl_training': True,             # Enable RL optimization
    'enable_pattern_discovery': True,       # Enable pattern analysis
    'enable_prompt_optimization': True,     # Enable prompt optimization
    'feedback_request_threshold': 0.8,      # CAT score threshold for feedback requests
    'trace_storage_path': './traces',       # Workflow trace storage
    'feedback_storage_path': './feedback',  # Feedback data storage
    'model_storage_path': './models'        # Trained model storage
}
```

## API Reference

### CollectiveLearningLoop

#### Methods
- `start_learning()`: Start the continuous learning process
- `stop_learning()`: Stop the learning process
- `get_learning_status()`: Get current learning system status
- `trigger_learning_cycle()`: Manually trigger a learning cycle
- `optimize_orchestrator_decision(state)`: Get optimization recommendations

### FeedbackCollector

#### Methods
- `submit_feedback(feedback)`: Submit user feedback
- `get_feedback(workflow_id)`: Get feedback for a workflow
- `request_feedback(workflow_id, user_id)`: Request feedback from user
- `get_feedback_statistics()`: Get feedback statistics

### PatternAnalyzer

#### Methods
- `analyze_patterns()`: Analyze patterns from workflow traces
- `get_pattern_recommendations(context)`: Get pattern-based recommendations
- `get_optimal_workflow_pattern(type)`: Get best workflow pattern
- `get_optimal_agent_combination(task_type)`: Get best agent combination

## Monitoring and Metrics

The system provides comprehensive monitoring:

```python
# Get learning metrics
status = await learning_loop.get_learning_status()
print(f"Learning cycles: {status['learning_cycles_completed']}")
print(f"Performance improvement: {status['metrics']['avg_performance_improvement']}")

# Export learning data for analysis
learning_loop.export_learning_data("learning_analysis")
```

## Data Storage

The system automatically manages data storage:

- `./traces/`: Workflow execution traces (JSON)
- `./feedback/`: User feedback data (JSON)
- `./models/`: Trained ML models (joblib)
- `./learning_exports/`: Exported analytics data (JSON)

## Performance Optimization

The learning system includes several optimization features:

1. **Batch Processing**: Processes multiple workflows together for efficiency
2. **Incremental Learning**: Updates models without full retraining
3. **Caching**: Caches frequently used patterns and recommendations
4. **Rate Limiting**: Prevents excessive learning cycles
5. **Error Recovery**: Graceful handling of learning failures

## Troubleshooting

### Common Issues

1. **Low Learning Efficiency**
   - Check minimum sample requirements
   - Verify feedback collection is working
   - Ensure workflow traces are being captured

2. **Memory Usage**
   - Configure appropriate batch sizes
   - Enable data cleanup in configuration
   - Monitor storage directory sizes

3. **RL Training Issues**
   - Ensure stable-baselines3 is properly installed
   - Check environment state dimensions
   - Verify reward function is providing meaningful signals

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements:

1. **Multi-Agent Learning**: Distributed learning across agent networks
2. **Meta-Learning**: Learning to learn new tasks faster
3. **Explainable AI**: Better explanations for learning decisions
4. **Federated Learning**: Privacy-preserving collaborative learning
5. **Real-time Adaptation**: Faster adaptation to changing conditions

## Contributing

When contributing to the learning system:

1. Maintain backward compatibility
2. Add comprehensive tests
3. Update documentation
4. Follow the existing code patterns
5. Ensure proper error handling

## License

This learning system is part of the Terra Constellata project and follows the same licensing terms.