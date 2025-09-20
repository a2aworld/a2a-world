# Terra Constellata Monitoring and Logging Setup

This document provides comprehensive information about the monitoring, logging, and observability setup for the Terra Constellata project.

## Overview

The monitoring stack includes:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Dashboard and visualization
- **Loki**: Log aggregation
- **Promtail**: Log shipping
- **AlertManager**: Alert management
- **Sentry**: Error tracking
- **cAdvisor**: Container metrics
- **Node Exporter**: System metrics

## Quick Start

### 1. Start Monitoring Stack

```bash
cd terra-constellata
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Access Services

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093
- **Loki**: http://localhost:3100

### 3. Configure Environment Variables

Create a `.env` file with the following variables:

```bash
# Sentry Configuration
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
ENVIRONMENT=development

# Application Configuration
RELEASE_VERSION=1.0.0
```

## Logging

### Structured Logging with Loguru

The project uses Loguru for structured logging with the following features:

- **Console Output**: Colored, human-readable logs for development
- **File Output**: Structured logs for production
- **JSON Serialization**: For log aggregation systems
- **Custom Log Levels**: BUSINESS and METRICS levels for specific use cases

#### Usage Examples

```python
from logging_config import app_logger, log_request, log_response, log_error

# Basic logging
app_logger.info("Application started")

# Structured logging
log_request("req-123", "GET", "/api/health", "user-456")
log_response("req-123", 200, 0.045)

# Error logging
log_error("DatabaseError", "Connection timeout", user_id="user-456")

# Business events
from logging_config import log_business_event
log_business_event("workflow_started", {"workflow_id": "wf-123"})

# Metrics logging
from logging_config import log_metrics
log_metrics("response_time", 0.045, {"endpoint": "/api/health"})
```

### Log Files

- `logs/terra_constellata.log`: Main application logs
- `logs/errors.log`: Error-only logs with stack traces

## Metrics

### Prometheus Metrics

The application exposes the following metrics:

#### HTTP Metrics
- `terra_constellata_requests_total`: Total number of requests
- `terra_constellata_request_duration_seconds`: Request duration histogram
- `terra_constellata_errors_total`: Total number of errors

#### Business Metrics
- `terra_constellata_workflows_started_total`: Workflow starts
- `terra_constellata_workflows_completed_total`: Workflow completions
- `terra_constellata_artworks_generated_total`: Artworks generated
- `terra_constellata_feedback_received_total`: User feedback received

#### System Metrics
- `terra_constellata_memory_usage_bytes`: Memory usage
- `terra_constellata_cpu_usage_percent`: CPU usage
- `terra_constellata_active_connections`: Active connections

### Metrics Endpoints

- **Application Metrics**: `GET /metrics` (Prometheus format)
- **Health Check**: `GET /health` (JSON format)

## Dashboards

### Pre-configured Dashboards

1. **Terra Constellata Overview**: Main dashboard with key metrics
   - API request rates and response times
   - System resource usage
   - Error rates
   - Workflow statistics

### Creating Custom Dashboards

1. Access Grafana at http://localhost:3000
2. Click "Create" → "Dashboard"
3. Add panels with Prometheus queries

Example queries:
```promql
# Request rate
rate(terra_constellata_requests_total[5m])

# Error rate
rate(terra_constellata_errors_total[5m]) / rate(terra_constellata_requests_total[5m])

# 95th percentile response time
histogram_quantile(0.95, rate(terra_constellata_request_duration_seconds_bucket[5m]))
```

## Alerting

### Alert Rules

The system includes the following alerts:

#### Critical Alerts
- **High Error Rate**: Error rate > 10% for 5 minutes
- **Backend Down**: Backend service unavailable for 1 minute
- **Database Down**: Database unavailable for 1 minute
- **High CPU Usage**: CPU usage > 90% for 5 minutes

#### Warning Alerts
- **Slow API Response**: 95th percentile > 5 seconds for 5 minutes
- **High Memory Usage**: Memory usage > 8GB for 5 minutes
- **High Workflow Failure Rate**: Failure rate > 20% for 5 minutes

#### Info Alerts
- **Low User Feedback**: Less than 1 feedback per hour

### Alert Configuration

Alerts are configured in `monitoring/rules/terra_constellata_alerts.yml` and can be customized based on your requirements.

## Error Tracking

### Sentry Integration

The application integrates with Sentry for comprehensive error tracking:

- **Automatic Error Capture**: All unhandled exceptions
- **Performance Monitoring**: Transaction tracing
- **User Context**: User information in error reports
- **Custom Error Reporting**: Structured error data

### Configuration

Set the `SENTRY_DSN` environment variable to enable Sentry:

```bash
export SENTRY_DSN="https://your-dsn@sentry.io/project-id"
```

## Feedback Collection

### API Endpoints

The system provides feedback collection endpoints:

- `POST /api/feedback/submit`: General user feedback
- `POST /api/feedback/workflow/{workflow_id}`: Workflow-specific feedback
- `POST /api/feedback/artwork/{artwork_id}`: Artwork-specific feedback
- `GET /api/feedback/stats`: Feedback statistics
- `GET /api/feedback/recent`: Recent feedback entries

### Feedback Integration

Feedback is automatically integrated into the learning loop:

1. **Collection**: User feedback stored in structured format
2. **Analysis**: Feedback analyzed for patterns and trends
3. **Learning**: Insights fed back into the system for improvement
4. **Reporting**: Feedback statistics available via API

## Log Aggregation

### Loki Setup

Logs are aggregated using Loki with the following features:

- **Multi-line Support**: Proper handling of stack traces
- **Label-based Filtering**: Filter logs by service, level, etc.
- **Grafana Integration**: View logs alongside metrics
- **Query Language**: LogQL for advanced log queries

### Log Queries

```logql
# All error logs
{job="terra_constellata"} |= "ERROR"

# Request logs with specific status
{job="terra_constellata"} |= "API Request" | json | status_code >= 400

# Workflow completion logs
{job="terra_constellata"} |= "workflow" |= "completed"
```

## Configuration Files

### Monitoring Configuration

- `monitoring/prometheus.yml`: Prometheus configuration
- `monitoring/grafana/`: Grafana provisioning
- `monitoring/loki-config.yml`: Loki configuration
- `monitoring/promtail-config.yml`: Promtail configuration
- `monitoring/alertmanager.yml`: AlertManager configuration
- `monitoring/rules/`: Alert rules

### Application Configuration

- `logging_config.py`: Logging configuration
- `metrics.py`: Metrics collection
- `error_tracking.py`: Sentry integration

## Troubleshooting

### Common Issues

1. **Grafana not loading dashboards**
   - Check Grafana logs: `docker logs terra_constellata_grafana`
   - Verify provisioning configuration

2. **Prometheus not scraping metrics**
   - Check targets: http://localhost:9090/targets
   - Verify service is running and exposing metrics

3. **Logs not appearing in Loki**
   - Check Promtail status: `docker logs terra_constellata_promtail`
   - Verify log file paths and permissions

4. **Alerts not firing**
   - Check AlertManager: http://localhost:9093
   - Verify Prometheus rules are loaded

### Health Checks

Use the health check endpoint to verify system status:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-09-02T10:54:17.358Z",
  "version": "1.0.0",
  "uptime": 3600.5,
  "system": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "disk_usage": 60.3
  }
}
```

## Development

### Local Development Setup

1. Start monitoring stack:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

2. Set environment variables:
```bash
export ENVIRONMENT=development
export SENTRY_DSN=your-development-dsn
```

3. Run application with logging:
```bash
python -m uvicorn backend.main:app --reload --log-level info
```

### Testing Monitoring

1. **Generate test traffic**:
```bash
# Health check requests
for i in {1..10}; do curl http://localhost:8000/health; done

# Error simulation
curl http://localhost:8000/nonexistent-endpoint
```

2. **Check metrics**:
```bash
curl http://localhost:8000/metrics
```

3. **View logs in Grafana**:
- Go to Explore → Select Loki datasource
- Query: `{job="terra_constellata"}`

## Production Deployment

### Environment Variables

```bash
# Required
SENTRY_DSN=https://prod-dsn@sentry.io/project-id
ENVIRONMENT=production
RELEASE_VERSION=1.0.0

# Optional
GF_SECURITY_ADMIN_PASSWORD=secure-password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Security Considerations

1. **Change default passwords**:
   - Grafana admin password
   - AlertManager credentials

2. **Network security**:
   - Restrict access to monitoring ports
   - Use HTTPS for external access

3. **Data retention**:
   - Configure appropriate retention policies
   - Set up log rotation

### Scaling

For high-traffic deployments:

1. **Prometheus**: Use federation or remote write
2. **Loki**: Use distributed deployment
3. **Grafana**: Use load balancer for multiple instances
4. **AlertManager**: Use clustering for high availability

## Support

For issues with the monitoring setup:

1. Check service logs: `docker-compose -f docker-compose.monitoring.yml logs`
2. Verify configuration files
3. Check network connectivity between services
4. Review Prometheus targets and alerting rules

## Contributing

When adding new metrics or logs:

1. Update this documentation
2. Add appropriate labels and metadata
3. Test in development environment
4. Update alerting rules if necessary
5. Ensure proper error handling