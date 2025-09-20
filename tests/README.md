# Terra Constellata Integration Testing Framework

This directory contains comprehensive integration tests, performance benchmarks, and optimization tools for the Terra Constellata system.

## Overview

The testing framework provides:

- **Integration Tests**: Verify interaction between all system components
- **Performance Benchmarks**: Measure and optimize system performance
- **Load Testing**: Simulate production workloads
- **Memory Profiling**: Detect memory leaks and optimize usage
- **Database Optimization**: Query performance analysis and optimization
- **Comprehensive Reporting**: Detailed test results and recommendations

## Test Structure

```
tests/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest fixtures and configuration
├── run_integration_tests.py    # Main test runner script
├── test_integration_system.py  # Full system integration tests
├── test_database_integration.py # Database integration tests
├── test_agent_integration.py   # Agent integration tests
├── test_performance_benchmarks.py # Performance benchmarks
├── test_load_testing.py        # Load testing framework
├── test_memory_profiling.py    # Memory profiling tests
├── test_query_optimization.py  # Database query optimization
└── README.md                   # This file
```

## Quick Start

### Run All Tests

```bash
# From project root
python tests/run_integration_tests.py
```

### Run Specific Test Categories

```bash
# Integration tests only
python -m pytest tests/ -m integration -v

# Performance tests only
python -m pytest tests/ -m performance -v

# Database tests only
python -m pytest tests/ -m database -v

# Agent tests only
python -m pytest tests/ -m agent -v
```

### Run Smoke Tests

```bash
# Quick verification of basic functionality
python tests/run_integration_tests.py --smoke
```

## Test Categories

### Integration Tests (`-m integration`)

- **System Integration**: End-to-end testing of all components
- **Database Integration**: Cross-database data synchronization
- **Agent Integration**: Multi-agent coordination and communication
- **API Integration**: Backend API functionality

### Performance Tests (`-m performance`)

- **Benchmarking**: Response times, throughput, resource usage
- **Load Testing**: Concurrent user simulation
- **Memory Profiling**: Leak detection and optimization
- **Query Optimization**: Database performance analysis

### Component Tests

- **Database Tests** (`-m database`): PostGIS and ArangoDB operations
- **Agent Tests** (`-m agent`): Specialist agent functionality
- **Slow Tests** (`-m slow`): Long-running performance tests

## Configuration

### Environment Variables

Set these environment variables for database connections:

```bash
# PostGIS
export TEST_POSTGIS_HOST=localhost
export TEST_POSTGIS_PORT=5432
export TEST_POSTGIS_DB=terra_constellata_test
export TEST_POSTGIS_USER=postgres
export TEST_POSTGIS_PASSWORD=postgres

# ArangoDB
export TEST_ARANGODB_HOST=localhost
export TEST_ARANGODB_PORT=8529
export TEST_ARANGODB_DB=terra_constellata_test
export TEST_ARANGODB_USER=root
export TEST_ARANGODB_PASSWORD=

# Services
export TEST_BACKEND_HOST=localhost
export TEST_BACKEND_PORT=8000
export TEST_A2A_HOST=localhost
export TEST_A2A_PORT=8080
```

### Pytest Configuration

The `pytest.ini` file contains test configuration:

```ini
[tool:pytest]
testpaths = .
python_files = test_*.py
addopts = --cov=terra_constellata --cov-report=html --cov-report=xml
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    performance: marks tests as performance benchmarks
    database: marks tests as database tests
    agent: marks tests as agent tests
```

## Test Fixtures

### Database Fixtures

- `postgis_connection`: PostGIS database connection
- `ckg_connection`: ArangoDB (CKG) connection

### Service Fixtures

- `a2a_server`: A2A protocol server
- `backend_app`: FastAPI backend application
- `agent_registry`: Initialized agent registry

### Utility Fixtures

- `mock_llm`: Mock LLM for testing without external dependencies
- `sample_data`: Test data for various components
- `temp_directories`: Temporary directories for test files

## Performance Benchmarking

### Running Benchmarks

```bash
# Run all performance tests
python -m pytest tests/test_performance_benchmarks.py -v

# Run specific benchmark
python -m pytest tests/test_performance_benchmarks.py::test_database_query_performance -v
```

### Benchmark Results

Performance tests generate detailed metrics:

- **Response Times**: Average, median, min, max
- **Throughput**: Operations per second
- **Resource Usage**: CPU, memory, I/O
- **Scalability**: Performance under increasing load

## Load Testing

### Running Load Tests

```bash
# Run load tests
python -m pytest tests/test_load_testing.py -v

# Test specific concurrency level
python -m pytest tests/test_load_testing.py::test_api_load_capacity -v
```

### Load Test Scenarios

- **Concurrent Users**: Simulate multiple users
- **Database Load**: High concurrent database operations
- **API Stress**: Endpoint load capacity testing
- **Agent Workload**: Multi-agent concurrent processing

## Memory Profiling

### Running Memory Tests

```bash
# Run memory profiling
python -m pytest tests/test_memory_profiling.py -v
```

### Memory Analysis

The memory profiler detects:

- **Memory Leaks**: Growing memory usage over time
- **Large Objects**: Objects consuming significant memory
- **Garbage Collection**: Efficiency of memory cleanup
- **Optimization Opportunities**: Memory usage recommendations

## Database Optimization

### Running Query Optimization

```bash
# Run query optimization tests
python -m pytest tests/test_query_optimization.py -v
```

### Optimization Features

- **Query Performance**: Execution time analysis
- **Index Effectiveness**: Index usage statistics
- **Query Plans**: Execution plan optimization
- **Connection Pooling**: Optimal pool sizing

## Test Reports

### Automatic Report Generation

The test runner generates comprehensive reports:

```bash
# Generate full test report
python tests/run_integration_tests.py --verbose
```

### Report Contents

Reports include:

- **Test Results**: Pass/fail status for all tests
- **Performance Metrics**: Benchmarks and load test results
- **Coverage Reports**: Code coverage analysis
- **Optimization Recommendations**: Performance improvement suggestions
- **Failure Analysis**: Detailed error information

### Report Location

Reports are saved in `test_reports/` directory:

- `integration_test_report_YYYYMMDD_HHMMSS.json`: Detailed JSON report
- `test_summary_YYYYMMDD_HHMMSS.txt`: Human-readable summary

## Continuous Integration

### GitHub Actions Example

```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run integration tests
      run: python tests/run_integration_tests.py
    - name: Upload coverage reports
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

## Troubleshooting

### Common Issues

1. **Database Connection Failures**
   - Ensure databases are running
   - Check environment variables
   - Verify network connectivity

2. **Service Unavailable**
   - Start required services (PostGIS, ArangoDB, backend)
   - Check service ports
   - Verify service health endpoints

3. **Test Timeouts**
   - Increase timeout values in test configuration
   - Check system resources
   - Run tests individually to isolate issues

4. **Memory Issues**
   - Ensure sufficient RAM (minimum 4GB)
   - Close other applications
   - Run memory profiling to identify leaks

### Debug Mode

Run tests with verbose output for debugging:

```bash
python tests/run_integration_tests.py --verbose
```

## Contributing

### Adding New Tests

1. Create test file in `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use appropriate pytest markers
4. Add fixtures in `conftest.py` if needed
5. Update this README

### Test Best Practices

- Use descriptive test names
- Include docstrings for complex tests
- Mock external dependencies
- Clean up test data
- Use appropriate assertion methods
- Handle test failures gracefully

## Performance Optimization

### Optimization Checklist

- [ ] Database indexes are effective
- [ ] Query plans are optimized
- [ ] Connection pooling is configured
- [ ] Memory usage is monitored
- [ ] Load testing passes thresholds
- [ ] API response times meet requirements
- [ ] Agent processing is efficient
- [ ] System scales under load

### Performance Targets

- **API Response Time**: < 500ms for 95th percentile
- **Database Query Time**: < 100ms average
- **Memory Usage**: < 1GB under normal load
- **Concurrent Users**: Support 100+ simultaneous users
- **Test Coverage**: > 80% code coverage

## Support

For issues with the testing framework:

1. Check test logs for error details
2. Run individual tests to isolate problems
3. Verify environment configuration
4. Check system resources
5. Review test documentation

## License

This testing framework is part of the Terra Constellata project.