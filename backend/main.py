from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from .database import engine, Base
from .api import (
    content,
    multimedia,
    pipeline,
    maps,
    artworks,
    codex,
    workflow,
    feedback,
    agents,
)
import time
import uuid
try:
    import psutil
except ImportError:
    psutil = None
from ..logging_config import app_logger, log_request, log_response, log_error
from ..metrics import record_request, get_metrics, update_system_metrics
from ..error_tracking import set_user_context, set_request_context, capture_exception

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Galactic Storybook CMS", description="Headless CMS for Terra Constellata"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Set Sentry context
    set_request_context(
        request_id=request_id, method=request.method, path=request.url.path
    )

    # Set user context if available
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        set_user_context(user_id=user_id)

    # Log incoming request
    log_request(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        user_id=user_id,
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Log response
        log_response(
            request_id=request_id,
            status_code=response.status_code,
            response_time=process_time,
        )

        # Record metrics
        record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=process_time,
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        process_time = time.time() - start_time

        # Capture exception in Sentry
        capture_exception(e, request_id=request_id, endpoint=request.url.path)

        log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=str(e.__traceback__),
            user_id=user_id,
        )
        raise


# Include routers
app.include_router(content.router, prefix="/api/content", tags=["content"])
app.include_router(multimedia.router, prefix="/api/multimedia", tags=["multimedia"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(maps.router, prefix="/api/maps", tags=["maps"])
app.include_router(artworks.router, prefix="/api/artworks", tags=["artworks"])
app.include_router(codex.router, prefix="/api/codex", tags=["codex"])
app.include_router(workflow.router, prefix="/api/workflow", tags=["workflow"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])


@app.get("/")
async def root():
    return {"message": "Galactic Storybook CMS API"}


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    from datetime import datetime

    # Basic health info
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",  # Should be read from config
    }

    if psutil:
        health_data["uptime"] = time.time() - psutil.boot_time()
        health_data["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        }
    else:
        health_data["uptime"] = "N/A (psutil not available)"
        health_data["system"] = "N/A (psutil not available)"

    app_logger.info("Health check requested", extra=health_data)
    return health_data


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Update system metrics before returning
    update_system_metrics()

    # Return metrics in Prometheus format
    from fastapi.responses import PlainTextResponse

    metrics_data = get_metrics()
    return PlainTextResponse(
        content=metrics_data.decode("utf-8"), media_type="text/plain; charset=utf-8"
    )


# Codex and Workflow integration
def initialize_systems():
    """Initialize and integrate all system components."""
    codex_manager = None
    workflow_tracer = None
    cocreation_workflow = None

    try:
        from ..codex import CodexManager
        from ..learning import WorkflowTracer
        from ..workflow import CoCreationWorkflow
        from ..agents.sentinel import SentinelOrchestrator
        from ..agents.apprentice import ApprenticeAgent

        # Initialize Codex manager
        codex_manager = CodexManager("./codex_data")

        # Initialize workflow tracer and integrate
        workflow_tracer = WorkflowTracer("./traces")
        workflow_tracer.set_codex_manager(codex_manager)
        codex_manager.integrate_workflow_tracer(workflow_tracer)

        # Set Codex manager in API
        codex.set_codex_manager(codex_manager)

        # Initialize Sentinel Orchestrator
        sentinel = SentinelOrchestrator(llm=None)  # Will be set by individual tasks

        # Initialize Apprentice Agent
        apprentice = ApprenticeAgent(llm=None)  # Will be set by individual tasks

        # Initialize Co-Creation Workflow
        cocreation_workflow = CoCreationWorkflow(
            sentinel=sentinel,
            apprentice=apprentice,
            codex=codex_manager,
            chatbot=None,  # Optional, can be added later
        )

        # Set workflow in API
        workflow.set_cocreation_workflow(cocreation_workflow)

        app_logger.info("All systems initialized successfully")

    except ImportError as e:
        app_logger.warning(f"Some systems not available: {e}")

    return codex_manager, workflow_tracer, cocreation_workflow


# Initialize systems on startup
codex_manager, workflow_tracer, cocreation_workflow = initialize_systems()


# Initialize agent health monitoring
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup."""
    from .api.agents import start_health_monitoring
    from fastapi import BackgroundTasks
    background_tasks = BackgroundTasks()
    start_health_monitoring(background_tasks)
    app_logger.info("Agent health monitoring initialized")
