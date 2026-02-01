"""Main FastAPI application for the trading dashboard."""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from .api.routes import router as api_router
from .config import settings

# Create FastAPI app
app = FastAPI(
    title="Trading Dashboard",
    description="Real-time trading dashboard with market data, sentiment analysis, and Fibonacci patterns",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Include API routes
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Trading Dashboard",
            "default_stocks": settings.DEFAULT_STOCKS,
            "default_crypto": settings.DEFAULT_CRYPTO,
            "default_forex": settings.DEFAULT_FOREX
        }
    )


@app.get("/metals", response_class=HTMLResponse)
async def metals_dashboard(request: Request):
    """Render the precious metals trading dashboard."""
    return templates.TemplateResponse(
        "metals_dashboard.html",
        {
            "request": request,
            "title": "Precious Metals Trading - XAU/USD & XAG/USD"
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
