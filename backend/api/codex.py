"""
Codex API Router for the Galactic Storybook CMS.

This module provides REST API endpoints for accessing and managing
the Agent's Codex data, including contributions, strategies, chapters,
and knowledge entries.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import Codex components (will be initialized by main app)
codex_manager = None


def set_codex_manager(manager):
    """Set the Codex manager instance."""
    global codex_manager
    codex_manager = manager


router = APIRouter()


@router.get("/contributions/", response_model=List[Dict[str, Any]])
async def get_contributions(
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    contribution_type: Optional[str] = Query(
        None, description="Filter by contribution type"
    ),
    limit: int = Query(50, description="Maximum number of results"),
):
    """
    Get agent contributions.

    Query parameters:
    - agent_name: Filter by specific agent
    - contribution_type: Filter by contribution type
    - limit: Maximum results to return
    """
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        if agent_name:
            contributions = codex_manager.archival_system.get_contributions_by_agent(
                agent_name
            )
        elif contribution_type:
            contributions = codex_manager.archival_system.get_contributions_by_type(
                contribution_type
            )
        else:
            # Return all contributions (limited)
            contributions = list(codex_manager.archival_system.contributions.values())[
                :limit
            ]

        return [c.to_dict() for c in contributions]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving contributions: {str(e)}"
        )


@router.get("/contributions/{contribution_id}", response_model=Dict[str, Any])
async def get_contribution(contribution_id: str):
    """Get a specific contribution by ID."""
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    contribution = codex_manager.archival_system.get_contribution(contribution_id)
    if not contribution:
        raise HTTPException(status_code=404, detail="Contribution not found")

    return contribution.to_dict()


@router.get("/strategies/", response_model=List[Dict[str, Any]])
async def get_strategies(
    strategy_type: Optional[str] = Query(None, description="Filter by strategy type"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    limit: int = Query(50, description="Maximum number of results"),
):
    """
    Get documented strategies.

    Query parameters:
    - strategy_type: Filter by strategy type
    - created_by: Filter by creator agent
    - limit: Maximum results to return
    """
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        if strategy_type:
            strategies = codex_manager.archival_system.get_strategies_by_type(
                strategy_type
            )
        elif created_by:
            # Get all strategies and filter by creator
            all_strategies = list(codex_manager.archival_system.strategies.values())
            strategies = [s for s in all_strategies if s.created_by == created_by]
        else:
            strategies = list(codex_manager.archival_system.strategies.values())[:limit]

        return [s.to_dict() for s in strategies]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving strategies: {str(e)}"
        )


@router.get("/strategies/{strategy_id}", response_model=Dict[str, Any])
async def get_strategy(strategy_id: str):
    """Get a specific strategy by ID."""
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    strategy = codex_manager.archival_system.get_strategy(strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    return strategy.to_dict()


@router.get("/chapters/", response_model=List[Dict[str, Any]])
async def get_chapters(
    theme: Optional[str] = Query(None, description="Filter by theme"),
    agent_name: Optional[str] = Query(None, description="Filter by agent"),
    published_only: bool = Query(True, description="Return only published chapters"),
    limit: int = Query(20, description="Maximum number of results"),
):
    """
    Get legacy chapters.

    Query parameters:
    - theme: Filter by narrative theme
    - agent_name: Filter by featured agent
    - published_only: Return only published chapters
    - limit: Maximum results to return
    """
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        if theme:
            chapters = codex_manager.chapter_generator.get_chapters_by_theme(theme)
        elif agent_name:
            chapters = codex_manager.chapter_generator.get_chapters_by_agent(agent_name)
        else:
            chapters = list(codex_manager.chapter_generator.chapters.values())

        # Filter by published status
        if published_only:
            chapters = [c for c in chapters if c.published]

        return [c.to_dict() for c in chapters[:limit]]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chapters: {str(e)}"
        )


@router.get("/chapters/{chapter_id}", response_model=Dict[str, Any])
async def get_chapter(chapter_id: str):
    """Get a specific chapter by ID."""
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    chapter = codex_manager.chapter_generator.get_chapter(chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    return chapter.to_dict()


@router.post("/chapters/{chapter_id}/publish")
async def publish_chapter(chapter_id: str):
    """Publish a chapter to the Galactic Storybook."""
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    success = codex_manager.chapter_generator.publish_chapter(chapter_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Chapter not found or already published"
        )

    return {"message": f"Chapter {chapter_id} published successfully"}


@router.get("/knowledge/", response_model=List[Dict[str, Any]])
async def search_knowledge(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    min_confidence: float = Query(0.0, description="Minimum confidence score"),
    limit: int = Query(20, description="Maximum number of results"),
):
    """
    Search the knowledge base.

    Query parameters:
    - query: Search terms
    - category: Filter by knowledge category
    - tags: Filter by tags
    - min_confidence: Minimum confidence score
    - limit: Maximum results to return
    """
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        results = codex_manager.knowledge_base.search_knowledge(
            query=query,
            category=category,
            tags=tags,
            min_confidence=min_confidence,
            limit=limit,
        )

        return [entry.to_dict() for entry in results]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error searching knowledge: {str(e)}"
        )


@router.get("/attribution/", response_model=Dict[str, Any])
async def get_attribution_summary(
    agent_name: Optional[str] = Query(None, description="Filter by agent"),
    days: Optional[int] = Query(None, description="Look back period in days"),
):
    """
    Get attribution summary.

    Query parameters:
    - agent_name: Filter by specific agent
    - days: Number of days to look back
    """
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        summary = codex_manager.attribution_tracker.get_contribution_summary(
            agent_name=agent_name, days=days
        )
        return summary

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving attribution: {str(e)}"
        )


@router.get("/statistics/", response_model=Dict[str, Any])
async def get_codex_statistics():
    """Get comprehensive Codex statistics."""
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        stats = codex_manager.get_codex_statistics()
        return stats.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving statistics: {str(e)}"
        )


@router.get(
    "/learning-recommendations/{agent_name}", response_model=List[Dict[str, Any]]
)
async def get_learning_recommendations(
    agent_name: str,
    context: str = Query(..., description="Current context or task type"),
):
    """
    Get learning recommendations for an agent.

    Path parameters:
    - agent_name: Name of the agent

    Query parameters:
    - context: Current context or task type
    """
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        recommendations = codex_manager.get_learning_recommendations(
            agent_name, context
        )
        return recommendations

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting recommendations: {str(e)}"
        )


@router.post("/search/")
async def search_codex(
    query: str = Query(..., description="Search query"),
    search_type: str = Query("all", description="Type of search"),
    **filters,
):
    """
    Search across all Codex components.

    Query parameters:
    - query: Search terms
    - search_type: Type of search (all, contributions, strategies, knowledge, chapters)
    - Additional filters as needed
    """
    if not codex_manager:
        raise HTTPException(status_code=503, detail="Codex system not available")

    try:
        results = codex_manager.search_codex(query, search_type, **filters)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching Codex: {str(e)}")
