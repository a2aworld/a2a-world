"""
Inspiration Engine Core

The main orchestration class for the Inspiration Engine that combines
novelty detection, data ingestion, prompt ranking, and A2A communication.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
import os

from .algorithms import NoveltyDetector
from .data_ingestion import DataIngestor
from .prompt_ranking import PromptRanker, CreativePrompt, PromptRanking
from .a2a_integration import A2AInspirationClient

logger = logging.getLogger(__name__)


class InspirationEngine:
    """
    Main Inspiration Engine class that orchestrates novelty detection,
    data processing, and agent communication for creative inspiration.
    """

    def __init__(
        self,
        ckg_config: Optional[Dict[str, Any]] = None,
        postgis_config: Optional[Dict[str, Any]] = None,
        a2a_config: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
    ):
        """
        Initialize the Inspiration Engine

        Args:
            ckg_config: Configuration for CKG database connection
            postgis_config: Configuration for PostGIS database connection
            a2a_config: Configuration for A2A protocol client
            config_file: Path to configuration file (optional)
        """
        self.config = self._load_config(config_file)

        # Override with provided configs
        if ckg_config:
            self.config["ckg"].update(ckg_config)
        if postgis_config:
            self.config["postgis"].update(postgis_config)
        if a2a_config:
            self.config["a2a"].update(a2a_config)

        # Initialize components
        self.novelty_detector = NoveltyDetector()
        self.data_ingestor = DataIngestor(
            ckg_host=self.config["ckg"]["host"],
            ckg_db=self.config["ckg"]["database"],
            ckg_user=self.config["ckg"]["username"],
            ckg_password=self.config["ckg"]["password"],
            postgis_host=self.config["postgis"]["host"],
            postgis_port=self.config["postgis"]["port"],
            postgis_db=self.config["postgis"]["database"],
            postgis_user=self.config["postgis"]["user"],
            postgis_password=self.config["postgis"]["password"],
        )

        self.prompt_ranker = PromptRanker(self.novelty_detector)
        self.a2a_client = None  # Will be initialized when needed

        # Engine state
        self.is_initialized = False
        self.last_analysis_time = None
        self.analysis_cache = {}

        logger.info("Inspiration Engine initialized")

    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "ckg": {
                "host": os.getenv("CKG_HOST", "http://localhost:8529"),
                "database": os.getenv("CKG_DATABASE", "ckg_db"),
                "username": os.getenv("CKG_USERNAME", "root"),
                "password": os.getenv("CKG_PASSWORD", ""),
            },
            "postgis": {
                "host": os.getenv("POSTGIS_HOST", "localhost"),
                "port": int(os.getenv("POSTGIS_PORT", "5432")),
                "database": os.getenv("POSTGIS_DATABASE", "terra_constellata"),
                "user": os.getenv("POSTGIS_USER", "postgres"),
                "password": os.getenv("POSTGIS_PASSWORD", ""),
            },
            "a2a": {
                "server_url": os.getenv("A2A_SERVER_URL", "http://localhost:8080"),
                "agent_name": os.getenv("A2A_AGENT_NAME", "inspiration_engine"),
                "timeout": float(os.getenv("A2A_TIMEOUT", "30.0")),
            },
            "engine": {
                "cache_ttl_minutes": int(os.getenv("CACHE_TTL_MINUTES", "60")),
                "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "100")),
                "novelty_threshold": float(os.getenv("NOVELTY_THRESHOLD", "0.7")),
            },
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                self._merge_configs(default_config, file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")

        return default_config

    def _merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ):
        """Recursively merge override config into base config"""
        for key, value in override_config.items():
            if (
                key in base_config
                and isinstance(base_config[key], dict)
                and isinstance(value, dict)
            ):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value

    async def initialize(self) -> bool:
        """
        Initialize all engine components and establish connections

        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize data connections
            if not self.data_ingestor.connect_databases():
                logger.error("Failed to connect to databases")
                return False

            # Initialize A2A client if configured
            if self.config["a2a"]["server_url"]:
                self.a2a_client = A2AInspirationClient(
                    server_url=self.config["a2a"]["server_url"],
                    agent_name=self.config["a2a"]["agent_name"],
                    timeout=self.config["a2a"]["timeout"],
                )

            self.is_initialized = True
            logger.info("Inspiration Engine fully initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Inspiration Engine: {e}")
            return False

    async def shutdown(self):
        """Shutdown the engine and close connections"""
        if self.data_ingestor:
            self.data_ingestor.disconnect_databases()

        if self.a2a_client:
            await self.a2a_client.client.disconnect()

        self.is_initialized = False
        logger.info("Inspiration Engine shutdown complete")

    async def analyze_novelty(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
        algorithms: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze data for novelty using all available algorithms

        Args:
            data: Data to analyze for novelty
            context: Additional context for analysis
            algorithms: Specific algorithms to use (default: all)
            use_cache: Whether to use cached results

        Returns:
            Dictionary with novelty analysis results
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        # Check cache first
        cache_key = self._generate_cache_key(data, context, algorithms)
        if use_cache and cache_key in self.analysis_cache:
            cached_result = self.analysis_cache[cache_key]
            if self._is_cache_valid(cached_result["timestamp"]):
                logger.info("Returning cached novelty analysis")
                return cached_result

        try:
            # Detect novelty
            novelty_scores = self.novelty_detector.detect_novelty(
                data, context, algorithms
            )

            # Calculate combined score
            combined_score = self.novelty_detector.calculate_combined_score(
                novelty_scores
            )

            # Prepare result
            result = {
                "novelty_scores": {
                    name: {
                        "score": score.score,
                        "algorithm": score.algorithm,
                        "confidence": score.confidence,
                        "metadata": score.metadata,
                    }
                    for name, score in novelty_scores.items()
                },
                "combined_score": combined_score,
                "is_novel": combined_score
                >= self.config["engine"]["novelty_threshold"],
                "analysis_timestamp": datetime.utcnow(),
                "data_summary": self._summarize_data(data),
                "context_used": context or {},
            }

            # Cache result
            if use_cache:
                self.analysis_cache[cache_key] = result
                self._cleanup_cache()

            self.last_analysis_time = datetime.utcnow()

            # Broadcast novelty alert if highly novel
            if result["is_novel"] and self.a2a_client:
                await self._broadcast_novelty_alert(result)

            logger.info(
                f"Novelty analysis complete. Combined score: {combined_score:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to analyze novelty: {e}")
            return {
                "error": str(e),
                "novelty_scores": {},
                "combined_score": 0.0,
                "is_novel": False,
                "analysis_timestamp": datetime.utcnow(),
            }

    async def generate_inspiration(
        self,
        domain: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
        num_prompts: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate creative inspiration prompts based on data analysis

        Args:
            domain: Creative domain (mythology, geography, etc.)
            context: Additional context for inspiration
            constraints: Constraints for prompt generation
            num_prompts: Number of prompts to generate

        Returns:
            Dictionary with inspiration results
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        try:
            # Get relevant data for the domain
            domain_data = await self._get_domain_data(domain)

            # Generate base prompts
            base_prompts = self._generate_base_prompts(
                domain, domain_data, context, constraints
            )

            # Convert to CreativePrompt objects and rank them
            prompt_dicts = [
                {
                    "id": f"prompt_{i}",
                    "content": prompt,
                    "domain": domain,
                    "context": context or {},
                    "metadata": {
                        "generated_by": "inspiration_engine",
                        "domain_data_used": bool(domain_data),
                    },
                }
                for i, prompt in enumerate(base_prompts)
            ]

            ranking = self.prompt_ranker.rank_prompts(prompt_dicts)

            # Get top prompts
            top_prompts = self.prompt_ranker.get_top_prompts(ranking, top_n=num_prompts)

            result = {
                "domain": domain,
                "total_prompts_generated": len(base_prompts),
                "ranking": {
                    "diversity_score": ranking.diversity_score,
                    "ranking_criteria": ranking.ranking_criteria,
                    "top_prompt_count": len(top_prompts),
                },
                "top_prompts": [
                    {
                        "id": prompt.id,
                        "content": prompt.content,
                        "creative_potential": prompt.creative_potential,
                        "combined_novelty": prompt.combined_score,
                        "domain": prompt.domain,
                    }
                    for prompt in top_prompts
                ],
                "generation_timestamp": datetime.utcnow(),
                "data_sources_used": list(domain_data.keys()) if domain_data else [],
            }

            logger.info(
                f"Generated {len(top_prompts)} inspiration prompts for {domain} domain"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to generate inspiration: {e}")
            return {
                "error": str(e),
                "domain": domain,
                "total_prompts_generated": 0,
                "top_prompts": [],
                "generation_timestamp": datetime.utcnow(),
            }

    async def process_recent_data(
        self, time_window_hours: int = 24, domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process recently added data for novelty detection

        Args:
            time_window_hours: Time window to look back
            domains: Specific domains to focus on

        Returns:
            Dictionary with processing results
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        try:
            # Get recent data
            recent_data = self.data_ingestor.get_recent_data(time_window_hours)

            analysis_results = {}

            # Analyze CKG data
            if recent_data["ckg_recent"]:
                for collection, df in recent_data["ckg_recent"].items():
                    if df is not None and not df.empty:
                        domain = self._infer_domain_from_collection(collection)
                        if domains is None or domain in domains:
                            analysis = await self.analyze_novelty(
                                df, {"domain": domain, "data_type": "ckg"}
                            )
                            analysis_results[f"ckg_{collection}"] = analysis

            # Analyze PostGIS data
            if (
                recent_data["postgis_recent"] is not None
                and not recent_data["postgis_recent"].empty
            ):
                analysis = await self.analyze_novelty(
                    recent_data["postgis_recent"],
                    {"domain": "geography", "data_type": "postgis"},
                )
                analysis_results["postgis_recent"] = analysis

            # Generate insights from analyses
            insights = self._generate_insights_from_analyses(analysis_results)

            result = {
                "time_window_hours": time_window_hours,
                "data_processed": {
                    "ckg_collections": list(recent_data["ckg_recent"].keys()),
                    "postgis_records": len(recent_data["postgis_recent"])
                    if recent_data["postgis_recent"] is not None
                    else 0,
                },
                "analysis_results": analysis_results,
                "insights": insights,
                "processing_timestamp": datetime.utcnow(),
            }

            logger.info(f"Processed recent data from {len(analysis_results)} sources")
            return result

        except Exception as e:
            logger.error(f"Failed to process recent data: {e}")
            return {
                "error": str(e),
                "time_window_hours": time_window_hours,
                "data_processed": {"ckg_collections": [], "postgis_records": 0},
                "analysis_results": {},
                "insights": [],
                "processing_timestamp": datetime.utcnow(),
            }

    async def collaborative_session(
        self,
        topic: str,
        participants: Optional[List[str]] = None,
        duration_minutes: int = 30,
    ) -> Dict[str, Any]:
        """
        Initiate a collaborative inspiration session with other agents

        Args:
            topic: Session topic
            participants: List of participating agents
            duration_minutes: Session duration

        Returns:
            Session results
        """
        if not self.a2a_client:
            return {"error": "A2A client not configured"}

        try:
            if participants is None:
                participants = ["mythology_agent", "geography_agent", "narrative_agent"]

            session_result = await self.a2a_client.collaborative_inspiration_session(
                topic, participants, duration_minutes
            )

            logger.info(
                f"Initiated collaborative session on '{topic}' with {len(participants)} participants"
            )
            return session_result

        except Exception as e:
            logger.error(f"Failed to initiate collaborative session: {e}")
            return {"error": str(e)}

    async def share_findings(
        self, findings: Dict[str, Any], target_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Share analysis findings with other agents

        Args:
            findings: Findings to share
            target_agents: Agents to share with

        Returns:
            Sharing results
        """
        if not self.a2a_client:
            return {"error": "A2A client not configured"}

        try:
            responses = await self.a2a_client.share_novelty_findings(
                findings, target_agents
            )

            result = {
                "findings_shared": findings,
                "target_agents": target_agents or ["all_agents"],
                "responses_received": len(responses),
                "sharing_timestamp": datetime.utcnow(),
            }

            logger.info(
                f"Shared findings with {len(target_agents) if target_agents else 'all'} agents"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to share findings: {e}")
            return {"error": str(e)}

    def _generate_cache_key(
        self,
        data: Any,
        context: Optional[Dict[str, Any]],
        algorithms: Optional[List[str]],
    ) -> str:
        """Generate cache key for analysis results"""
        data_hash = hash(str(data)[:1000])  # Limit data size for hashing
        context_hash = hash(json.dumps(context or {}, sort_keys=True))
        algo_hash = hash(str(sorted(algorithms or [])))

        return f"{data_hash}_{context_hash}_{algo_hash}"

    def _is_cache_valid(self, cache_timestamp: datetime) -> bool:
        """Check if cached result is still valid"""
        ttl_minutes = self.config["engine"]["cache_ttl_minutes"]
        return (datetime.utcnow() - cache_timestamp) < timedelta(minutes=ttl_minutes)

    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = datetime.utcnow()
        ttl_minutes = self.config["engine"]["cache_ttl_minutes"]

        expired_keys = [
            key
            for key, result in self.analysis_cache.items()
            if (current_time - result["timestamp"]) > timedelta(minutes=ttl_minutes)
        ]

        for key in expired_keys:
            del self.analysis_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """Generate a summary of input data"""
        if isinstance(data, dict):
            return {"type": "dict", "keys": list(data.keys()), "size": len(data)}
        elif isinstance(data, list):
            return {
                "type": "list",
                "length": len(data),
                "sample_items": data[:3] if len(data) > 0 else [],
            }
        elif hasattr(data, "shape"):  # DataFrame or numpy array
            return {
                "type": "dataframe",
                "shape": data.shape,
                "columns": list(data.columns) if hasattr(data, "columns") else None,
            }
        else:
            return {
                "type": str(type(data).__name__),
                "size": len(str(data)) if hasattr(data, "__len__") else "unknown",
            }

    async def _broadcast_novelty_alert(self, analysis_result: Dict[str, Any]):
        """Broadcast novelty alert to other agents"""
        if not self.a2a_client:
            return

        try:
            await self.a2a_client.broadcast_inspiration_update(
                "novelty_discovered",
                {
                    "novelty_score": analysis_result["combined_score"],
                    "data_summary": analysis_result["data_summary"],
                    "analysis_timestamp": analysis_result[
                        "analysis_timestamp"
                    ].isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to broadcast novelty alert: {e}")

    async def _get_domain_data(self, domain: str) -> Dict[str, Any]:
        """Get relevant data for a specific domain"""
        domain_data = {}

        try:
            if domain == "mythology":
                ckg_data = self.data_ingestor.get_ckg_data(
                    ["MythologicalEntity", "CulturalConcept"]
                )
                domain_data.update(ckg_data)
            elif domain == "geography":
                postgis_data = self.data_ingestor.get_postgis_data()
                domain_data["postgis"] = postgis_data
                ckg_data = self.data_ingestor.get_ckg_data(["GeographicFeature"])
                domain_data.update(ckg_data)
            elif domain == "cultural":
                ckg_data = self.data_ingestor.get_ckg_data(
                    ["CulturalConcept", "TextSource"]
                )
                domain_data.update(ckg_data)

            return domain_data

        except Exception as e:
            logger.error(f"Failed to get domain data for {domain}: {e}")
            return {}

    def _generate_base_prompts(
        self,
        domain: str,
        domain_data: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        constraints: Optional[List[str]],
    ) -> List[str]:
        """Generate base creative prompts for a domain"""
        prompts = []

        # Domain-specific prompt templates
        templates = {
            "mythology": [
                "Explore the mythological significance of {entity} in relation to {concept}",
                "Create a narrative that bridges {myth1} and {myth2} through {theme}",
                "Imagine how {deity} would respond to {modern_concept}",
                "Develop a ritual that honors {entity} while addressing {contemporary_issue}",
            ],
            "geography": [
                "Describe the journey from {location1} to {location2} through the eyes of {traveler}",
                "Explore how the geography of {place} has shaped {cultural_aspect}",
                "Imagine the stories that {landmark} could tell about {historical_period}",
                "Design a path that connects {site1}, {site2}, and {site3} with thematic significance",
            ],
            "cultural": [
                "Examine how {tradition} has evolved to address {modern_challenge}",
                "Create a dialogue between {culture1} and {culture2} perspectives on {topic}",
                "Explore the symbolism of {symbol} across different {cultural_contexts}",
                "Design a contemporary interpretation of {traditional_element} for {audience}",
            ],
        }

        domain_templates = templates.get(domain, templates["mythology"])

        # Generate prompts using available data
        for template in domain_templates[:5]:  # Limit to 5 base prompts
            try:
                prompt = self._fill_template(template, domain_data, context)
                if prompt:
                    prompts.append(prompt)
            except Exception as e:
                logger.warning(f"Failed to generate prompt from template: {e}")

        return prompts

    def _fill_template(
        self,
        template: str,
        domain_data: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Fill template placeholders with actual data"""
        # This is a simplified implementation - in practice, you'd want more sophisticated
        # template filling based on the available data
        filled = template

        # Replace placeholders with sample data
        placeholders = {
            "{entity}": "ancient deity",
            "{concept}": "modern ethics",
            "{myth1}": "creation story",
            "{myth2}": "hero's journey",
            "{theme}": "transformation",
            "{deity}": "Athena",
            "{modern_concept}": "artificial intelligence",
            "{contemporary_issue}": "climate change",
            "{location1}": "sacred mountain",
            "{location2}": "mystical forest",
            "{traveler}": "wandering scholar",
            "{place}": "ancient city",
            "{cultural_aspect}": "social customs",
            "{landmark}": "forgotten temple",
            "{historical_period}": "bronze age",
            "{site1}": "oracle",
            "{site2}": "marketplace",
            "{site3}": "burial ground",
            "{tradition}": "storytelling",
            "{modern_challenge}": "digital age",
            "{culture1}": "indigenous wisdom",
            "{culture2}": "scientific method",
            "{topic}": "nature",
            "{symbol}": "sacred geometry",
            "{cultural_contexts}": "spiritual traditions",
            "{traditional_element}": "ceremony",
            "{audience}": "young explorers",
        }

        for placeholder, replacement in placeholders.items():
            filled = filled.replace(placeholder, replacement)

        return filled

    def _infer_domain_from_collection(self, collection: str) -> str:
        """Infer domain from CKG collection name"""
        domain_map = {
            "MythologicalEntity": "mythology",
            "GeographicFeature": "geography",
            "CulturalConcept": "cultural",
            "TextSource": "cultural",
            "GeospatialPoint": "geography",
        }
        return domain_map.get(collection, "general")

    def _generate_insights_from_analyses(
        self, analysis_results: Dict[str, Any]
    ) -> List[str]:
        """Generate insights from multiple analysis results"""
        insights = []

        # Find highly novel results
        novel_results = [
            (name, result)
            for name, result in analysis_results.items()
            if result.get("is_novel", False)
        ]

        if novel_results:
            insights.append(
                f"Discovered {len(novel_results)} novel patterns across data sources"
            )

        # Find most novel result
        if analysis_results:
            most_novel = max(
                analysis_results.items(), key=lambda x: x[1].get("combined_score", 0)
            )
            if most_novel[1].get("combined_score", 0) > 0.5:
                insights.append(
                    f"Highest novelty score ({most_novel[1]['combined_score']:.3f}) found in {most_novel[0]}"
                )

        return insights

    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and statistics"""
        return {
            "is_initialized": self.is_initialized,
            "last_analysis_time": self.last_analysis_time.isoformat()
            if self.last_analysis_time
            else None,
            "cache_size": len(self.analysis_cache),
            "config_summary": {
                "ckg_connected": self.data_ingestor.ckg_db is not None,
                "postgis_connected": self.data_ingestor.postgis_db is not None,
                "a2a_configured": self.a2a_client is not None,
            },
            "engine_version": "1.0.0",
        }
