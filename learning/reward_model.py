"""
Reward Model Training System

This module implements reward model training for the Collective Learning Loop,
using user feedback (CAT scores) and system performance metrics to train
predictive models that can estimate the quality and value of agent actions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

import joblib
import os

from .workflow_tracer import WorkflowTrace
from .pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)


class RewardDataPoint:
    """Represents a single reward training data point."""

    def __init__(
        self,
        workflow_id: str,
        action_sequence: List[str],
        performance_metrics: Dict[str, float],
        user_feedback: Optional[Dict[str, Any]] = None,
    ):
        self.workflow_id = workflow_id
        self.action_sequence = action_sequence
        self.performance_metrics = performance_metrics
        self.user_feedback = user_feedback or {}
        self.timestamp = datetime.utcnow()

        # Calculate composite reward score
        self.reward_score = self._calculate_reward_score()

    def _calculate_reward_score(self) -> float:
        """Calculate the overall reward score for this data point."""
        score = 0.0

        # Performance-based rewards
        if "workflow_efficiency" in self.performance_metrics:
            score += self.performance_metrics["workflow_efficiency"] * 0.3

        if "agent_coordination_score" in self.performance_metrics:
            score += self.performance_metrics["agent_coordination_score"] * 0.2

        if "task_completion_rate" in self.performance_metrics:
            score += self.performance_metrics["task_completion_rate"] * 0.3

        # User feedback rewards
        if "cat_score" in self.user_feedback:
            cat_score = self.user_feedback["cat_score"]
            # Normalize CAT score (assuming 1-10 scale) to 0-1
            normalized_cat = (cat_score - 1) / 9.0
            score += normalized_cat * 0.2

        if "user_satisfaction" in self.user_feedback:
            score += self.user_feedback["user_satisfaction"] * 0.1

        # Penalty for errors
        if self.performance_metrics.get("error_count", 0) > 0:
            score -= 0.1 * min(self.performance_metrics["error_count"], 5)

        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    def to_features(self) -> Dict[str, Any]:
        """Convert data point to feature dictionary for model training."""
        features = {
            "workflow_id": self.workflow_id,
            "num_actions": len(self.action_sequence),
            "avg_action_length": np.mean(
                [len(action) for action in self.action_sequence]
            )
            if self.action_sequence
            else 0,
            "unique_actions": len(set(self.action_sequence)),
            "workflow_efficiency": self.performance_metrics.get(
                "workflow_efficiency", 0.0
            ),
            "agent_coordination_score": self.performance_metrics.get(
                "agent_coordination_score", 0.0
            ),
            "task_completion_rate": self.performance_metrics.get(
                "task_completion_rate", 0.0
            ),
            "avg_response_time": self.performance_metrics.get(
                "avg_agent_response_time", 0.0
            ),
            "error_count": self.performance_metrics.get("error_count", 0),
            "has_user_feedback": 1 if self.user_feedback else 0,
            "cat_score": self.user_feedback.get("cat_score", 0),
            "user_satisfaction": self.user_feedback.get("user_satisfaction", 0.0),
            "reward_score": self.reward_score,
        }

        # Add action type frequencies
        action_counts = {}
        for action in self.action_sequence:
            action_counts[f"action_{action}"] = (
                action_counts.get(f"action_{action}", 0) + 1
            )

        features.update(action_counts)

        return features


class RewardModel:
    """Machine learning model for predicting rewards from workflow features."""

    def __init__(self, model_type: str = "xgboost", model_path: str = "./models"):
        """
        Initialize the reward model.

        Args:
            model_type: Type of ML model ('xgboost', 'lightgbm', 'random_forest', 'linear')
            model_path: Path to save/load models
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False

        # Create model directory
        os.makedirs(model_path, exist_ok=True)

        # Initialize the appropriate model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the machine learning model."""
        if self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost is not installed. Please install it to use XGBoost models.")
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("lightgbm is not installed. Please install it to use LightGBM models.")
            self.model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
        elif self.model_type == "linear":
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(
        self, training_data: List[RewardDataPoint], test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the reward model.

        Args:
            training_data: List of training data points
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with training metrics
        """
        if len(training_data) < 10:
            logger.warning("Insufficient training data for reward model")
            return {"error": "insufficient_data"}

        # Convert to feature DataFrame
        features_list = [dp.to_features() for dp in training_data]
        df = pd.DataFrame(features_list)

        # Prepare features and target
        feature_cols = [
            col for col in df.columns if col not in ["workflow_id", "reward_score"]
        ]
        self.feature_columns = feature_cols

        X = df[feature_cols]
        y = df["reward_score"]

        # Handle missing values
        X = X.fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if self.model_type in ["xgboost", "lightgbm"]:
            self.model.fit(X_train_scaled, y_train)
        else:
            self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.is_trained = True

        # Save model
        self.save_model()

        metrics = {
            "mse": mse,
            "r2_score": r2,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(feature_cols),
        }

        logger.info(f"Reward model trained. MSE: {mse:.4f}, R²: {r2:.4f}")
        return metrics

    def predict_reward(self, features: Dict[str, Any]) -> float:
        """
        Predict reward score for given features.

        Args:
            features: Dictionary of feature values

        Returns:
            Predicted reward score (0-1)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default reward")
            return 0.5

        # Create feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))

        # Scale features
        feature_vector_scaled = self.scaler.transform([feature_vector])

        # Make prediction
        prediction = self.model.predict(feature_vector_scaled)[0]

        # Clamp to valid range
        return max(0.0, min(1.0, prediction))

    def save_model(self, filename: Optional[str] = None):
        """Save the trained model to disk."""
        if not filename:
            filename = f"reward_model_{self.model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        model_path = os.path.join(self.model_path, f"{filename}.joblib")
        scaler_path = os.path.join(self.model_path, f"{filename}_scaler.joblib")

        if self.model:
            joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        # Save feature columns
        feature_path = os.path.join(self.model_path, f"{filename}_features.txt")
        with open(feature_path, "w") as f:
            f.write("\n".join(self.feature_columns))

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str) -> bool:
        """Load a trained model from disk."""
        try:
            model_path = os.path.join(self.model_path, f"{filename}.joblib")
            scaler_path = os.path.join(self.model_path, f"{filename}_scaler.joblib")
            feature_path = os.path.join(self.model_path, f"{filename}_features.txt")

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)

            with open(feature_path, "r") as f:
                self.feature_columns = [line.strip() for line in f.readlines()]

            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return {}

        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance_scores))


class RewardModelTrainer:
    """System for training and managing reward models."""

    def __init__(self, model_path: str = "./models"):
        self.model_path = model_path
        self.current_model: Optional[RewardModel] = None
        self.training_history: List[Dict[str, Any]] = []
        self.data_points: List[RewardDataPoint] = []

        # Create model directory
        os.makedirs(model_path, exist_ok=True)

    def add_training_data(
        self,
        workflow_traces: List[WorkflowTrace],
        user_feedback: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Add training data from workflow traces and user feedback.

        Args:
            workflow_traces: List of workflow traces
            user_feedback: Dictionary mapping workflow IDs to user feedback
        """
        user_feedback = user_feedback or {}

        for trace in workflow_traces:
            if not trace.success_metrics:
                continue

            # Extract action sequence from trace
            action_sequence = []
            for node in trace.nodes_executed:
                action_sequence.append(node.get("node_name", "unknown"))

            # Get user feedback for this workflow
            feedback = user_feedback.get(trace.workflow_id, {})

            # Create data point
            data_point = RewardDataPoint(
                workflow_id=trace.workflow_id,
                action_sequence=action_sequence,
                performance_metrics=trace.success_metrics,
                user_feedback=feedback,
            )

            self.data_points.append(data_point)

        logger.info(f"Added {len(workflow_traces)} training data points")

    def train_model(
        self, model_type: str = "xgboost", min_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Train a new reward model.

        Args:
            model_type: Type of model to train
            min_samples: Minimum number of samples required for training

        Returns:
            Training results and metrics
        """
        if len(self.data_points) < min_samples:
            return {
                "success": False,
                "error": f"Insufficient training data: {len(self.data_points)} < {min_samples}",
            }

        # Create and train model
        model = RewardModel(model_type, self.model_path)
        metrics = model.train(self.data_points)

        if "error" not in metrics:
            self.current_model = model

            # Record training history
            training_record = {
                "timestamp": datetime.utcnow(),
                "model_type": model_type,
                "training_samples": len(self.data_points),
                "metrics": metrics,
                "model_filename": f"reward_model_{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            }
            self.training_history.append(training_record)

            return {"success": True, "metrics": metrics, "model": model}
        else:
            return {"success": False, "error": metrics["error"]}

    def get_reward_prediction(self, workflow_features: Dict[str, Any]) -> float:
        """Get reward prediction for workflow features."""
        if not self.current_model:
            logger.warning("No trained model available")
            return 0.5

        return self.current_model.predict_reward(workflow_features)

    def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate the current model's performance."""
        if not self.current_model or not self.current_model.is_trained:
            return {"error": "No trained model available"}

        # Use recent data points for evaluation
        if len(self.data_points) < 10:
            return {"error": "Insufficient evaluation data"}

        # Get predictions for recent data
        recent_data = self.data_points[-min(50, len(self.data_points)) :]
        predictions = []
        actuals = []

        for dp in recent_data:
            pred = self.current_model.predict_reward(dp.to_features())
            actual = dp.reward_score
            predictions.append(pred)
            actuals.append(actual)

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

        return {
            "mse": mse,
            "r2_score": r2,
            "mae": mae,
            "evaluation_samples": len(recent_data),
        }

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the training history."""
        return self.training_history

    def load_best_model(self) -> bool:
        """Load the best performing model from training history."""
        if not self.training_history:
            return False

        # Find model with best R² score
        best_record = max(
            self.training_history, key=lambda x: x["metrics"].get("r2_score", -1)
        )

        model_filename = best_record["model_filename"]
        model = RewardModel(best_record["model_type"], self.model_path)

        if model.load_model(model_filename):
            self.current_model = model
            logger.info(f"Loaded best model: {model_filename}")
            return True

        return False

    def export_training_data(self, filename: str):
        """Export training data to CSV for analysis."""
        if not self.data_points:
            return

        features_list = [dp.to_features() for dp in self.data_points]
        df = pd.DataFrame(features_list)
        filepath = os.path.join(self.model_path, f"{filename}.csv")
        df.to_csv(filepath, index=False)
        logger.info(f"Training data exported to {filepath}")
