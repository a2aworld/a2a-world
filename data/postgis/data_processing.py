#!/usr/bin/env python3
"""
Data Processing Module for AI Puzzle Pieces Data Pipeline

This module contains functions for data cleaning, validation, and quality assurance.
"""

import logging
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning operations for puzzle pieces data.
    """

    def __init__(self):
        self.geolocator = Nominatim(user_agent="terra-constellata-pipeline")

    def clean_text_field(self, text: str, field_name: str) -> Optional[str]:
        """
        Clean text fields by removing extra whitespace, normalizing, etc.

        Args:
            text: Input text
            field_name: Name of the field for context

        Returns:
            Cleaned text or None if invalid
        """
        if pd.isna(text) or text == "":
            return None

        # Convert to string and strip whitespace
        cleaned = str(text).strip()

        # Remove multiple spaces
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Field-specific cleaning
        if field_name == "name":
            # Capitalize first letter of each word
            cleaned = cleaned.title()
        elif field_name in ["entity", "sub_entity"]:
            # Normalize to lowercase for consistency
            cleaned = cleaned.lower()
        elif field_name == "description":
            # Remove excessive punctuation
            cleaned = re.sub(r"[.!?]{2,}", ".", cleaned)

        return cleaned if cleaned else None

    def clean_coordinates(
        self, lat: float, lon: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Clean and validate coordinates.

        Args:
            lat: Latitude value
            lon: Longitude value

        Returns:
            Tuple of (cleaned_lat, cleaned_lon)
        """
        try:
            # Convert to float if possible
            if pd.isna(lat) or pd.isna(lon):
                return None, None

            lat_clean = float(lat)
            lon_clean = float(lon)

            # Validate ranges
            if not (-90 <= lat_clean <= 90):
                logger.warning(f"Invalid latitude: {lat_clean}")
                return None, None

            if not (-180 <= lon_clean <= 180):
                logger.warning(f"Invalid longitude: {lon_clean}")
                return None, None

            # Round to reasonable precision (about 1 meter)
            lat_clean = round(lat_clean, 6)
            lon_clean = round(lon_clean, 6)

            return lat_clean, lon_clean

        except (ValueError, TypeError):
            return None, None

    def clean_url(self, url: str) -> Optional[str]:
        """
        Clean and validate URLs.

        Args:
            url: Input URL

        Returns:
            Cleaned URL or None if invalid
        """
        if pd.isna(url) or url == "":
            return None

        url = str(url).strip()

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Validate URL format
        try:
            parsed = urlparse(url)
            if parsed.netloc and parsed.scheme in ["http", "https"]:
                return url
        except:
            pass

        logger.warning(f"Invalid URL: {url}")
        return None

    def geocode_location(
        self, location_name: str, max_retries: int = 3
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Geocode a location name to coordinates.

        Args:
            location_name: Name of the location
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (latitude, longitude) or (None, None) if geocoding fails
        """
        if pd.isna(location_name) or location_name == "":
            return None, None

        for attempt in range(max_retries):
            try:
                location = self.geolocator.geocode(location_name, timeout=10)
                if location:
                    return location.latitude, location.longitude
                else:
                    logger.warning(f"Could not geocode: {location_name}")
                    return None, None
            except (GeocoderTimedOut, GeocoderUnavailable) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Geocoding attempt {attempt + 1} failed for {location_name}: {e}"
                    )
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"Geocoding failed after {max_retries} attempts for {location_name}: {e}"
                    )
                    return None, None

        return None, None

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean an entire DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting comprehensive data cleaning...")

        cleaned_df = df.copy()

        # Clean text fields
        text_fields = ["name", "entity", "sub_entity", "description"]
        for field in text_fields:
            if field in cleaned_df.columns:
                cleaned_df[field] = cleaned_df[field].apply(
                    lambda x: self.clean_text_field(x, field)
                )

        # Clean URLs
        if "source_url" in cleaned_df.columns:
            cleaned_df["source_url"] = cleaned_df["source_url"].apply(self.clean_url)

        # Clean coordinates
        if "latitude" in cleaned_df.columns and "longitude" in cleaned_df.columns:
            cleaned_coords = cleaned_df.apply(
                lambda row: self.clean_coordinates(row["latitude"], row["longitude"]),
                axis=1,
            )
            cleaned_df["latitude"] = cleaned_coords.apply(lambda x: x[0])
            cleaned_df["longitude"] = cleaned_coords.apply(lambda x: x[1])

        # Remove rows with missing critical data
        critical_fields = ["row_number", "name"]
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=critical_fields)
        removed_count = initial_count - len(cleaned_df)

        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with missing critical data")

        # Remove duplicates
        if "row_number" in cleaned_df.columns:
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates(subset=["row_number"])
            duplicate_count = initial_count - len(cleaned_df)

            if duplicate_count > 0:
                logger.info(f"Removed {duplicate_count} duplicate rows")

        logger.info(f"Data cleaning completed. Final row count: {len(cleaned_df)}")
        return cleaned_df


class DataValidator:
    """
    Handles data validation operations.
    """

    def __init__(self):
        self.validation_rules = {
            "row_number": {"type": "integer", "required": True, "min_value": 1},
            "name": {
                "type": "string",
                "required": True,
                "min_length": 1,
                "max_length": 255,
            },
            "entity": {"type": "string", "required": False, "max_length": 255},
            "sub_entity": {"type": "string", "required": False, "max_length": 255},
            "description": {"type": "string", "required": False, "max_length": 1000},
            "source_url": {"type": "url", "required": False},
            "latitude": {
                "type": "float",
                "required": False,
                "min_value": -90,
                "max_value": 90,
            },
            "longitude": {
                "type": "float",
                "required": False,
                "min_value": -180,
                "max_value": 180,
            },
        }

    def validate_field(self, value: Any, field_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a single field value.

        Args:
            value: Field value to validate
            field_name: Name of the field

        Returns:
            Tuple of (is_valid, error_message)
        """
        if field_name not in self.validation_rules:
            return True, None

        rules = self.validation_rules[field_name]

        # Check required fields
        if rules.get("required", False) and (pd.isna(value) or value == ""):
            return False, f"{field_name} is required"

        # Skip validation for None/NaN values on optional fields
        if pd.isna(value) or value == "":
            return True, None

        # Type validation
        field_type = rules["type"]

        if field_type == "integer":
            try:
                int_val = int(value)
                if "min_value" in rules and int_val < rules["min_value"]:
                    return False, f"{field_name} must be >= {rules['min_value']}"
            except (ValueError, TypeError):
                return False, f"{field_name} must be an integer"

        elif field_type == "float":
            try:
                float_val = float(value)
                if "min_value" in rules and float_val < rules["min_value"]:
                    return False, f"{field_name} must be >= {rules['min_value']}"
                if "max_value" in rules and float_val > rules["max_value"]:
                    return False, f"{field_name} must be <= {rules['max_value']}"
            except (ValueError, TypeError):
                return False, f"{field_name} must be a number"

        elif field_type == "string":
            str_val = str(value)
            if "min_length" in rules and len(str_val) < rules["min_length"]:
                return (
                    False,
                    f"{field_name} must be at least {rules['min_length']} characters",
                )
            if "max_length" in rules and len(str_val) > rules["max_length"]:
                return (
                    False,
                    f"{field_name} must be at most {rules['max_length']} characters",
                )

        elif field_type == "url":
            if not re.match(r"^https?://", str(value)):
                return False, f"{field_name} must be a valid URL"

        return True, None

    def validate_row(self, row: pd.Series) -> Tuple[bool, List[str]]:
        """
        Validate a single row of data.

        Args:
            row: DataFrame row to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        for field_name in row.index:
            is_valid, error_msg = self.validate_field(row[field_name], field_name)
            if not is_valid:
                errors.append(error_msg)

        return len(errors) == 0, errors

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate an entire DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        logger.info("Starting data validation...")

        validation_errors = []

        for idx, row in df.iterrows():
            is_valid, errors = self.validate_row(row)
            if not is_valid:
                validation_errors.append(
                    {
                        "row_index": idx,
                        "row_number": row.get("row_number", "N/A"),
                        "errors": errors,
                    }
                )

        is_valid = len(validation_errors) == 0
        logger.info(
            f"Data validation completed. Valid: {is_valid}, Errors: {len(validation_errors)}"
        )

        return is_valid, validation_errors

    def validate_geospatial_consistency(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Validate geospatial data consistency.

        Args:
            df: DataFrame with geospatial data

        Returns:
            List of consistency issues
        """
        issues = []

        if "latitude" in df.columns and "longitude" in df.columns:
            # Check for coordinates at (0,0) which are likely placeholders
            zero_coords = df[(df["latitude"] == 0) & (df["longitude"] == 0)]
            if not zero_coords.empty:
                issues.append(
                    {
                        "type": "zero_coordinates",
                        "count": len(zero_coords),
                        "severity": "WARNING",
                        "description": f"{len(zero_coords)} records have coordinates at (0,0)",
                    }
                )

            # Check for coordinates that might be swapped (lat/lon)
            # This is a heuristic: if lat > 90 or lon > 180, they might be swapped
            potential_swaps = df[
                ((df["latitude"] > 90) | (df["latitude"] < -90))
                | ((df["longitude"] > 180) | (df["longitude"] < -180))
            ]
            if not potential_swaps.empty:
                issues.append(
                    {
                        "type": "coordinate_range_error",
                        "count": len(potential_swaps),
                        "severity": "ERROR",
                        "description": f"{len(potential_swaps)} records have coordinates outside valid ranges",
                    }
                )

        return issues


class DataQualityChecker:
    """
    Performs comprehensive data quality checks.
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.validator = DataValidator()

    def perform_quality_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality check.

        Args:
            df: DataFrame to check

        Returns:
            Dictionary with quality check results
        """
        logger.info("Performing comprehensive data quality check...")

        results = {
            "total_records": len(df),
            "validation_errors": [],
            "quality_issues": [],
            "summary": {},
        }

        # Basic validation
        is_valid, validation_errors = self.validator.validate_dataframe(df)
        results["validation_errors"] = validation_errors

        # Geospatial consistency
        geospatial_issues = self.validator.validate_geospatial_consistency(df)
        results["quality_issues"].extend(geospatial_issues)

        # Completeness check
        completeness = {}
        for col in df.columns:
            non_null = df[col].notna().sum()
            completeness[col] = {
                "non_null_count": non_null,
                "null_count": len(df) - non_null,
                "completeness_percentage": (non_null / len(df)) * 100,
            }

        results["summary"]["completeness"] = completeness

        # Uniqueness check
        if "row_number" in df.columns:
            unique_row_numbers = df["row_number"].nunique()
            results["summary"]["uniqueness"] = {
                "unique_row_numbers": unique_row_numbers,
                "duplicate_row_numbers": len(df) - unique_row_numbers,
            }

        # Overall quality score
        quality_score = self._calculate_quality_score(results)
        results["summary"]["overall_quality_score"] = quality_score

        logger.info(f"Quality check completed. Score: {quality_score:.2f}%")
        return results

    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall quality score.

        Args:
            results: Quality check results

        Returns:
            Quality score as percentage
        """
        score = 100.0

        # Deduct for validation errors
        error_count = len(results["validation_errors"])
        if error_count > 0:
            score -= min(error_count * 5, 50)  # Max 50 points deduction

        # Deduct for quality issues
        for issue in results["quality_issues"]:
            if issue["severity"] == "ERROR":
                score -= 10
            elif issue["severity"] == "WARNING":
                score -= 5

        # Deduct for incomplete data
        completeness = results["summary"].get("completeness", {})
        for field, stats in completeness.items():
            if stats["completeness_percentage"] < 80:
                score -= (100 - stats["completeness_percentage"]) * 0.5

        # Deduct for duplicates
        uniqueness = results["summary"].get("uniqueness", {})
        duplicate_count = uniqueness.get("duplicate_row_numbers", 0)
        if duplicate_count > 0:
            score -= min(duplicate_count * 2, 20)

        return max(0, score)


def clean_and_validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to clean and validate data.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (cleaned_df, quality_report)
    """
    cleaner = DataCleaner()
    quality_checker = DataQualityChecker()

    # Clean data
    cleaned_df = cleaner.clean_dataframe(df)

    # Quality check
    quality_report = quality_checker.perform_quality_check(cleaned_df)

    return cleaned_df, quality_report


if __name__ == "__main__":
    # Test data processing
    logging.basicConfig(level=logging.INFO)

    # Example DataFrame
    test_data = {
        "row_number": [1, 2, 3, 4],
        "name": [
            "Test Location 1",
            "Test Location 2",
            "Test Location 3",
            "Test Location 4",
        ],
        "entity": ["city", "landmark", "river", "mountain"],
        "latitude": [40.7128, 34.0522, 51.5074, 0],
        "longitude": [-74.0060, -118.2437, -0.1278, 0],
        "source_url": ["https://example.com/1", "example.com/2", "invalid-url", None],
    }

    df = pd.DataFrame(test_data)

    cleaned_df, quality_report = clean_and_validate_data(df)

    print("Quality Report:")
    print(
        f"Overall Quality Score: {quality_report['summary']['overall_quality_score']:.2f}%"
    )
    print(f"Validation Errors: {len(quality_report['validation_errors'])}")
    print(f"Quality Issues: {len(quality_report['quality_issues'])}")
