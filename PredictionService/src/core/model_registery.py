from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib

from config.settings import (
    ARTIFACTS_DIR,
    DEFAULT_REGISTRY_MODE,
    ENABLE_WEIGHTED_ENSEMBLES,
    FEATURES_FILE_NAME,
    METADATA_FILE_NAME,
    METRICS_FILE_NAME,
    MODEL_FILE_NAME,
    REGISTRY_FILE,
    REQUIRED_ARTIFACT_FILES,
)
from utils.decorators import service_step
from utils.logging_utils import get_logger
from utils.time_utils import TimeBudget


logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedArtifact:
    model_name: str
    artifact_dir: Path
    predictor: Any
    features: List[str]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    loaded_at: str


@dataclass(frozen=True)
class RegistryEntry:
    target_name: str
    serving_type: str
    artifact_model_name: Optional[str]
    members: Dict[str, str]
    weights: Dict[str, float]
    raw: Dict[str, Any]


class ModelRegistry:
    def __init__(
        self,
        artifacts_root: Union[Path, str] = ARTIFACTS_DIR,
        registry_file: Union[Path, str] = REGISTRY_FILE,
    ) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.registry_file = Path(registry_file)

        self._models: Dict[str, LoadedArtifact] = {}
        self._registry: Dict[str, RegistryEntry] = {}
        self._load_status: Dict[str, str] = {}

    def clear(self) -> None:
        self._models.clear()
        self._registry.clear()
        self._load_status.clear()
        gc.collect()

    @service_step
    def load_registry(self) -> Dict[str, RegistryEntry]:
        if not self.registry_file.exists():
            raise FileNotFoundError(
                f"best_models_registry.json not found: {self.registry_file}"
            )

        registry_raw = self._read_json(self.registry_file)
        parsed_registry: Dict[str, RegistryEntry] = {}

        for target_name, info in registry_raw.items():
            if not isinstance(info, dict):
                raise ValueError(
                    f"Registry entry for target '{target_name}' must be a JSON object."
                )

            serving_type = info.get("type", DEFAULT_REGISTRY_MODE)

            if serving_type == "single_model":
                artifact_model_name = info.get("artifact_model_name")

                if not artifact_model_name:
                    raise KeyError(
                        f"Missing artifact_model_name for target '{target_name}'."
                    )

                entry = RegistryEntry(
                    target_name=target_name,
                    serving_type=serving_type,
                    artifact_model_name=artifact_model_name,
                    members={},
                    weights={},
                    raw=info,
                )

            elif serving_type == "weighted_ensemble":
                if not ENABLE_WEIGHTED_ENSEMBLES:
                    raise ValueError(
                        f"Weighted ensemble is disabled but target '{target_name}' uses weighted_ensemble."
                    )

                members = info.get("members", {})
                weights = info.get("weights", {})

                if not isinstance(members, dict) or not members:
                    raise ValueError(
                        f"Weighted ensemble target '{target_name}' has no members."
                    )

                if not isinstance(weights, dict) or not weights:
                    raise ValueError(
                        f"Weighted ensemble target '{target_name}' has no weights."
                    )

                missing_weights = set(members.keys()) - set(weights.keys())

                if missing_weights:
                    raise ValueError(
                        f"Missing ensemble weights for target '{target_name}': {sorted(missing_weights)}"
                    )

                entry = RegistryEntry(
                    target_name=target_name,
                    serving_type=serving_type,
                    artifact_model_name=None,
                    members=members,
                    weights={
                        key: float(value)
                        for key, value in weights.items()
                    },
                    raw=info,
                )

            else:
                raise ValueError(
                    f"Unsupported serving type '{serving_type}' for target '{target_name}'."
                )

            parsed_registry[target_name] = entry

        self._registry = parsed_registry

        logger.info(
            "Loaded registry | target_count=%s | registry_file=%s | targets=%s",
            len(self._registry),
            self.registry_file,
            sorted(self._registry.keys()),
        )

        return self._registry

    @service_step
    def load_all_from_registry(self) -> Dict[str, str]:
        self.clear()

        registry = self.load_registry()

        required_model_names = self._collect_required_model_names(
            registry
        )

        loaded_status: Dict[str, str] = {}

        logger.info(
            "Artifact loading started | artifact_count=%s | artifacts_root=%s | models=%s",
            len(required_model_names),
            self.artifacts_root,
            sorted(required_model_names),
        )

        for model_name in sorted(required_model_names):
            try:
                self.load_one(model_name)
                loaded_status[model_name] = "loaded"

                logger.info(
                    "Artifact loaded successfully | model_name=%s",
                    model_name,
                )

            except Exception as exc:
                loaded_status[model_name] = f"failed: {exc}"

                logger.exception(
                    "Artifact load failed | model_name=%s | artifact_dir=%s | error=%s",
                    model_name,
                    self.artifacts_root / model_name,
                    str(exc),
                )

        self._load_status = loaded_status

        logger.info(
            "Artifact loading completed | loaded=%s | failed=%s | status=%s",
            len([v for v in loaded_status.values() if v == "loaded"]),
            len([v for v in loaded_status.values() if v != "loaded"]),
            loaded_status,
        )

        logger.info(
            "Runtime registry state | registered_targets=%s | loaded_models=%s",
            self.target_names(),
            self.model_names(),
        )

        return loaded_status

    @service_step
    def load_one(self, model_name: str) -> LoadedArtifact:
        artifact_dir = self.artifacts_root / model_name

        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Artifact directory not found for model '{model_name}': {artifact_dir}"
            )

        self._validate_artifact_contract(
            artifact_dir=artifact_dir,
            model_name=model_name,
        )

        model_file = artifact_dir / MODEL_FILE_NAME
        features_file = artifact_dir / FEATURES_FILE_NAME
        metrics_file = artifact_dir / METRICS_FILE_NAME
        metadata_file = artifact_dir / METADATA_FILE_NAME

        with TimeBudget(f"Loading artifact {model_name}"):
            predictor = joblib.load(model_file)
            features = self._load_features(features_file)
            metrics = self._read_json(metrics_file)
            metadata = self._read_json(metadata_file)

        loaded = LoadedArtifact(
            model_name=model_name,
            artifact_dir=artifact_dir,
            predictor=predictor,
            features=features,
            metrics=metrics,
            metadata=metadata,
            loaded_at=datetime.now().astimezone().isoformat(),
        )

        self._models[model_name] = loaded

        logger.info(
            "Loaded artifact | model_name=%s | feature_count=%s | artifact_dir=%s",
            model_name,
            len(features),
            artifact_dir,
        )

        return loaded

    def reload_one(self, model_name: str) -> LoadedArtifact:
        self._models.pop(model_name, None)
        self._load_status.pop(model_name, None)

        gc.collect()

        loaded = self.load_one(model_name)
        self._load_status[model_name] = "loaded"

        return loaded

    def get_model(self, model_name: str) -> LoadedArtifact:
        if model_name not in self._models:
            status = self._load_status.get(model_name, "not loaded")

            raise KeyError(
                f"Model artifact is not loaded: {model_name}. Load status: {status}"
            )

        return self._models[model_name]

    def get_registry_entry(
        self,
        target_name: str,
    ) -> RegistryEntry:
        if target_name not in self._registry:
            raise KeyError(
                f"Target is not registered: {target_name}"
            )

        return self._registry[target_name]

    def get_artifact_for_target(
        self,
        target_name: str,
    ) -> LoadedArtifact:
        entry = self.get_registry_entry(target_name)

        if entry.serving_type != "single_model":
            raise ValueError(
                f"Target '{target_name}' is not a single_model target."
            )

        if entry.artifact_model_name is None:
            raise ValueError(
                f"No artifact_model_name defined for target '{target_name}'."
            )

        return self.get_model(entry.artifact_model_name)

    def get_ensemble_members_for_target(
        self,
        target_name: str,
    ) -> List[Tuple[str, float, LoadedArtifact]]:
        entry = self.get_registry_entry(target_name)

        if entry.serving_type != "weighted_ensemble":
            raise ValueError(
                f"Target '{target_name}' is not a weighted_ensemble target."
            )

        members: List[Tuple[str, float, LoadedArtifact]] = []

        for member_key, model_name in entry.members.items():
            weight = entry.weights[member_key]
            artifact = self.get_model(model_name)

            members.append(
                (
                    member_key,
                    weight,
                    artifact,
                )
            )

        return members

    def target_names(self) -> List[str]:
        return sorted(self._registry.keys())

    def model_names(self) -> List[str]:
        return sorted(self._models.keys())

    def load_status(self) -> Dict[str, str]:
        return dict(self._load_status)

    def exists(self, model_name: str) -> bool:
        return model_name in self._models

    def summary(self) -> Dict[str, Any]:
        return {
            "registry_file": str(self.registry_file),
            "artifacts_root": str(self.artifacts_root),
            "registered_targets": self.target_names(),
            "loaded_model_names": self.model_names(),
            "load_status": self.load_status(),
            "loaded_models": [
                {
                    "model_name": item.model_name,
                    "loaded_at": item.loaded_at,
                    "feature_count": len(item.features),
                    "metadata": item.metadata,
                    "metrics": item.metrics,
                    "artifact_dir": str(item.artifact_dir),
                }
                for item in self._models.values()
            ],
        }

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object in file: {path}"
            )

        return data

    @staticmethod
    def _load_features(path: Path) -> List[str]:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            features = data

        elif isinstance(data, dict):
            if "feature_names" in data:
                features = data["feature_names"]

            elif "features" in data:
                features = data["features"]

            else:
                raise KeyError(
                    f"Feature file must contain 'feature_names' or 'features': {path}"
                )

        else:
            raise ValueError(
                f"Invalid feature file format: {path}"
            )

        if not all(isinstance(feature, str) for feature in features):
            raise ValueError(
                f"All feature names must be strings: {path}"
            )

        if not features:
            raise ValueError(
                f"Feature list is empty: {path}"
            )

        return features

    def _validate_artifact_contract(
        self,
        artifact_dir: Path,
        model_name: str,
    ) -> None:
        missing_files = [
            file_name
            for file_name in REQUIRED_ARTIFACT_FILES
            if not (artifact_dir / file_name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Artifact '{model_name}' is missing required file(s): {missing_files}"
            )

    @staticmethod
    def _collect_required_model_names(
        registry: Dict[str, RegistryEntry],
    ) -> Set[str]:
        required_model_names: Set[str] = set()

        for entry in registry.values():
            if entry.serving_type == "single_model":
                if entry.artifact_model_name is None:
                    raise ValueError(
                        f"No artifact_model_name for target '{entry.target_name}'."
                    )

                required_model_names.add(entry.artifact_model_name)

            elif entry.serving_type == "weighted_ensemble":
                required_model_names.update(entry.members.values())

            else:
                raise ValueError(
                    f"Unsupported serving type '{entry.serving_type}' for target '{entry.target_name}'."
                )

        return required_model_names