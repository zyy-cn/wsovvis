from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def _install_detectron2_stub(monkeypatch):
    dataset_registry = {}
    metadata_registry = {}

    class DatasetCatalog:
        @staticmethod
        def list():
            return list(dataset_registry)

        @staticmethod
        def register(name, func):
            dataset_registry[name] = func

    class _Metadata:
        def set(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            return self

    class MetadataCatalog:
        @staticmethod
        def get(name):
            metadata_registry.setdefault(name, _Metadata())
            return metadata_registry[name]

    class BoxMode:
        XYWH_ABS = 1

    detectron2 = types.ModuleType("detectron2")
    data = types.ModuleType("detectron2.data")
    structures = types.ModuleType("detectron2.structures")
    data.DatasetCatalog = DatasetCatalog
    data.MetadataCatalog = MetadataCatalog
    structures.BoxMode = BoxMode
    monkeypatch.setitem(sys.modules, "detectron2", detectron2)
    monkeypatch.setitem(sys.modules, "detectron2.data", data)
    monkeypatch.setitem(sys.modules, "detectron2.structures", structures)
    return dataset_registry, metadata_registry


def _reload_lvvis(monkeypatch):
    dataset_registry, metadata_registry = _install_detectron2_stub(monkeypatch)
    sys.modules.pop("videocutler.ext_stageb_ovvis.data.datasets.lvvis", None)
    module = importlib.import_module("videocutler.ext_stageb_ovvis.data.datasets.lvvis")
    return module, dataset_registry, metadata_registry


def test_import_side_effect_registers_lvvis_datasets(monkeypatch):
    _, dataset_registry, _ = _reload_lvvis(monkeypatch)

    assert sorted(dataset_registry) == ["lvvis_train_base", "lvvis_val"]


def test_metadata_and_root_binding_match_reference(monkeypatch):
    module, _, metadata_registry = _reload_lvvis(monkeypatch)
    reference = json.loads(Path("package/assets/reference/lvvis_root_binding.json").read_text())

    for dataset_name in ("lvvis_train_base", "lvvis_val"):
        metadata = metadata_registry[dataset_name]
        assert metadata.evaluator_type == "lvvis"
        assert metadata.root_binding_mode == reference["binding_mode"]
        assert metadata.root_env_var == reference["env_var"]
        assert metadata.root_required_children == [item.rstrip("/") for item in reference["required_children"]]
        assert metadata.json_file.endswith("train_instances.json") or metadata.json_file.endswith("val_instances.json")
        assert metadata.image_root.endswith("train") or metadata.image_root.endswith("val")

    assert module.ROOT_FALLBACK == reference["repo_fallback"].rstrip("/")


def test_env_first_root_resolution(monkeypatch, tmp_path):
    module, _, _ = _reload_lvvis(monkeypatch)
    expected = tmp_path / "LV-VIS"
    monkeypatch.setenv("WSOVVIS_LVVIS_ROOT", str(expected))

    assert module.resolve_lvvis_root() == expected.resolve()
