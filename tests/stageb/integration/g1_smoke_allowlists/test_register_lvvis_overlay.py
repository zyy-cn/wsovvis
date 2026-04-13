from __future__ import annotations

import importlib
import sys
import types


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


def test_register_lvvis_overlay_is_import_side_effect_only(monkeypatch):
    dataset_registry, metadata_registry = _install_detectron2_stub(monkeypatch)
    sys.modules.pop("videocutler.ext_stageb_ovvis.data.datasets.lvvis", None)

    importlib.import_module("videocutler.ext_stageb_ovvis.data.datasets.lvvis")

    assert "lvvis_train_base" in dataset_registry
    assert "lvvis_val" in dataset_registry
    assert metadata_registry["lvvis_train_base"].root_binding_mode == "env_first_repo_fallback"
    assert metadata_registry["lvvis_val"].root_env_var == "WSOVVIS_LVVIS_ROOT"
