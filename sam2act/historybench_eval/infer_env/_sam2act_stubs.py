"""Inference-only import shim for the SAM2Act uv venv.

The SAM2Act agent shares modules with its training / RLBench-sim pipeline, so
importing the agent pulls in top-level imports of dependencies that pure
*inference* never executes: RLBench, PyRep (CoppeliaSim), pytorch3d, pyrender,
trimesh, tensorflow. Installing those (CoppeliaSim, compiled CUDA extensions, GL
stacks, the TF/tensorboard logging stack) just to satisfy import statements is
pointless for a forward-pass-only server.

This installs meta-path finders that fabricate dummy modules for those imports.
Attribute access is *chainable*: any attribute of a dummy is itself a
subclassable / callable / further-attributable dummy. That survives import-time
annotations, base classes, default args, and capability probes. None of these
dummy paths run during actual inference.

Exception — a few symbols ARE used at inference runtime by the HTTP server's
observation preprocessing, so we inject real lightweight implementations
(copied from the vendored PyRep / RLBench) into the relevant dummy modules:
  * rlbench.backend.observation.Observation  -> a plain attribute container
  * pyrep.objects.VisionSensor.pointcloud_from_depth_and_camera_params -> the
    real depth->world-frame point-cloud unprojection (pure numpy).

Finders:
  * _StubFinder (appended) handles whole third-party roots, only firing when the
    real package is absent, so genuine installs always win.
  * _ExactStubFinder (prepended) shadows exactly ``torch.utils.tensorboard``
    (torch is real, but that submodule drags in protobuf/tensorboard for a
    logger the agent imports transitively via yarr and never uses at inference).
"""
import sys
import types
import numpy as _np
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

_STUB_ROOTS = {
    "rlbench",
    "pyrep",
    "pytorch3d",
    "pyrender",
    "trimesh",
    "tensorflow",
    "transformers",
}

_SHADOW_PREFIX = "torch.utils.tensorboard"


# --------------------------------------------------------------------------- #
# Real lightweight implementations needed at inference runtime.
# --------------------------------------------------------------------------- #
class _Observation:
    """Stand-in for rlbench.backend.observation.Observation: the server's
    extract_obs only uses it as a mutable attribute container."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _create_uniform_pixel_coords_image(resolution):
    pixel_x_coords = _np.reshape(
        _np.tile(_np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1)).astype(_np.float32)
    pixel_y_coords = _np.reshape(
        _np.tile(_np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1)).astype(_np.float32)
    pixel_y_coords = _np.transpose(pixel_y_coords, (1, 0, 2))
    return _np.concatenate(
        (pixel_x_coords, pixel_y_coords, _np.ones_like(pixel_x_coords)), -1)


def _transform(coords, trans):
    h, w = coords.shape[:2]
    coords = _np.reshape(coords, (h * w, -1))
    coords = _np.transpose(coords, (1, 0))
    t = _np.matmul(trans, coords)
    t = _np.transpose(t, (1, 0))
    return _np.reshape(t, (h, w, -1))


def _pixel_to_world_coords(pixel_coords, cam_proj_mat_inv):
    h, w = pixel_coords.shape[:2]
    pixel_coords = _np.concatenate([pixel_coords, _np.ones((h, w, 1))], -1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    return _np.concatenate([world_coords, _np.ones((h, w, 1))], axis=-1)


class _VisionSensor:
    @staticmethod
    def pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics):
        """Converts depth (meters) to a (H, W, 3) point cloud in world frame."""
        upc = _create_uniform_pixel_coords_image(depth.shape)
        pc = upc * _np.expand_dims(depth, -1)
        C = _np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T
        R_inv_C = _np.matmul(R_inv, C)
        ext = _np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = _np.matmul(intrinsics, ext)
        cam_proj_mat_homo = _np.concatenate(
            [cam_proj_mat, [_np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = _np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = _np.expand_dims(
            _pixel_to_world_coords(pc, cam_proj_mat_inv), 0)
        return world_coords_homo[..., :-1][0]


# Real symbols to graft onto specific (otherwise dummy) stub modules.
_REAL_SYMBOLS = {
    "rlbench.backend.observation": {"Observation": _Observation},
    "pyrep.objects": {"VisionSensor": _VisionSensor},
}


# --------------------------------------------------------------------------- #
# Dummy machinery.
# --------------------------------------------------------------------------- #
class _AnyMeta(type):
    """Classes resolve any attribute to another dummy class; instantiation
    yields the (still chainable) dummy class itself."""

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _new(name)

    def __call__(cls, *args, **kwargs):
        return cls


def _new(name):
    return _AnyMeta(str(name), (), {})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name == "__version__":
            return "99.0.0"
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        dummy = _new(name)
        setattr(self, name, dummy)
        return dummy


class _StubLoader(Loader):
    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        real = _REAL_SYMBOLS.get(module.__name__)
        if real:
            for k, v in real.items():
                setattr(module, k, v)


def _stub_spec(fullname):
    spec = ModuleSpec(fullname, _StubLoader())
    spec.submodule_search_locations = []  # mark as package
    return spec


class _StubFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS:
            return _stub_spec(fullname)
        return None


class _ExactStubFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _SHADOW_PREFIX or fullname.startswith(_SHADOW_PREFIX + "."):
            return _stub_spec(fullname)
        return None


if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
    sys.meta_path.append(_StubFinder())
if not any(isinstance(f, _ExactStubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _ExactStubFinder())
