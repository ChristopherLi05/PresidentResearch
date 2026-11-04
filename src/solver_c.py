"""Native exact solver bridge. Build with python -m src.solver_c --build."""
import argparse
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("play", "bomb", "pass", "complete", "decline_completion")


class State(ct.Structure):
    _fields_ = [("hands", (ct.c_int * 13) * 2)] + [(name, ct.c_int) for name in (
        "actor", "rank", "count", "stack")] + [("window", ct.c_int * 2)] + [
        (name, ct.c_int) for name in ("window_count", "clear_after", "next_actor", "pass_clears", "pass_actor", "winner")]


class Result(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ("kind", "rank", "count", "winner")] + [
        ("states", ct.c_uint64), ("hits", ct.c_uint64)]


def library_path():
    name = "solver.dll" if os.name == "nt" else "libsolver.dylib" if sys.platform == "darwin" else "libsolver.so"
    return ROOT / "build" / name


def build_library(*, output=None, compiler=None):
    compiler = compiler or os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        raise RuntimeError("GCC or Clang is required to build the native solver")
    path = Path(output).resolve() if output else library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(compiler), "-std=c11", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
               "-shared", str(ROOT / "src/solver_native.c"), "-o", str(path)]
    if os.name != "nt":
        command.insert(1, "-fPIC")
    if sys.platform == "darwin":
        command[command.index("-shared")] = "-dynamiclib"
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Native solver build failed:\n{error.stderr}") from error
    return path


@lru_cache(maxsize=1)
def library():
    path = library_path()
    if not path.exists() or (ROOT / "src/solver_native.c").stat().st_mtime_ns > path.stat().st_mtime_ns:
        return None
    try:
        lib = ct.CDLL(str(path))
        lib.president_solver_version.argtypes = []
        lib.president_solver_version.restype = ct.c_int
        if lib.president_solver_version() != 1:
            return None
        lib.president_solve.argtypes = [ct.POINTER(State), ct.POINTER(Result)]
        lib.president_solve.restype = ct.c_int
        return lib
    except (OSError, AttributeError):
        return None


def solve(root, backend):
    if backend == "python":
        return None
    lib = library()
    if lib is None:
        if backend == "c":
            raise RuntimeError("Native solver unavailable; run python -m src.solver_c --build")
        return None
    state, result = State(), Result()
    if len(root.window) > 2:
        raise ValueError("invalid two-player completion window")
    for p in range(2):
        state.hands[p][:] = root.hands[p]
    for name in ("actor", "rank", "count", "stack", "clear_after", "next_actor", "pass_clears", "pass_actor", "winner"):
        setattr(state, name, getattr(root, name))
    state.window_count = len(root.window)
    state.window[:len(root.window)] = root.window
    status = lib.president_solve(ct.byref(state), ct.byref(result))
    if status == -2:
        raise MemoryError("Native endgame solver could not allocate its memo table")
    if status != 1:
        raise ValueError("Native endgame solver rejected the state")
    return KINDS[result.kind], result.rank, result.count, result.winner, result.states, result.hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    print(build_library())
