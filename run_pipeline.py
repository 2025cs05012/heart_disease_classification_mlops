#!/usr/bin/env python3
# CSCDZG548 - MLOps Assignment 1 - Heart Disease Classification
# Single entry-point that runs everything from data download to PDF report.
# Tested on macOS; should also work on Linux and Windows (no bash needed).
"""
run_pipeline.py
Runs all 9 tasks from the assignment in one shot:
  1. download + clean dataset
  2. train + log to MLflow
  3. tests
  4. docker build
  5. deploy to local kind cluster
  6. start Prometheus + Grafana
  7. build the PDF report

Examples:
    python3 run_pipeline.py
    python3 run_pipeline.py --quick           (skips docker/k8s/monitoring)
    python3 run_pipeline.py --only train
    python3 run_pipeline.py --open            (open all live URLs after run)
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"
# venv layout: POSIX uses .venv/bin/python, Windows uses .venv\Scripts\python.exe
_VENV_CANDIDATES = [
    ROOT / ".venv" / "Scripts" / "python.exe",
    ROOT / ".venv" / "bin" / "python",
    ROOT / ".venv" / "bin" / "python3",
]
VENV_PY = next((p for p in _VENV_CANDIDATES if p.exists()), None)
PY = str(VENV_PY) if VENV_PY else sys.executable


def find_chrome():
    # we need a Chrome/Chromium binary for the headless PDF render in step_report.
    # try whatever's on PATH first, then fall back to common install paths.
    for name in ("google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser", "chrome", "msedge"):
        p = shutil.which(name)
        if p:
            return p
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
        "/snap/bin/chromium",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


# Small ANSI color helpers. Disabled when output is not a TTY (e.g. redirected
# to a file or run inside a CI log). Keeps things readable either way.
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s

def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def cyan(s):   return _c("36", s)
def bold(s):   return _c("1",  s)
def dim(s):    return _c("2",  s)

CHECK  = green("OK ")
CROSS  = red("FAIL")
ARROW  = cyan(">")
BULLET = dim("*")


class Spinner:
    # quick spinner shown while a sub-step is running. nothing fancy - a
    # rotating |/-\ on a single line. exits cleanly via context manager.
    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, label):
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if not USE_COLOR:
            # no TTY, just print the label once and move on
            print(f"  ... {self.label}", flush=True)
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # wipe the spinner line so the next print starts clean
        sys.stdout.write("\r" + " " * (len(self.label) + 8) + "\r")
        sys.stdout.flush()

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            sys.stdout.write(f"\r  {cyan(self.FRAMES[i % len(self.FRAMES)])} {self.label}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1


def banner():
    line = "=" * 68
    print()
    print(cyan(line))
    print(bold("  Heart Disease MLOps - Pipeline Runner"))
    print(dim("  CSCDZG548  |  Tasks 1 to 9"))
    print(cyan(line))
    print()


def run(cmd, cwd=ROOT, env=None, verbose=False, timeout=None):
    # Run a sub-command and return (returncode, combined_output).
    # When verbose=True we just stream the child's output to our own stdout,
    # otherwise we capture stdout+stderr together so we can show a tail on failure.
    e = os.environ.copy()
    if env:
        e.update(env)
    if verbose:
        proc = subprocess.run(cmd, cwd=str(cwd), env=e, timeout=timeout)
        return proc.returncode, ""
    try:
        out = subprocess.run(cmd, cwd=str(cwd), env=e, timeout=timeout,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, check=False)
        return out.returncode, out.stdout or ""
    except subprocess.TimeoutExpired as ex:
        return 124, f"timeout after {ex.timeout}s"
    except FileNotFoundError as ex:
        return 127, str(ex)


def have(tool):
    # tiny wrapper - just makes the call sites read better
    return shutil.which(tool) is not None


@dataclass
class StepResult:
    ok: bool
    elapsed: float
    message: str = ""


@dataclass
class Step:
    key: str
    title: str
    fn: object  # callable(ctx) -> StepResult
    optional: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class Context:
    args: argparse.Namespace
    summary: List[tuple] = field(default_factory=list)  # (key, ok, elapsed, note)
    started: float = field(default_factory=time.time)


def header(idx, total, title):
    print(f"  {bold(f'[{idx}/{total}]')} {title}")


def emit(result, key, ctx):
    icon = CHECK if result.ok else CROSS
    print(f"      {icon} {dim(f'{result.elapsed:5.1f}s')}  {result.message}")
    ctx.summary.append((key, result.ok, result.elapsed, result.message))


def fail_dump(out):
    # show the last 12 lines of a failed sub-process so we can see the actual error
    # without dumping the whole log to the terminal
    tail = "\n".join(out.strip().splitlines()[-12:])
    if tail:
        prefix = dim("      | ")
        print(prefix + tail.replace("\n", "\n" + prefix))


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def step_preflight(ctx):
    # Task 0 (sanity check): make sure the basic tools are around.
    # Required things must be present, optional ones are just nice-to-haves -
    # if they're missing the corresponding step will be skipped later.
    required = ["python3", "pip3", "git", "curl"]
    optional = ["docker", "kind", "kubectl", "helm", "pandoc", "node", "npx"]
    t0 = time.time()
    missing_req = []
    for tool in required:
        time.sleep(0.12)
        ok = have(tool) or (tool == "pip3" and have("pip"))
        print(f"      {CHECK if ok else CROSS} {tool:<10} {dim(shutil.which(tool) or '(not found)')}")
        if not ok:
            missing_req.append(tool)
    print(dim("      -- optional tools (needed for tasks 6-9) --"))
    for tool in optional:
        time.sleep(0.10)
        ok = have(tool)
        sym = CHECK if ok else yellow("--")
        note = shutil.which(tool) or "(missing - that step will be skipped)"
        print(f"      {sym} {tool:<10} {dim(note)}")
    cur = f"{sys.version_info.major}.{sys.version_info.minor}"
    if cur not in _SUPPORTED_PY:
        alt = next((f"python{v}" for v in _SUPPORTED_PY if shutil.which(f"python{v}")), None)
        msg = (f"Python {cur} is outside the tested range (3.10-3.12); "
               + (f"will bootstrap .venv with {alt}" if alt
                  else "install a compatible Python (e.g. brew install python@3.12)"))
        print(f"      {yellow('!!')} {msg}")
    if missing_req:
        return StepResult(False, time.time() - t0,
                          red("missing required: " + ", ".join(missing_req)))
    return StepResult(True, time.time() - t0, "all required tools detected")


# requirements.txt pins numpy 1.26 / scikit-learn 1.4 - both require
# Python 3.10..3.12. CI runs on 3.11. If the user is on 3.13+, prefer a
# compatible interpreter from PATH when bootstrapping the venv.
_SUPPORTED_PY = ("3.12", "3.11", "3.10")


def _pick_bootstrap_python():
    # Prefer the current interpreter if its version is in the supported set,
    # else look for python3.12 / 3.11 / 3.10 on PATH.
    cur = f"{sys.version_info.major}.{sys.version_info.minor}"
    if cur in _SUPPORTED_PY:
        return sys.executable
    for v in _SUPPORTED_PY:
        path = shutil.which(f"python{v}")
        if path:
            return path
    return sys.executable  # fallback - install will likely fail loudly


def _ensure_venv(verbose=False):
    # PEP 668 escape hatch: Homebrew / Debian system Python ships an
    # EXTERNALLY-MANAGED marker that blocks `pip install`. If we're running
    # on such an interpreter and there is no .venv/ yet, create one and
    # switch PY to it so the rest of the pipeline uses the venv.
    global PY, VENV_PY
    if VENV_PY:
        return
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    import sysconfig
    marker = Path(sysconfig.get_paths()["stdlib"]) / "EXTERNALLY-MANAGED"
    cur = f"{sys.version_info.major}.{sys.version_info.minor}"
    needs_venv = (not in_venv and marker.exists()) or (cur not in _SUPPORTED_PY)
    if not needs_venv:
        return
    venv_dir = ROOT / ".venv"
    if not venv_dir.exists():
        boot = _pick_bootstrap_python()
        boot_v = Path(boot).name
        with Spinner(f"creating .venv (using {boot_v}; supported: 3.10-3.12)"):
            rc, out = run([boot, "-m", "venv", str(venv_dir)],
                          verbose=verbose, timeout=180)
        if rc != 0:
            fail_dump(out)
            return
    new_py = next((p for p in _VENV_CANDIDATES if p.exists()), None)
    if new_py:
        VENV_PY = new_py
        PY = str(new_py)


def step_install_deps(ctx):
    # Task 1 prerequisite - install everything from requirements.txt into
    # whichever interpreter we're running with (venv if present, system otherwise).
    req = ROOT / "requirements.txt"
    if not req.exists():
        return StepResult(False, 0.0, red("requirements.txt missing"))
    t0 = time.time()
    _ensure_venv(verbose=ctx.args.verbose)
    with Spinner(f"pip install -r requirements.txt  (using {Path(PY).name})"):
        rc, out = run([PY, "-m", "pip", "install", "--quiet",
                       "--disable-pip-version-check", "-r", str(req)],
                      verbose=ctx.args.verbose, timeout=600)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("pip install failed"))
    return StepResult(True, time.time() - t0, "Python dependencies satisfied")


def step_data(ctx):
    # Tasks 1 + 2 - fetch the UCI Heart Disease files and merge/clean them.
    t0 = time.time()
    with Spinner("downloading UCI Heart Disease dataset"):
        rc1, out1 = run([PY, "-m", "src.data.download"], verbose=ctx.args.verbose, timeout=120)
    if rc1 != 0:
        fail_dump(out1)
        return StepResult(False, time.time() - t0, red("download failed"))
    with Spinner("cleaning + merging the 4 source files"):
        rc2, out2 = run([PY, "-m", "src.data.preprocess"], verbose=ctx.args.verbose, timeout=120)
    if rc2 != 0:
        fail_dump(out2)
        return StepResult(False, time.time() - t0, red("preprocess failed"))
    csv = ROOT / "data" / "processed" / "heart_disease_clean.csv"
    return StepResult(csv.exists(), time.time() - t0,
                      f"processed dataset: {csv.relative_to(ROOT)}")


def step_train(ctx):
    # Tasks 3 + 4 - train LogReg / RF / GB with 5-fold CV and log to MLflow.
    t0 = time.time()
    with Spinner("training LogReg + RandomForest + GradientBoosting (5-fold CV)"):
        rc, out = run([PY, "-m", "src.models.train"], verbose=ctx.args.verbose, timeout=900)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("training failed"))
    metrics = ROOT / "reports" / "metrics.json"
    note = "models/heart_pipeline.joblib + mlruns/ + reports/metrics.json"
    # try to pull the best AUC out of metrics.json so we have something nice to print
    if metrics.exists():
        import json
        try:
            m = json.loads(metrics.read_text())
            best = m.get("best_model", "?")
            auc = m["models"][best]["test_metrics"]["roc_auc"]
            note = f"best={best}  test ROC-AUC={auc:.3f}"
        except Exception:
            pass  # not critical, just fall back to the generic note
    return StepResult(True, time.time() - t0, note)


def step_tests(ctx):
    # Task: unit + integration tests via pytest.
    t0 = time.time()
    with Spinner("pytest -q (unit + integration tests)"):
        rc, out = run([PY, "-m", "pytest", "-q", "--no-header"],
                      verbose=ctx.args.verbose, timeout=300)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("tests failed"))
    # grab the "N passed in X.XXs" line that pytest prints at the end
    last = next((l for l in reversed(out.splitlines()) if "passed" in l), "")
    return StepResult(True, time.time() - t0, last.strip() or "all tests passed")


def step_ci_check(ctx):
    # Task 5 - confirm the GitHub Actions workflows exist (we can't run them here).
    t0 = time.time()
    paths = [ROOT.parent / ".github" / "workflows", ROOT / ".github" / "workflows"]
    found = [p for p in paths if p.exists()]
    if not found:
        return StepResult(False, time.time() - t0, yellow("no .github/workflows/ found"))
    workflows = sorted(p.name for p in found[0].glob("*.y*ml"))
    return StepResult(True, time.time() - t0,
                      f"{len(workflows)} workflow(s): {', '.join(workflows)}")


def step_docker(ctx):
    # Task 6 - build the Flask API container.
    if not have("docker"):
        return StepResult(False, 0.0, yellow("docker not installed - skipped"))
    t0 = time.time()
    with Spinner("docker build -t heart-api:latest ."):
        rc, out = run(["docker", "build", "-f", "docker/Dockerfile",
                       "-t", "heart-api:latest", "."], verbose=ctx.args.verbose, timeout=900)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("docker build failed"))
    rc2, out2 = run(["docker", "images", "heart-api:latest", "--format", "{{.Size}}"])
    return StepResult(True, time.time() - t0,
                      f"image heart-api:latest built ({out2.strip() or 'ok'})")


def step_k8s(ctx):
    # Task 7 - bring up the kind cluster, load our image, install ingress-nginx,
    # and apply our deployment + service + ingress. This is basically a Python
    # rewrite of scripts/demo_up.sh so it also works on Windows.
    for tool in ("docker", "kind", "kubectl"):
        if not have(tool):
            return StepResult(False, 0.0, yellow(f"{tool} missing - skipped"))
    t0 = time.time()
    cluster = "heart"
    image = "heart-api:latest"
    v = ctx.args.verbose

    # 1. make sure the docker image exists locally - if not, build it
    rc, out = run(["docker", "image", "inspect", image], verbose=False)
    if rc != 0:
        with Spinner(f"docker build {image}"):
            rc, out = run(["docker", "build", "-f", "docker/Dockerfile",
                           "-t", image, "."], verbose=v, timeout=900)
        if rc != 0:
            fail_dump(out)
            return StepResult(False, time.time() - t0, red("docker build failed"))

    # 2. create the kind cluster only if it doesn't exist already
    rc, out = run(["kind", "get", "clusters"], verbose=False)
    if cluster not in out.split():
        with Spinner(f"kind create cluster '{cluster}'"):
            rc, out = run(["kind", "create", "cluster",
                           "--config", "k8s/setup/kind-cluster-config.yaml"],
                          verbose=v, timeout=300)
        if rc != 0:
            fail_dump(out)
            return StepResult(False, time.time() - t0, red("kind create failed"))
    run(["kubectl", "config", "use-context", f"kind-{cluster}"])

    # 3. push our local image into the kind nodes
    with Spinner(f"kind load docker-image {image}"):
        rc, out = run(["kind", "load", "docker-image", image, "--name", cluster],
                      verbose=v, timeout=300)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("kind load failed"))

    # 4. install ingress-nginx if it isn't there yet
    rc, _ = run(["kubectl", "get", "ns", "ingress-nginx"], verbose=False)
    if rc != 0:
        with Spinner("install ingress-nginx"):
            rc, out = run(["kubectl", "apply", "-f",
                           "https://raw.githubusercontent.com/kubernetes/ingress-nginx/"
                           "main/deploy/static/provider/kind/deploy.yaml"],
                          verbose=v, timeout=180)
        if rc != 0:
            fail_dump(out)
            return StepResult(False, time.time() - t0, red("ingress-nginx apply failed"))
    with Spinner("wait for ingress-nginx controller"):
        rc, out = run(["kubectl", "wait", "--namespace", "ingress-nginx",
                       "--for=condition=ready", "pod",
                       "--selector=app.kubernetes.io/component=controller",
                       "--timeout=180s"], verbose=v, timeout=200)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("ingress controller not ready"))

    # 5. apply our own manifests
    with Spinner("kubectl apply heart-api manifests"):
        rc, out = run(["kubectl", "apply",
                       "-f", "k8s/configmap.yaml",
                       "-f", "k8s/deployment.yaml",
                       "-f", "k8s/service.yaml",
                       "-f", "k8s/hpa.yaml",
                       "-f", "k8s/ingress.yaml"], verbose=v, timeout=120)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("manifest apply failed"))
    # 5b. force a restart so pods pick up the freshly-loaded image even when
    # the deployment manifest itself didn't change (image tag stays "latest").
    # without this, re-runs serve stale code from containers started earlier.
    with Spinner("kubectl rollout restart deploy/heart-api"):
        rc, _ = run(["kubectl", "rollout", "restart",
                     "deployment/heart-api"], verbose=v, timeout=30)
    with Spinner("rollout status deployment/heart-api"):
        rc, out = run(["kubectl", "rollout", "status",
                       "deployment/heart-api", "--timeout=180s"], verbose=v, timeout=200)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("rollout failed"))

    # 6. quick smoke test - hit /health AND / through the ingress so we catch
    # the case where the form route is missing from a stale image.
    time.sleep(3)
    try:
        from urllib.request import urlopen
        with urlopen("http://localhost/health", timeout=10) as r:
            health_ok = (r.status == 200)
        with urlopen("http://localhost/", timeout=10) as r:
            form_ok = (r.status == 200)
    except Exception:
        health_ok = False
        form_ok = False
    if not health_ok:
        return StepResult(False, time.time() - t0, red("smoke test /health failed"))
    if not form_ok:
        return StepResult(False, time.time() - t0,
                          red("smoke test GET / failed (stale image? rebuild docker step)"))
    return StepResult(True, time.time() - t0,
                      "ingress at http://localhost/  (form, health, predict, metrics)")


def _mlflow_ui_owner_of_port_5500(mlruns: Path):
    # Inspect what is bound to localhost:5500. PIDs are global so the
    # per-checkout .mlflow_ui.pid file lies as soon as a second clone is on
    # the same machine - we look at the actual listener instead.
    # Returns ("free", None) | ("ours", pid_str) | ("other", cmdline)
    if not have("lsof"):
        return "free", None
    rc, out = run(["lsof", "-nP", "-iTCP:5500", "-sTCP:LISTEN", "-Fp"],
                  verbose=False, timeout=5)
    if rc != 0 or not out.strip():
        return "free", None
    pids = sorted({int(line[1:]) for line in out.splitlines()
                   if line.startswith("p") and line[1:].isdigit()})
    candidates = list(pids)
    for pid in pids:
        rc2, ppline = run(["ps", "-o", "ppid=", "-p", str(pid)],
                          verbose=False, timeout=3)
        if rc2 == 0 and ppline.strip().isdigit():
            candidates.append(int(ppline.strip()))
    for pid in candidates:
        rc2, cmd = run(["ps", "-o", "command=", "-p", str(pid)],
                       verbose=False, timeout=3)
        cmd = cmd.strip()
        if rc2 != 0 or "mlflow" not in cmd or " ui" not in cmd:
            continue
        if str(mlruns) in cmd or mlruns.as_uri() in cmd:
            return "ours", str(pid)
        return "other", cmd
    return "other", "non-mlflow process on :5500"


def step_mlflow_ui(ctx):
    # Starts the MLflow tracking UI in the background so we can show the
    # experiments page during the demo. We use a port-based ownership check
    # (not just .mlflow_ui.pid) so a stale UI from another checkout pointing
    # at a different mlruns/ is detected and replaced.
    mlruns = ROOT / "mlruns"
    if not mlruns.exists():
        return StepResult(False, 0.0, yellow("mlruns/ missing - run 'train' first"))
    t0 = time.time()
    pid_file = ROOT / ".mlflow_ui.pid"
    state, info = _mlflow_ui_owner_of_port_5500(mlruns)
    if state == "ours":
        return StepResult(True, time.time() - t0,
                          "MLflow UI already running at http://localhost:5500")
    if state == "other" and "mlflow" in (info or "").lower():
        # stale MLflow UI from another checkout - it would serve old experiments
        # to anyone opening :5500. Kill it and relaunch against our mlruns/.
        rc, out = run(["lsof", "-ti:5500"], verbose=False, timeout=3)
        for pid in (out.split() if rc == 0 else []):
            try:
                os.kill(int(pid), 15)
            except (OSError, ValueError):
                pass
        time.sleep(2)
    elif state == "other":
        return StepResult(False, time.time() - t0,
                          red(f"port 5500 is held by another process: {info[:100]}"))
    pid_file.unlink(missing_ok=True)
    log = ROOT / ".mlflow_ui.log"
    # detach the child so it survives this script exiting (different flag on Windows)
    kwargs = {"stdout": log.open("w"), "stderr": subprocess.STDOUT,
              "stdin": subprocess.DEVNULL, "cwd": str(ROOT)}
    if IS_WINDOWS:
        kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [PY, "-m", "mlflow", "ui",
         "--backend-store-uri", str(mlruns),
         "--host", "127.0.0.1", "--port", "5500"], **kwargs)
    pid_file.write_text(str(proc.pid))
    time.sleep(2)
    if proc.poll() is not None:
        return StepResult(False, time.time() - t0,
                          red("MLflow UI exited immediately - see .mlflow_ui.log"))
    return StepResult(True, time.time() - t0,
                      f"MLflow UI at http://localhost:5500  (pid {proc.pid}, log: .mlflow_ui.log)")


def step_monitor(ctx):
    # Task 8 - bring up Prometheus + Grafana via docker compose.
    if not have("docker"):
        return StepResult(False, 0.0, yellow("docker missing - skipped"))
    t0 = time.time()
    with Spinner("docker compose up Prometheus + Grafana"):
        rc, out = run(["docker", "compose", "-f", "monitoring/docker-compose.yml",
                       "up", "-d"], verbose=ctx.args.verbose, timeout=180)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("monitoring stack failed"))
    return StepResult(True, time.time() - t0,
                      "Prometheus :9090  |  Grafana :3000 (anonymous)")


def step_report(ctx):
    # Task 9 - build the final PDF report. Pandoc -> HTML, then headless Chrome
    # turns the HTML into a PDF (we used to use weasyprint but it wouldn't render
    # the embedded mermaid SVGs cleanly, so this approach works better).
    if not have("pandoc"):
        return StepResult(False, 0.0, yellow("pandoc missing - skipped"))
    t0 = time.time()
    v = ctx.args.verbose
    reports = ROOT / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # if there's a helper that pulls mermaid blocks out of the markdown, use it
    extract = reports / "_extract_mmd.py"
    if extract.exists() and have("npx"):
        with Spinner("render Mermaid architecture diagrams"):
            rc, out = run([PY, str(extract)], verbose=v, timeout=60)
            if rc == 0:
                for src, dst, w, h in (("_arch_flow.mmd", "architecture.png", "1800", "1400"),
                                       ("_arch_seq.mmd", "architecture_sequence.png", "1400", "900")):
                    s = reports / src
                    if s.exists():
                        run(["npx", "-y", "-p", "@mermaid-js/mermaid-cli", "mmdc",
                             "-i", str(s), "-o", str(figures / dst),
                             "-t", "default", "-b", "white", "-w", w, "-H", h,
                             "--scale", "2"],
                            verbose=v, timeout=180)
                        try:
                            s.unlink()
                        except OSError:
                            pass  # leave temp file - not a big deal

    md   = reports / "REPORT.md"
    html = reports / "REPORT.html"
    docx = reports / "REPORT.docx"
    pdf  = reports / "REPORT.pdf"
    css  = reports / "_report.css"
    # include ROOT itself so ../screenshots/*.png references inside REPORT.md resolve
    rp = os.pathsep.join([str(reports), str(figures), str(ROOT), str(ROOT / "screenshots")])
    with Spinner("pandoc -> HTML"):
        rc, out = run(["pandoc", str(md), "-s",
                       "--metadata", "title=Heart Disease MLOps - Final Report",
                       f"--resource-path={rp}", "--embed-resources", "--standalone"]
                      + (["-c", str(css)] if css.exists() else [])
                      + ["-o", str(html)], verbose=v, timeout=180)
    if rc != 0:
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("pandoc HTML failed"))
    with Spinner("pandoc -> DOCX"):
        run(["pandoc", str(md), f"--resource-path={rp}",
             "-o", str(docx)], verbose=v, timeout=180)

    chrome = find_chrome()
    if not chrome:
        # we can still hand in the HTML - just tell the user what happened
        return StepResult(html.exists(), time.time() - t0,
                          yellow(f"HTML built; Chrome not found, PDF skipped -> {html.relative_to(ROOT)}"))
    with Spinner("headless Chrome -> REPORT.pdf"):
        rc, out = run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                       f"--print-to-pdf={pdf}",
                       "--print-to-pdf-no-header", "--no-pdf-header-footer",
                       "--virtual-time-budget=10000",
                       html.resolve().as_uri()], verbose=v, timeout=180)
    if rc != 0 or not pdf.exists():
        fail_dump(out)
        return StepResult(False, time.time() - t0, red("Chrome PDF render failed"))
    return StepResult(True, time.time() - t0,
                      f"reports/REPORT.pdf ({pdf.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Step registry + main
# ---------------------------------------------------------------------------

# Order matters - each step assumes the previous ones have run.
# Tags: "always" runs in every mode, "quick" runs in --quick mode.
STEPS = [
    Step("preflight", "System pre-flight check",                step_preflight,    tags=["always"]),
    Step("install",   "Install Python dependencies",            step_install_deps, tags=["always"]),
    Step("data",      "Task 1+2: Download + preprocess data",   step_data,         tags=["quick"]),
    Step("train",     "Task 3+4: Train + log to MLflow",        step_train,        tags=["quick"]),
    Step("tests",     "Run unit tests (74% coverage target)",   step_tests,        tags=["quick"]),
    Step("ci",        "Task 5: Verify CI/CD workflows",         step_ci_check,     tags=["quick"]),
    Step("docker",    "Task 6: Build Docker image",             step_docker),
    Step("k8s",       "Task 7: Deploy to Kubernetes (kind)",    step_k8s),
    Step("mlflow_ui", "Task 3: Start MLflow tracking UI",       step_mlflow_ui,    tags=["quick"]),
    Step("monitor",   "Task 8: Start Prometheus + Grafana",     step_monitor),
    Step("report",    "Task 9: Build PDF report",               step_report,       tags=["quick"]),
]


def select_steps(args):
    # Pick the steps to run based on the CLI flags.
    if args.only:
        wanted = set(args.only)
        return [s for s in STEPS if s.key in wanted]
    if args.quick:
        return [s for s in STEPS if "always" in s.tags or "quick" in s.tags]
    return STEPS


def _step_ok(ctx, key):
    # tiny helper: did a given step run AND succeed in this session?
    return any(k == key and ok for k, ok, *_ in ctx.summary)


def collect_demo_links(ctx):
    # Build the list of (live?, label, url, note) shown in the summary block.
    # A link is "live" only if the step that produces it actually succeeded.
    pdf = ROOT / "reports" / "REPORT.pdf"
    return [
        (_step_ok(ctx, "k8s"),       "Flask web form (Task 6)",       "http://localhost/",        "POST /predict, GET /health, /metrics"),
        (_step_ok(ctx, "k8s"),       "Health JSON",                   "http://localhost/health",  "liveness probe"),
        (_step_ok(ctx, "k8s"),       "Prometheus metrics",            "http://localhost/metrics", "scraped by Prometheus"),
        (_step_ok(ctx, "mlflow_ui"), "MLflow tracking UI (Task 3)",   "http://localhost:5500",    "experiments + runs + artefacts"),
        (_step_ok(ctx, "monitor"),   "Prometheus UI (Task 8)",        "http://localhost:9090",    "Status -> Targets, Graph"),
        (_step_ok(ctx, "monitor"),   "Grafana 'Heart API' dashboard", "http://localhost:3000",    "anonymous viewer enabled"),
        (pdf.exists(),               "Final report (Task 9)",         str(pdf),                   f"{pdf.stat().st_size // 1024} KB" if pdf.exists() else "not built yet"),
    ]


def print_summary(ctx):
    total = time.time() - ctx.started
    ok_count = sum(1 for _, ok, *_ in ctx.summary if ok)
    line = "=" * 68
    print()
    print(cyan(line))
    print(bold(f"  Summary: {ok_count}/{len(ctx.summary)} steps OK  ({total:.1f}s total)"))
    print(cyan(line))
    for key, ok, elapsed, note in ctx.summary:
        icon = CHECK if ok else CROSS
        print(f"  {icon} {key:<10} {dim(f'{elapsed:5.1f}s')}  {note}")
    print()
    print(bold("  Demo links  (ok = live now, -- = not started)"))
    for live, label, url, note in collect_demo_links(ctx):
        mark = green("ok") if live else yellow("--")
        url_s = cyan(url) if live else dim(url)
        print(f"  {mark} {label:<34} {url_s}")
        if note:
            print(f"      {dim(note)}")
    print()
    # gentle nudge if the user forgot to start the MLflow UI
    if not any(k == "mlflow_ui" for k, *_ in ctx.summary) and (ROOT / "mlruns").exists():
        print(dim("  (MLflow UI not started - to start it: "
                  "python3 run_pipeline.py --only mlflow_ui)"))
    print(dim("  Tear-down:  docker compose -f monitoring/docker-compose.yml down"))
    print(dim("              kind delete cluster --name heart"))
    print(dim("              kill $(cat .mlflow_ui.pid)"))
    print()


def open_demo_links(ctx):
    # Open every live demo URL in the default browser - handy during the demo.
    import webbrowser
    opened = 0
    for live, label, url, _ in collect_demo_links(ctx):
        if live and url.startswith("http"):
            print(f"  {ARROW} opening  {label}  ->  {cyan(url)}")
            webbrowser.open_new_tab(url)
            opened += 1
            time.sleep(0.4)  # small gap so the browser doesn't drop tabs
    if opened == 0:
        print(dim("  (no live URLs to open - run the pipeline first)"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Heart Disease MLOps pipeline runner")
    p.add_argument("--quick", action="store_true",
                   help="run only Tasks 1-5 + report (skip docker/k8s/monitoring)")
    p.add_argument("--only", nargs="+", metavar="STEP",
                   help="run a subset of steps: " + ", ".join(s.key for s in STEPS))
    p.add_argument("--verbose", "-v", action="store_true",
                   help="stream sub-process stdout/stderr instead of capturing")
    p.add_argument("--no-stop", action="store_true",
                   help="keep going even if a step fails")
    p.add_argument("--open", action="store_true",
                   help="after running, open every live demo URL in the default browser")
    args = p.parse_args(argv)

    ctx = Context(args=args)
    banner()

    plan = select_steps(args)
    print(f"  {BULLET} Plan: {len(plan)} step(s) - {', '.join(s.key for s in plan)}")
    print(f"  {BULLET} Python: {dim(PY)}")
    print(f"  {BULLET} Repo:   {dim(str(ROOT))}")
    print()

    overall_ok = True
    for i, step in enumerate(plan, 1):
        header(i, len(plan), step.title)
        try:
            res = step.fn(ctx)
        except KeyboardInterrupt:
            print(red("\n      interrupted by user"))
            return 130
        except Exception as ex:
            # Catch-all so one buggy step doesn't kill the whole summary.
            res = StepResult(False, 0.0, red(f"unhandled error: {ex}"))
        emit(res, step.key, ctx)
        if not res.ok:
            overall_ok = False
            if not args.no_stop:
                print(red(f"\n  Aborting at step '{step.key}' "
                          f"(use --no-stop to continue, --verbose to see logs).\n"))
                break

    print_summary(ctx)
    if args.open:
        print(bold("  Opening demo URLs in browser..."))
        open_demo_links(ctx)
        print()
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())

