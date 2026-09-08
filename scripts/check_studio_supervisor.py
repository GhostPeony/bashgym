"""Verify installed Windows service lifecycle with an isolated temporary registration.

Uses a unique per-user autostart value and a disposable state directory/port.
Always uninstalls the test service; never targets the normal BashGym service.
Run using the installed release Python with -I. No model or worker is started.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def check(*, verify_keyring: bool = False) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("This check exercises the Windows service implementation")
    import psutil

    import bashgym
    from bashgym.campaigns.worker_service import (
        WINDOWS_RUN_KEY,
        ApiServiceManager,
        BackgroundServiceLaunch,
        build_api_service_definition,
        probe_api_health,
        run_command,
    )

    if not Path(bashgym.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Run with an installed release environment and Python -I")
    started = time.perf_counter()
    if verify_keyring:
        import keyring

        service = "BashGym-Studio-Proof"
        username = uuid4().hex
        value = uuid4().hex + uuid4().hex
        assert keyring.get_password(service, username) is None
        try:
            keyring.set_password(service, username, value)
            assert keyring.get_password(service, username) == value
        finally:
            keyring.delete_password(service, username)
        assert keyring.get_password(service, username) is None
    with tempfile.TemporaryDirectory(prefix="bashgym-supervisor-proof-") as temporary:
        root = Path(temporary)
        prior_cwd = Path.cwd()
        old_environment = dict(os.environ)
        for name in list(os.environ):
            if name.startswith("BASHGYM_") or name == "PYTHONPATH":
                os.environ.pop(name)
        os.environ.update(BASHGYM_DIR=str(root / "state"), BASHGYM_DISABLE_KEYRING="1")
        os.chdir(root)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        definition = build_api_service_definition(
            home=root / "service-home",
            executable=Path(sys.executable),
            data_directory=root / "state",
            port=port,
        )
        value_name = "BashGym-Studio-Proof-" + uuid4().hex

        def isolated_registration(argv):
            value = list(argv)
            if value[0].casefold() == "reg.exe":
                value[value.index("/V") + 1] = value_name
            return tuple(value)

        definition = dataclasses.replace(
            definition,
            install_argvs=tuple(isolated_registration(v) for v in definition.install_argvs),
            uninstall_argvs=tuple(isolated_registration(v) for v in definition.uninstall_argvs),
        )
        assert run_command(("reg.exe", "QUERY", WINDOWS_RUN_KEY, "/V", value_name)).returncode != 0
        manager = ApiServiceManager()
        launch = BackgroundServiceLaunch.model_validate_json(definition.definition_payload)

        def receipt():
            return json.loads(launch.receipt_path.read_text(encoding="utf-8"))

        def ready(*, previous_child=None):
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                health = probe_api_health(expected_state_root=root / "state", port=port)
                if (
                    health.get("healthy")
                    and health.get("state_root_match")
                    and health.get("studio_compatible")
                ):
                    current = receipt()
                    if current.get("child_pid") and current["child_pid"] != previous_child:
                        return current
                time.sleep(0.2)
            raise RuntimeError("Test service did not become ready")

        def stopped():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with socket.socket() as probe:
                    probe.settimeout(0.2)
                    if probe.connect_ex(("127.0.0.1", port)) != 0:
                        return
                time.sleep(0.2)
            raise RuntimeError("Stopped test service still accepts connections")

        installed = False
        try:
            installed = True  # Cleanup also applies to a partially successful install.
            manager.install(definition)
            first = ready()
            assert manager.status(definition)["supervisor_state"] == "available"
            manager.start(definition)
            assert receipt()["start_token"] == first["start_token"]
            child = psutil.Process(first["child_pid"])
            assert abs(child.create_time() - first["child_create_time"]) < 0.01
            owned = [*child.children(recursive=True), child]
            for process in owned:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            _, remaining = psutil.wait_procs(owned, timeout=10)
            assert not remaining
            restarted = ready(previous_child=first["child_pid"])
            assert restarted["start_token"] == first["start_token"]
            manager.stop(definition)
            assert manager.status(definition)["supervisor_state"] != "available"
            stopped()
            manager.start(definition)
            resumed = ready()
            assert resumed["start_token"] != first["start_token"]
        finally:
            try:
                if installed:
                    manager.uninstall(definition)
                    assert (
                        run_command(
                            ("reg.exe", "QUERY", WINDOWS_RUN_KEY, "/V", value_name)
                        ).returncode
                        != 0
                    )
                    assert not launch.receipt_path.exists()
                    stopped()
            finally:
                os.chdir(prior_cwd)
                os.environ.clear()
                os.environ.update(old_environment)
        return {
            "schema_version": "bashgym.studio_supervisor_check.v1",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "checks": [
                "isolated_user_autostart_registration",
                "start",
                "idempotent_start",
                "restart_after_child_failure",
                "stop",
                "resume",
                "uninstall_and_cleanup",
            ],
            "normal_service_modified": False,
            "training_started": False,
            "os_keyring_roundtrip_verified": verify_keyring,
            "limitations": ["Login/reboot persistence not exercised"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--keyring", action="store_true", help="Verify an isolated OS credential round trip"
    )
    args = parser.parse_args()
    rendered = json.dumps(check(verify_keyring=args.keyring), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
