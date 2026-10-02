import logging
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _isolated_log_scrubber():
    """`finagent.logsafe` is process-wide (a log-record factory plus a set of secret values) and a
    `ChatRuntime` installs it. Start and leave every test without it, so no test's log assertions
    depend on which test ran before."""
    from finagent import logsafe

    def reset() -> None:
        logsafe.uninstall_log_scrubber()
        logsafe.set_log_secrets(())

    reset()
    stock = logging.getLogRecordFactory()
    try:
        yield
    finally:
        reset()
        logging.setLogRecordFactory(stock)
