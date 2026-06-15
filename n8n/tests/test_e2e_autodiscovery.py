# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
import os

import pytest

from datadog_checks.base import AgentCheck
from datadog_checks.dev import WaitFor, run_command

from . import common

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not common.AUTODISCOVERY, reason='Requires N8N_AUTODISCOVERY=true'),
]


def _agent_container_name() -> str:
    env = os.environ['HATCH_ENV_ACTIVE']
    return f'dd_n8n_{env}'


def _autodiscovery_ready() -> None:
    result = run_command(
        ['docker', 'exec', _agent_container_name(), 'agent', 'configcheck'],
        capture=True,
        check=True,
    )
    assert 'n8n' in result.stdout, result.stdout


@pytest.fixture
def autodiscovery_ready() -> None:
    WaitFor(_autodiscovery_ready, attempts=30, wait=2)()


def test_e2e_autodiscovery_default_port(dd_agent_check, autodiscovery_ready):
    aggregator = dd_agent_check(
        {'init_config': {}, 'instances': []},
        rate=True,
        discovery_min_instances=1,
        discovery_timeout=30,
    )
    service_checks = aggregator.service_checks('n8n.openmetrics.health')
    assert any(sc.status == AgentCheck.OK for sc in service_checks), service_checks
