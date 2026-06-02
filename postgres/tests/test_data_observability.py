# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)
from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import MagicMock, patch

import psycopg
import pytest
import yaml

from datadog_checks.postgres import PostgreSql
from datadog_checks.postgres.data_observability import EVENT_TRACK_TYPE

pytestmark = pytest.mark.unit

BASE_QUERY = {
    'monitor_id': 1,
    'dbname': 'test_db',
    'query': 'SELECT count(*) FROM orders',
    'interval_seconds': 60,
    'timeout_seconds': 30,
    'type': 'freshness',
    'entity': {
        'platform': 'aws',
        'account': '123456',
        'database': 'test_db',
        'schema': 'public',
        'table': 'orders',
    },
}

MULTI_QUERIES = [
    BASE_QUERY,
    {
        'monitor_id': 2,
        'dbname': 'test_db',
        'query': 'SELECT count(*) FROM users',
        'interval_seconds': 120,
        'timeout_seconds': 30,
        'type': 'freshness',
        'entity': {
            'platform': 'aws',
            'account': '123456',
            'database': 'test_db',
            'schema': 'public',
            'table': 'users',
        },
    },
]


def _make_do_instance(pg_instance, queries=None, config_id='test-config-123'):
    instance = deepcopy(pg_instance)
    instance['data_observability'] = {
        'enabled': True,
        'run_sync': True,
        'collection_interval': 10,
        'config_id': config_id,
        'queries': queries if queries is not None else [deepcopy(BASE_QUERY)],
    }
    return instance


def _make_mock_conn(rows=None, description=None, broken=False):
    mock_conn = MagicMock()
    mock_conn.broken = broken
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.description = description or [('count',)]
    mock_cursor.fetchmany.return_value = rows if rows is not None else [(42,)]
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


def _mock_db_pool(mock_conn):
    """Create a mock db_pool whose get_connection returns a context manager wrapping mock_conn."""
    mock_pool = MagicMock()

    @contextmanager
    def get_connection(dbname=None):
        yield mock_conn

    mock_pool.get_connection = MagicMock(side_effect=get_connection)
    return mock_pool


def _create_check(pg_instance, queries=None, config_id='test-config-123'):
    instance = _make_do_instance(pg_instance, queries=queries, config_id=config_id)
    check = PostgreSql('postgres', {}, [instance])
    return check


def _setup_and_run(pg_instance, queries=None, config_id='test-config-123', mock_conn=None, mock_cursor=None):
    if mock_conn is None:
        mock_conn, mock_cursor = _make_mock_conn()

    check = _create_check(pg_instance, queries=queries, config_id=config_id)
    check.db_pool = _mock_db_pool(mock_conn)
    check.data_observability.run_job()
    return check, mock_conn, mock_cursor


def _get_do_event_calls(mock_epe):
    """Filter event_platform_event calls to only do-query-results events."""
    return [c for c in mock_epe.call_args_list if len(c[0]) >= 2 and c[0][1] == EVENT_TRACK_TYPE]


def test_no_queries_does_nothing(aggregator, pg_instance):
    check = _create_check(pg_instance, queries=[])
    mock_pool = MagicMock()
    check.db_pool = mock_pool

    check.data_observability.run_job()

    mock_pool.get_connection.assert_not_called()
    assert len(aggregator.metrics('dd.postgres.data_observability.query_executions')) == 0


def test_single_query_success(aggregator, pg_instance):
    _setup_and_run(pg_instance)

    aggregator.assert_metric('dd.postgres.data_observability.query_execution_time')
    metrics = aggregator.metrics('dd.postgres.data_observability.query_executions')
    assert len(metrics) == 1
    assert metrics[0].value == 1
    assert 'status:success' in metrics[0].tags


def test_query_failure_database_error(aggregator, pg_instance):
    """DatabaseError (e.g. syntax error) is caught per-query; execution continues."""
    mock_conn, mock_cursor = _make_mock_conn()
    mock_cursor.execute.side_effect = psycopg.errors.ProgrammingError("syntax error")

    _setup_and_run(pg_instance, mock_conn=mock_conn, mock_cursor=mock_cursor)

    metrics = aggregator.metrics('dd.postgres.data_observability.query_executions')
    assert len(metrics) == 1
    assert metrics[0].value == 1
    assert 'status:error' in metrics[0].tags


def test_connection_failure_propagates(pg_instance):
    """Connection errors propagate to _job_loop for proper crash detection."""
    check = _create_check(pg_instance)
    mock_pool = MagicMock()
    mock_pool.get_connection = MagicMock(side_effect=psycopg.OperationalError("Connection refused"))
    check.db_pool = mock_pool

    with pytest.raises(psycopg.OperationalError, match="Connection refused"):
        check.data_observability.run_job()


def test_interface_error_propagates(pg_instance):
    """InterfaceError (broken connection mid-loop) propagates instead of being swallowed per-query."""
    mock_conn, mock_cursor = _make_mock_conn()
    mock_cursor.execute.side_effect = psycopg.InterfaceError("connection closed")

    check = _create_check(pg_instance)
    check.db_pool = _mock_db_pool(mock_conn)

    with pytest.raises(psycopg.InterfaceError, match="connection closed"):
        check.data_observability.run_job()


def test_operational_error_with_broken_conn_propagates(pg_instance):
    """OperationalError on a broken connection re-raises instead of being caught per-query."""
    mock_conn, mock_cursor = _make_mock_conn(broken=True)
    mock_cursor.execute.side_effect = psycopg.OperationalError("server closed the connection")

    check = _create_check(pg_instance)
    check.db_pool = _mock_db_pool(mock_conn)

    with pytest.raises(psycopg.OperationalError, match="server closed"):
        check.data_observability.run_job()


def test_per_query_interval_tracking(aggregator, pg_instance):
    mock_conn, _ = _make_mock_conn()

    check = _create_check(pg_instance)
    check.db_pool = _mock_db_pool(mock_conn)

    # First run: query executes
    check.data_observability.run_job()
    assert len(aggregator.metrics('dd.postgres.data_observability.query_executions')) == 1

    # Immediate second run: query skipped (interval not elapsed)
    aggregator.reset()
    check.data_observability.run_job()
    assert len(aggregator.metrics('dd.postgres.data_observability.query_executions')) == 0

    # Reset _last_execution to force re-run
    aggregator.reset()
    check.data_observability._last_execution = {1: 0.0}
    check.data_observability.run_job()
    assert len(aggregator.metrics('dd.postgres.data_observability.query_executions')) == 1


def test_multi_query_execution(aggregator, pg_instance):
    _setup_and_run(pg_instance, queries=deepcopy(MULTI_QUERIES))

    time_metrics = aggregator.metrics('dd.postgres.data_observability.query_execution_time')
    assert len(time_metrics) == 2

    status_metrics = aggregator.metrics('dd.postgres.data_observability.query_executions')
    assert len(status_metrics) == 2
    assert all(m.value == 1 for m in status_metrics)
    assert all('status:success' in m.tags for m in status_metrics)


def test_event_payload_structure(aggregator, pg_instance):
    mock_conn, _ = _make_mock_conn(rows=[(42,)], description=[('count',)])

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance)
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        do_calls = _get_do_event_calls(mock_epe)
        assert len(do_calls) == 1
        raw_event = do_calls[0][0][0]
        event_type = do_calls[0][0][1]
        payload = json.loads(raw_event)

    assert event_type == EVENT_TRACK_TYPE
    assert payload['config_id'] == 'test-config-123'
    assert payload['db_type'] == 'postgres'
    assert payload['monitor_id'] == 1
    assert payload['status'] == 'success'
    assert payload['columns'] == ['count']
    assert payload['rows'] == [[42]]
    assert payload['row_count'] == 1
    assert payload['error'] is None
    assert 'duration_s' in payload
    assert 'timestamp' in payload
    assert 'db_host' in payload
    assert 'db_port' in payload
    assert 'db_name' in payload


def test_entity_schema_alias(aggregator, pg_instance):
    mock_conn, _ = _make_mock_conn()

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance)
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        do_calls = _get_do_event_calls(mock_epe)
        payload = json.loads(do_calls[0][0][0])

    assert payload['entity']['schema'] == 'public'
    assert 'schema_' not in payload['entity']


def test_query_failure_does_not_block_subsequent(aggregator, pg_instance):
    """First query raises DatabaseError, second query still runs."""
    mock_conn, mock_cursor = _make_mock_conn()
    call_count = 0

    def execute_side_effect(sql, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise psycopg.errors.ProgrammingError("table not found")

    mock_cursor.execute = MagicMock(side_effect=execute_side_effect)

    _setup_and_run(pg_instance, queries=deepcopy(MULTI_QUERIES), mock_conn=mock_conn, mock_cursor=mock_cursor)

    status_metrics = aggregator.metrics('dd.postgres.data_observability.query_executions')
    assert len(status_metrics) == 2


def _set_local_timeout_ms(mock_cursor):
    """Return the statement_timeout (ms) passed to the SET LOCAL execute, or None."""
    for call in mock_cursor.execute.call_args_list:
        sql = call.args[0]
        if sql.strip().upper().startswith('SET LOCAL STATEMENT_TIMEOUT'):
            return call.args[1][0]
    return None


def test_timeout_seconds_applied_in_transaction(pg_instance):
    """The query's timeout_seconds is applied as SET LOCAL statement_timeout inside a transaction."""
    mock_conn, mock_cursor = _make_mock_conn()

    _setup_and_run(pg_instance, mock_conn=mock_conn, mock_cursor=mock_cursor)

    mock_conn.transaction.assert_called_once()
    assert _set_local_timeout_ms(mock_cursor) == 30_000


def test_timeout_seconds_converted_to_milliseconds(pg_instance):
    """timeout_seconds from the query payload is converted to milliseconds."""
    query = deepcopy(BASE_QUERY)
    query['timeout_seconds'] = 180
    mock_conn, mock_cursor = _make_mock_conn()

    _setup_and_run(pg_instance, queries=[query], mock_conn=mock_conn, mock_cursor=mock_cursor)

    assert _set_local_timeout_ms(mock_cursor) == 180_000


def test_no_description_does_not_block_subsequent(aggregator, pg_instance):
    """First query returns None description (non-SELECT), second query still runs."""
    mock_conn, mock_cursor = _make_mock_conn()
    query_count = 0

    def execute_side_effect(sql, *args, **kwargs):
        nonlocal query_count
        if sql.strip().upper().startswith('SET LOCAL'):
            return
        query_count += 1
        mock_cursor.description = None if query_count == 1 else [('count',)]

    mock_cursor.execute = MagicMock(side_effect=execute_side_effect)

    _setup_and_run(pg_instance, queries=deepcopy(MULTI_QUERIES), mock_conn=mock_conn, mock_cursor=mock_cursor)

    status_metrics = aggregator.metrics('dd.postgres.data_observability.query_executions')
    assert len(status_metrics) == 2
    assert 'status:error' in status_metrics[0].tags
    assert 'status:success' in status_metrics[1].tags


def test_custom_sql_select_fields_in_payload(aggregator, pg_instance):
    query = deepcopy(BASE_QUERY)
    query['custom_sql_select_fields'] = {
        'metric_config_id': 42,
        'entity_id': 'ent-abc-123',
    }
    mock_conn, _ = _make_mock_conn()

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance, queries=[query])
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        do_calls = _get_do_event_calls(mock_epe)
        payload = json.loads(do_calls[0][0][0])

    custom = payload['custom_sql_select_fields']
    assert custom['metric_config_id'] == 42
    assert custom['entity_id'] == 'ent-abc-123'


def test_dd_column_names_in_payload(aggregator, pg_instance):
    dd_cols = [('dd_count_failed_queries',), ('dd_gauge_latency_ms',), ('dd_tag_schema_name',)]
    mock_conn, _ = _make_mock_conn(rows=[(500, 0.42, 'public')], description=dd_cols)

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance)
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        do_calls = _get_do_event_calls(mock_epe)
        payload = json.loads(do_calls[0][0][0])

    assert payload['columns'] == ['dd_count_failed_queries', 'dd_gauge_latency_ms', 'dd_tag_schema_name']
    assert payload['rows'] == [[500, 0.42, 'public']]


def test_tags_include_monitor_id(aggregator, pg_instance):
    _setup_and_run(pg_instance)

    time_metrics = aggregator.metrics('dd.postgres.data_observability.query_execution_time')
    assert len(time_metrics) == 1
    assert 'monitor_id:1' in time_metrics[0].tags
    assert 'config_id:test-config-123' in time_metrics[0].tags
    assert 'db_type:postgres' in time_metrics[0].tags

    exec_metrics = aggregator.metrics('dd.postgres.data_observability.query_executions')
    assert len(exec_metrics) == 1
    assert 'monitor_id:1' in exec_metrics[0].tags
    assert 'config_id:test-config-123' in exec_metrics[0].tags
    assert 'db_type:postgres' in exec_metrics[0].tags
    assert 'status:success' in exec_metrics[0].tags


def test_query_with_no_description(aggregator, pg_instance):
    """Non-SELECT queries (cursor.description is None) are caught per-query and emit an error result."""
    mock_conn, mock_cursor = _make_mock_conn()

    def execute_side_effect(sql, *args, **kwargs):
        mock_cursor.description = None

    mock_cursor.execute = MagicMock(side_effect=execute_side_effect)

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance)
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        do_calls = _get_do_event_calls(mock_epe)
        payload = json.loads(do_calls[0][0][0])

    assert payload['status'] == 'error'
    assert 'result set' in payload['error']
    metrics = aggregator.metrics('dd.postgres.data_observability.query_executions')
    assert len(metrics) == 1
    assert 'status:error' in metrics[0].tags


def test_collection_interval_none_uses_default(pg_instance):
    """collection_interval=None should not crash, uses default."""
    instance = deepcopy(pg_instance)
    instance['data_observability'] = {
        'enabled': True,
        'run_sync': True,
        'collection_interval': None,
        'queries': [],
    }
    check = PostgreSql('postgres', {}, [instance])
    assert check.data_observability._enabled


def test_failed_query_updates_last_execution(aggregator, pg_instance):
    """A failed query still updates _last_execution so it's not retried until the next interval."""
    mock_conn, mock_cursor = _make_mock_conn()
    mock_cursor.execute.side_effect = psycopg.errors.ProgrammingError("syntax error")

    check = _create_check(pg_instance)
    check.db_pool = _mock_db_pool(mock_conn)

    check.data_observability.run_job()
    assert 1 in check.data_observability._last_execution

    # Immediate re-run should skip the query (interval not elapsed)
    aggregator.reset()
    check.data_observability.run_job()
    assert len(aggregator.metrics('dd.postgres.data_observability.query_executions')) == 0


def test_error_event_payload(aggregator, pg_instance):
    """When a query fails, the event payload contains error details."""
    mock_conn, mock_cursor = _make_mock_conn()
    mock_cursor.execute.side_effect = psycopg.errors.ProgrammingError("relation does not exist")

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance)
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        do_calls = _get_do_event_calls(mock_epe)
        assert len(do_calls) == 1
        payload = json.loads(do_calls[0][0][0])

    assert payload['status'] == 'error'
    assert 'relation does not exist' in payload['error']
    assert payload['columns'] == []
    assert payload['rows'] == []
    assert payload['row_count'] == 0
    assert payload['monitor_id'] == 1
    assert 'duration_s' in payload


def test_fetchmany_called_with_max_rows(pg_instance):
    """fetchmany is called with MAX_RESULT_ROWS to cap memory usage."""
    from datadog_checks.postgres.data_observability import MAX_RESULT_ROWS

    mock_conn, mock_cursor = _make_mock_conn()

    check = _create_check(pg_instance)
    check.db_pool = _mock_db_pool(mock_conn)
    check.data_observability.run_job()

    mock_cursor.fetchmany.assert_called_once_with(MAX_RESULT_ROWS)


# --- Per-query dbname tests ---


def test_per_query_dbname_used_for_connection(aggregator, pg_instance):
    """db_pool.get_connection is called with the query's dbname."""
    query = deepcopy(BASE_QUERY)
    query['dbname'] = 'other_db'
    mock_conn, _ = _make_mock_conn()

    check = _create_check(pg_instance, queries=[query])
    check.db_pool = _mock_db_pool(mock_conn)
    check.data_observability.run_job()

    check.db_pool.get_connection.assert_called_once_with('other_db')


def test_per_query_dbname_in_event_payload(aggregator, pg_instance):
    """The event payload db_name reflects the query's dbname."""
    query = deepcopy(BASE_QUERY)
    query['dbname'] = 'analytics_db'
    mock_conn, _ = _make_mock_conn()

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance, queries=[query])
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        do_calls = _get_do_event_calls(mock_epe)
        payload = json.loads(do_calls[0][0][0])

    assert payload['db_name'] == 'analytics_db'


def test_multi_query_different_dbnames(aggregator, pg_instance):
    """Multiple queries with different dbnames each connect to the correct database."""
    queries = [
        {**deepcopy(BASE_QUERY), 'dbname': 'db_one'},
        {**deepcopy(MULTI_QUERIES[1]), 'dbname': 'db_two'},
    ]
    mock_conn, _ = _make_mock_conn()

    with patch.object(PostgreSql, 'event_platform_event') as mock_epe:
        check = _create_check(pg_instance, queries=queries)
        check.db_pool = _mock_db_pool(mock_conn)
        check.data_observability.run_job()

        calls = check.db_pool.get_connection.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 'db_one'
        assert calls[1][0][0] == 'db_two'

        do_calls = _get_do_event_calls(mock_epe)
        payload_1 = json.loads(do_calls[0][0][0])
        payload_2 = json.loads(do_calls[1][0][0])

    assert payload_1['db_name'] == 'db_one'
    assert payload_2['db_name'] == 'db_two'


# --- Agent YAML-delivery round-trip tests ---
#
# The DO queries originate from a Remote Configuration payload handled by the Datadog Agent's
# Go RC handler (comp/dataobs/queryactions/impl/handler.go). The agent injects them into the
# postgres instance config, serializes the instance to YAML, and hands that YAML *string* to
# this check; datadog_checks.base parses it with yaml.safe_load before the dict ever reaches
# PostgreSql.__init__.
#
# DO query strings are multi-line SQL that routinely mixes indented lines with a trailing
# column-0 "-- Datadog {...}" annotation. yaml.v3 used to serialize such a string as a literal
# block scalar ("|") whose later, less-indented line escaped the block and produced YAML that
# neither go-yaml nor PyYAML can parse (a "did not find expected key" / ParserError). The agent
# now forces a double-quoted scalar for the query so the round-trip is exact. The tests above
# all inject a Python dict directly and therefore never exercise this serialize→safe_load
# boundary; these tests close that gap from the consumer side.

# Indented SELECT lines followed by a column-0 "-- Datadog" comment — the canonical failing shape.
MULTILINE_QUERY = (
    "  SELECT count(*) AS dd_value\n"
    "  FROM events.clicks c\n"
    "  LEFT JOIN events.page_views pv\n"
    "    ON c.user_id = pv.user_id AND c.page_url = pv.url\n"
    "  WHERE pv.id IS NULL\n"
    '-- Datadog {"monitor_ids":[26724188]}\n'
)


def _instance_yaml_as_agent_delivers(pg_instance, query):
    """Render the DO instance to a YAML string the way the agent delivers it to the check.

    The query is emitted as a double-quoted scalar (matching the agent's yaml.Node fix); the
    rest of the instance is dumped normally. Returns the YAML text, which mirrors exactly what
    datadog_checks.base feeds to yaml.safe_load.
    """
    instance = _make_do_instance(pg_instance, queries=[{**deepcopy(BASE_QUERY), 'query': query}])
    do = instance.pop('data_observability')
    queries_block = do.pop('queries')
    q = queries_block[0]
    # Build the query entry by hand so the SQL is a double-quoted scalar, as the agent emits it.
    # json.dumps produces a valid YAML double-quoted flow scalar for these characters.
    query_entry_lines = [f"      - query: {json.dumps(query)}"]
    for key, value in q.items():
        if key == 'query':
            continue
        query_entry_lines.append(f"        {key}: {json.dumps(value)}")
    do_yaml = ["data_observability:"]
    for key, value in do.items():
        do_yaml.append(f"    {key}: {json.dumps(value)}")
    do_yaml.append("    queries:")
    do_yaml.extend(query_entry_lines)
    return yaml.safe_dump(instance, default_flow_style=False) + "\n".join(do_yaml) + "\n"


@pytest.mark.parametrize(
    'query',
    [
        pytest.param(MULTILINE_QUERY, id='indented_then_col0_comment'),
        pytest.param('SELECT 23 as dd_value;\n-- Datadog {"monitor_ids":[26386160]}\n', id='trailing_comment'),
        pytest.param(
            'SELECT COUNT(1) AS dd_a_1, COUNT(DISTINCT "customer_id") AS dd_b_2 FROM "testdb"."shop"."orders"\n'
            '-- Datadog {"monitor_ids":[26358412,26386112]}\n',
            id='embedded_quotes',
        ),
        pytest.param('SELECT 1', id='simple_single_line'),
    ],
)
def test_agent_yaml_delivery_round_trips_query(pg_instance, query):
    """The query survives the agent's YAML serialization + safe_load round-trip byte-for-byte
    and reaches cursor.execute unchanged. This is the consumer-side guard for the yaml.v3
    block-scalar bug fixed in the agent (handler.go)."""
    instance_yaml = _instance_yaml_as_agent_delivers(pg_instance, query)

    # The agent → Python boundary: base.load_config feeds this YAML string to yaml.safe_load.
    parsed = yaml.safe_load(instance_yaml)
    assert parsed['data_observability']['queries'][0]['query'] == query, "query must survive safe_load intact"

    mock_conn, mock_cursor = _make_mock_conn()
    check = PostgreSql('postgres', {}, [parsed])
    check.db_pool = _mock_db_pool(mock_conn)
    check.data_observability.run_job()

    # The exact SQL string (not the SET LOCAL statement_timeout call) must reach the driver.
    executed = [call.args[0] for call in mock_cursor.execute.call_args_list]
    assert query in executed, f"original query string must be executed verbatim; got {executed!r}"


# The YAML the agent emitted for MULTILINE_QUERY *before* the handler.go fix: a literal block
# scalar ("|4") whose trailing "-- Datadog" line is indented 12 spaces while the block content
# sits at 14. Being less-indented than the block, that line terminates the scalar and is then
# read as a sibling node at indent 12 — deeper than the mapping keys at 10 — and the ":" inside
# it makes the parser expect a key. yaml.v3 emitted this without error; both go-yaml and PyYAML
# fail to parse it. In production this never reached the check: the agent's autodiscovery digest
# re-parsed the YAML first, failed, and dropped the config before scheduling.
BROKEN_AGENT_YAML = (
    "data_observability:\n"
    "    enabled: true\n"
    "    queries:\n"
    "        - dbname: analyticsdb\n"
    "          interval_seconds: 3600\n"
    "          query: |4\n"
    "              SELECT count(*) AS dd_value\n"
    "              FROM events.clicks c\n"
    "              LEFT JOIN events.page_views pv\n"
    "                ON c.user_id = pv.user_id AND c.page_url = pv.url\n"
    "              WHERE pv.id IS NULL\n"
    '            -- Datadog {"monitor_ids":[26724188]}\n'
    "          timeout_seconds: 300\n"
    "          type: run_query\n"
)


def test_pre_fix_agent_yaml_was_unparseable():
    """Documents the cross-language bug. The agent's old literal-block output fails yaml.safe_load
    with the same parse error go-yaml hit. The agent now emits a double-quoted scalar (handler.go),
    so this shape is no longer produced; test_agent_yaml_delivery_round_trips_query covers the fix."""
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(BROKEN_AGENT_YAML)
