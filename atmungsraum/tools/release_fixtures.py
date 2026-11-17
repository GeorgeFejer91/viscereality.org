"""Synthetic native versions share the public release policy, never live device data."""
import json
from pathlib import Path

PACKAGE_NAME = 'org.viscereality.atmungsraum.developer'
POLICY_PATH = Path(__file__).resolve().parents[1] / 'quest-release.json'


def release_policy():
    return json.loads(POLICY_PATH.read_text(encoding='utf-8'))


def apk_fixture(policy=None):
    published = (release_policy() if policy is None else policy)['packages'][PACKAGE_NAME]
    return {'v': 1, 'package_name': PACKAGE_NAME,
            'version_code': published['latest_version_code'],
            'version_name': published['version_name'], 'build_id': 'e' * 64}


def meta_update_fixture(apk=None, v=2, **patch):
    installed = (apk_fixture() if apk is None else apk)['version_code']
    value = dict(v=v, state='current', reason='', installed_code=installed,
                 available_code=installed, downloaded_bytes=0, total_bytes=None,
                 session_allowed=True)
    if v == 2:
        value.update(required_code=installed, cycle='2', revision='7', attempt='19',
                     can_check=True, busy_reason='')
    value.update(patch)
    return value
