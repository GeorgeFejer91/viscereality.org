"""Policy-following defaults and explicit adverse native fixtures stay distinct."""
import unittest
from release_fixtures import PACKAGE_NAME, apk_fixture, meta_update_fixture, release_policy


class ReleaseFixtures(unittest.TestCase):
    def test_current_apk_follows_published_latest(self):
        published = release_policy()['packages'][PACKAGE_NAME]
        current = apk_fixture()
        self.assertEqual(current['version_code'], published['latest_version_code'])
        self.assertEqual(current['version_name'], published['version_name'])

    def test_future_policy_changes_defaults_without_changing_fixture_code(self):
        future = {'v': 1, 'packages': {PACKAGE_NAME: {
            'minimum_version_code': 24, 'latest_version_code': 25, 'version_name': 'future-dev'}}}
        current = apk_fixture(future)
        self.assertEqual((current['version_code'], current['version_name']), (25, 'future-dev'))
        meta = meta_update_fixture(apk=current)
        self.assertEqual([meta[k] for k in ('installed_code', 'available_code', 'required_code')], [25, 25, 25])

    def test_current_meta_matches_current_apk(self):
        current = apk_fixture()
        meta = meta_update_fixture()
        self.assertEqual([meta[k] for k in ('installed_code', 'available_code', 'required_code')], [current['version_code']] * 3)

    def test_explicit_old_malformed_and_unavailable_values_remain_explicit(self):
        meta = meta_update_fixture(installed_code=0, available_code=None, required_code=-1,
                                   state='update_required', session_allowed=False)
        self.assertEqual([meta[k] for k in ('installed_code', 'available_code', 'required_code')], [0, None, -1])
        self.assertFalse(meta['session_allowed'])
        self.assertEqual(meta['state'], 'update_required')
        legacy = meta_update_fixture(v=1, installed_code=15, available_code=16)
        self.assertEqual(legacy['v'], 1)
        self.assertNotIn('required_code', legacy)


if __name__ == '__main__':
    unittest.main()
