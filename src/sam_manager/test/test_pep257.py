# Copyright 2026 cyc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    # Disable rules that conflict with Google Python Style Guide:
    #   D213: multi-line summary on 2nd line (Google wants 1st line, D212)
    #   D401: imperative-mood first word (too narrow — collides with
    #         valid noun-phrase summaries like "Result of a single ...")
    #   D403: first-word capitalization (false-positives on CamelCase
    #         identifiers like 'BackendInterface' / 'MockBackend')
    #   D406/D407/D413: NumPy section formatting (Google uses
    #         'Args:' / 'Returns:' / 'Raises:' colon-style, not
    #         dashed-underline NumPy style)
    rc = main(argv=[
        '.', 'test',
        '--add-ignore=D213,D401,D403,D406,D407,D413',
    ])
    assert rc == 0, 'Found code style errors / warnings'
