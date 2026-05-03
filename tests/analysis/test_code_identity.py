"""Unit tests for analysis.phase4.code_identity."""
import textwrap

from analysis.phase4.code_identity import (
    code_metrics,
    family_prompt,
)


def test_code_metrics_counts_loc_and_depth():
    code = textwrap.dedent('''
        import numpy as np

        class Demo:
            def __init__(self, budget, dim):
                self.b = budget
                if budget > 0:
                    for i in range(3):
                        if i % 2 == 0:
                            print(i)
    ''').strip()
    m = code_metrics(code)
    assert m['lines_of_code'] >= 8
    assert m['max_nesting'] >= 4   # def → if → for → if


def test_code_metrics_skips_blank_and_comments():
    code = '\n'.join([
        '# top comment',
        '',
        'import numpy as np',
        '',
        'class A:',
        '    pass',
    ])
    m = code_metrics(code)
    assert m['lines_of_code'] == 3   # import, class, pass
    assert m['max_nesting'] == 1


def test_family_prompt_contains_class_name_and_code():
    code = '''
    import numpy as np

    class CMA_ES_Like:
        pass
    '''.strip()
    prompt = family_prompt(condition='neutral', algorithm_name='CMA_ES_Like',
                           code=code)
    assert 'CMA_ES_Like' in prompt
    assert 'neutral' in prompt
    assert 'CMA-ES' in prompt or 'metaheuristic family' in prompt
