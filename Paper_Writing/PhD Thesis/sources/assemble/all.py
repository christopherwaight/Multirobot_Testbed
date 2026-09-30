"""Regenerate every chapter and appendix, then renumber the bibliography."""
import runpy, os
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
for s in ['ch01', 'ch02', 'ch03', 'ch04', 'ch05', 'ch06', 'ch07', 'ch08',
          'ch09', 'ch10', 'apps']:
    runpy.run_path(f'{s}.py')
runpy.run_path('build_bib.py')
