"""Command-line alias kept because the report cites ``python -m bgarch.reproduce``.

The implementation is ``bgarch_experiments.reproduce`` (experiment orchestration is not part of the library).
"""
from bgarch_experiments.reproduce import main

if __name__ == '__main__':
    main()
