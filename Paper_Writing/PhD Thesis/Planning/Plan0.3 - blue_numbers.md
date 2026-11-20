# Plan 0.3: blue numbers (pending the 10^4 overnight runs)
Date: 2026-11-19. Every `\prelim{}` number in the thesis, by chapter. Chapter 7 numbers are ported from `Draft_14.tex` and update by themselves: after `overnight_runs.txt` is done, turn `\prelim` into `\rev` and set the values in Draft_14, then run `python3 sources/assemble/all.py` and three `pdflatex` passes with `TEXMFHOME=/nonexistent`. Chapter 8 holds three numbers copied by hand; update those in `sources/assemble/ch08.py`.

## ch07_second_order.tex: 83 numbers
8000, 0.40, 0.25, 1000, 1000, 60, 5.0, 4.7, 2.6, 4.4, 5.6, 3.5, 1.3, 1.5, 2.8, 4.9, 1.3, 1.4, 0.005, 2.3, 96, 100, 0.13, 0.002, 0.004, 95, 86, 32, 38, 85, 81, 75, 84, 0.01, 0.02, 97, 96, 97, 61, 3, 4, 0.11, 0.07, 41, 62, 13, 15, 15, 26, 0.10, 0.20, 71, 100, 67, 29, 71, 70, 0.25, 0.33, 51, 62, 8, 9, 50, 26, 9, 0.52, 0.53, 9, 20, 0.61, 0.60, 500, 300, 70, 29, 97, 29, 67, 9, 0.007, 9, 61

## ch08_crosscutting.tex: 3 numbers
4.4, 5.6, 60

## Copied by hand in `sources/assemble/ch08.py` (not ported)
Values 4.4, 5.6, 60: the 4.4 and 5.6 times of Sec. 8.4 (Draft 14 Sec. IV-C, measurement noise, above the origin) and the 60% of Table 8.3 (Draft 14 Sec. IV-C, neither tracker keeps the straddle above 60% at sigma_uv = 0.01).

Total in chapters: 86.
