for args in "--bias --n 8" "--bias --n 3" "--bias --n 2" "--bias --n 3 --D 32 --P 128"; do
  PYTHONIOENCODING=utf-8 python exp06_nextchar.py --parallel $args 2>&1 | tail -1
done
