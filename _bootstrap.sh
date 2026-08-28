# venv 준비. run.sh / analyze.sh 가 공통으로 쓴다.
if [ ! -d venv ]; then
  echo "[spikeviz] venv 생성 중... (처음 한 번만)"
  python3 -m venv venv
  ./venv/bin/python -m pip install -q --upgrade pip
  ./venv/bin/pip install -q -r requirements.txt
elif [ requirements.txt -nt venv/pyvenv.cfg ]; then
  echo "[spikeviz] 의존성 갱신 중..."
  ./venv/bin/pip install -q -r requirements.txt
  touch venv/pyvenv.cfg
fi
