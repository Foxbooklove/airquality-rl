"""실시간 모니터링 서버 (wandb 처럼 브라우저로 보기).

서버만 Python 이고, 보는 쪽은 브라우저에서 주소만 열면 된다.
(설치·플러그인 불필요. 같은 네트워크면 폰·다른 노트북에서도 열림)

동작 방식:
  시뮬레이션 스레드가 matplotlib figure 를 JPEG 으로 렌더 -> FrameBus 에 게시
  Flask 가 그걸 MJPEG(multipart/x-mixed-replace)으로 흘려보냄
  브라우저는 <img src="/stream"> 하나로 계속 갱신된 화면을 받는다

matplotlib 은 스레드 안전하지 않으므로, figure 를 만든 스레드에서만 렌더하고
서버 스레드는 '이미 만들어진 바이트'만 내보낸다.
"""
from __future__ import annotations

import io
import threading
import socket

from flask import Flask, Response, jsonify

_BOUNDARY = "frameboundary"


class Controller:
    """브라우저에서 시뮬레이션을 제어(일시정지/재개/속도).

    시뮬레이션 스레드는 매 프레임 gate() 를 부르고, 일시정지 상태면
    그 지점에서 멈춘다(재개하면 이어서 진행). 프레임 경계에서만 멈추므로
    그림이 중간에 깨지지 않는다.
    """

    def __init__(self):
        self._running = threading.Event()
        self._running.set()          # set = 진행 중
        self.speed = 1.0             # 1.0=기본, 크면 빠름
        self.stop = False

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()

    def toggle(self):
        self.resume() if self.paused else self.pause()

    def gate(self):
        """일시정지면 여기서 대기."""
        self._running.wait()


class FrameBus:
    """최신 프레임 1장 + 상태를 들고 있는 스레드 안전 버스."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0
        self.status: dict = {"step": 0, "phase": "대기 중", "battery": "-",
                             "info_gain": 0.0, "sim": 0, "sims_total": 0,
                             "episode": 0, "best": 0.0, "paused": False,
                             "loss": None, "training": False}

    def publish(self, jpeg: bytes):
        with self._cond:
            self._frame = jpeg
            self._seq += 1
            self._cond.notify_all()

    def set_status(self, **kw):
        self.status.update(kw)

    def frames(self, timeout: float = 30.0):
        """새 프레임이 올 때마다 yield (없으면 마지막 걸 유지)."""
        last = -1
        while True:
            with self._cond:
                if self._seq == last:
                    self._cond.wait(timeout=timeout)
                if self._frame is None:
                    continue
                frame, last = self._frame, self._seq
            yield frame


def local_ip() -> str:
    """같은 네트워크의 다른 기기에서 접속할 때 쓸 IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>무인기 대기질 모니터링 — 실시간</title>
<style>
  :root { --bg:#0B1622; --panel:#132433; --line:#1F394E;
          --blue:#4F9BE8; --orange:#E8833A; --green:#3FBF77; --mute:#8FA6B8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:#EAF2F8;
         font-family:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0; font-weight:700; letter-spacing:.2px; }
  .sub { color:var(--mute); font-size:13px; }
  .wrap { display:flex; gap:16px; padding:16px 20px; align-items:flex-start;
          flex-wrap:wrap; }
  .stage { background:var(--panel); border:1px solid var(--line);
           border-radius:10px; padding:10px; flex:1 1 620px; min-width:320px; }
  .stage img { width:100%; height:auto; max-height:calc(100vh - 120px);
               object-fit:contain; display:block; border-radius:6px;
               background:#fff; }
  .side { flex:0 0 250px; display:flex; flex-direction:column; gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:14px; }
  .card h2 { font-size:12px; margin:0 0 10px; color:var(--mute);
             font-weight:600; letter-spacing:.6px; }
  .row { display:flex; justify-content:space-between; align-items:baseline;
         padding:5px 0; font-size:14px; }
  .row b { font-size:19px; font-weight:700; }
  .lg { display:flex; align-items:center; gap:9px; padding:4px 0; font-size:13px; }
  .sw { width:22px; height:4px; border-radius:2px; flex:0 0 auto; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--green);
         display:inline-block; margin-right:6px; animation:p 1.4s infinite; }
  @keyframes p { 0%,100%{opacity:1} 50%{opacity:.25} }
  .phase { color:var(--mute); font-size:13px; }
  .ctl { margin-left:auto; display:flex; align-items:center; gap:14px;
         font-size:13px; color:var(--mute); }
  .ctl button { background:var(--panel); color:#EAF2F8; border:1px solid var(--line);
                border-radius:7px; padding:7px 14px; font-size:13px; cursor:pointer;
                font-family:inherit; }
  .ctl button:hover { border-color:var(--blue); }
  body.paused .dot { animation:none; background:var(--orange); }
  .ro { border:1px solid var(--line); border-radius:20px; padding:5px 12px;
        font-size:12px; color:var(--mute); }
</style></head><body>
<header>
  <h1>강화학습 기반 무인기 대기질 모니터링</h1>
  <span class="sub"><span class="dot" id="dot"></span><span id="phase">연결 중…</span></span>
  <span class="ctl">
    <button id="pp" onclick="toggle()">⏸ 일시정지</button>
    <span id="ro" class="ro" hidden>관람 모드</span>
  </span>
</header>
<div class="wrap">
  <div class="stage"><img id="v" src="/stream" alt="live"></div>
  <div class="side">
    <div class="card"><h2>진행 상황</h2>
      <div class="row"><span>에피소드</span><b id="ep">0</b></div>
      <div class="row"><span>스텝</span><b id="step">0</b></div>
      <div class="row"><span>배터리</span><b id="bat">-</b></div>
      <div class="row"><span>누적 정보이득</span><b id="ig">0.000</b></div>
      <div class="row"><span>수읽기</span><b id="sim">0</b></div>
      <div class="row"><span>최고 기록</span><b id="best">0.000</b></div>
      <div class="row"><span>loss</span><b id="loss">-</b></div>
    </div>
    <div class="card"><h2>범례</h2>
      <div class="lg"><span class="sw" style="background:var(--blue)"></span>확정 경로</div>
      <div class="lg"><span class="sw" style="background:var(--orange)"></span>탐색 중 (수읽기)</div>
      <div class="lg"><span class="sw" style="background:var(--green)"></span>선택된 수</div>
      <div class="lg"><span class="sw" style="background:#fff"></span>backup (값 전파)</div>
    </div>
    <div class="card"><h2>배경</h2>
      <div class="phase">밝을수록 관측 불확실성이 큰 지역.
        무인기가 방문하면 그 주변이 어두워집니다.</div>
    </div>
  </div>
</div>
<script>
const READONLY = __READONLY__;
if (READONLY) { pp.hidden = true; ro.hidden = false; }
async function tick(){
  try{
    const r = await fetch('/status', {cache:'no-store'});
    const s = await r.json();
    ep.textContent   = s.episode;
    step.textContent = s.step;
    bat.textContent  = s.battery;
    ig.textContent   = Number(s.info_gain).toFixed(3);
    best.textContent = Number(s.best).toFixed(3);
    loss.textContent = (s.loss === null || s.loss === undefined)
                       ? '-' : Number(s.loss).toFixed(3);
    sim.textContent  = s.sims_total ? `${s.sim}/${s.sims_total}` : '-';
    phase.textContent = s.paused ? '일시정지' : s.phase;
    document.body.classList.toggle('paused', !!s.paused);
    if (!READONLY) pp.textContent = s.paused ? '▶ 재개' : '⏸ 일시정지';
  }catch(e){ phase.textContent = '연결 끊김'; }
}
async function toggle(){
  if (READONLY) return;
  await fetch('/control?action=toggle'); tick();
}
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && !READONLY) { e.preventDefault(); toggle(); }
});
setInterval(tick, 400); tick();
</script></body></html>"""


def create_app(bus: FrameBus, ctl: "Controller | None" = None,
               readonly: bool = False) -> Flask:
    app = Flask(__name__)
    app.logger.disabled = True

    @app.get("/")
    def index():
        page = _PAGE.replace("__READONLY__", "true" if readonly else "false")
        return Response(page, mimetype="text/html")

    @app.get("/status")
    def status():
        st = dict(bus.status)
        if ctl is not None:
            st["paused"] = ctl.paused
        st["readonly"] = readonly
        return jsonify(st)

    @app.get("/control")
    def control():
        from flask import request
        if readonly:
            return jsonify({"ok": False, "reason": "읽기 전용 모드"}), 403
        if ctl is None:
            return jsonify({"ok": False, "reason": "controller 없음"})
        action = request.args.get("action", "")
        if action == "toggle":
            ctl.toggle()
        elif action == "pause":
            ctl.pause()
        elif action == "resume":
            ctl.resume()
        elif action == "speed":
            try:
                ctl.speed = max(0.05, float(request.args.get("value", 1.0)))
            except ValueError:
                pass
        return jsonify({"ok": True, "paused": ctl.paused, "speed": ctl.speed})

    @app.get("/stream")
    def stream():
        def gen():
            for frame in bus.frames():
                yield (b"--" + _BOUNDARY.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame)).encode() +
                       b"\r\n\r\n" + frame + b"\r\n")
        return Response(gen(), mimetype=
                        f"multipart/x-mixed-replace; boundary={_BOUNDARY}")

    return app


def serve_in_background(bus: FrameBus, ctl: "Controller | None" = None,
                        host="0.0.0.0", port=8000, readonly=False) -> str:
    """서버를 데몬 스레드로 띄우고 접속 주소를 돌려준다.
    readonly=True 면 관람객이 일시정지 등 제어를 할 수 없다(공개용)."""
    app = create_app(bus, ctl, readonly)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True,
                               debug=False, use_reloader=False),
        daemon=True)
    t.start()
    return f"http://{local_ip()}:{port}"