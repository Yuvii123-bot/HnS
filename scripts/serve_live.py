#!/usr/bin/env python3
"""Live preview of the model that is still training.

Bridges the run on disk to the deployed serverless code path:

* every 30 s it looks at ``artifacts/run250/latest.pt``; when the checkpoint
  changes it re-exports a NumPy bundle into ``serve_live/`` and drops the
  cached runtime, so the preview always answers with the newest weights;
* the HTTP routes are the production ones from ``api/index.py`` — same
  Devanagari-only policy, same /api/chat streaming, same /api/health;
* ``GRAMMAR_ROUTER=0`` by default: what you type is answered by the weights,
  with no table lookup standing in for the model.

    python3 scripts/serve_live.py --port 8080
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "artifacts" / "run250" / "latest.pt"
BUNDLE = ROOT / "serve_live"

PAGE = """<!doctype html>
<html lang="sa"><meta charset="utf-8">
<title>भाषामयः — live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;
      background:#0f1115;color:#e8e8ea}
 h1{font-size:1.15rem;font-weight:600;margin:0 0 .2rem}
 #meta{color:#8b8f98;font-size:.8rem;margin-bottom:1rem}
 #log{border:1px solid #24272e;border-radius:10px;padding:.8rem;min-height:16rem;
      white-space:pre-wrap;line-height:1.6;background:#14161b;font-size:1.05rem}
 .u{color:#7cc4ff}.a{color:#e8e8ea}
 #row{display:flex;gap:.5rem;margin-top:.6rem}
 input{flex:1;background:#14161b;border:1px solid #24272e;color:#e8e8ea;border-radius:8px;
       padding:.7rem;font-size:1rem}
 button{background:#2b6cb0;color:#fff;border:0;border-radius:8px;padding:.7rem 1.1rem;
        font-size:1rem;cursor:pointer}
 button:disabled{opacity:.5;cursor:default}
</style>
<h1>भाषामयः — प्रशिक्षणरतं रूपम् (live)</h1>
<div id="meta">loading…</div>
<div id="log"></div>
<div id="row"><input id="q" placeholder="देवनागर्यां लिखतु… e.g. द्वे + त्रीणि = किम्?" autofocus>
<button id="send">पृच्छ</button></div>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),
      send=document.getElementById('send'),meta=document.getElementById('meta');
let hist=[];
async function refresh(){
  try{const r=await fetch('/api/health');const j=await r.json();
    if(j.ok!==false&&j.model){
      meta.textContent=`${j.model.params.toLocaleString()} params · bundle ${j.bundle_step||'?'} · `
        +`${j.trained_tokens?j.trained_tokens.toLocaleString()+' tokens trained':'live weights'}`;
    } else meta.textContent='model not ready yet';
  }catch(e){meta.textContent='health check failed'}
}
async function ask(){
  const text=q.value.trim(); if(!text||send.disabled)return;
  q.value=''; send.disabled=true;
  log.innerHTML+=`<span class="u">▸ ${text}</span>\\n<span class="a">`;
  const logEl=log;
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,history:hist,params:{max_new_tokens:90,temperature:0.7}})});
    const rd=r.body.getReader(),dec=new TextDecoder();let buf='',ans='';
    while(true){const {value,done}=await rd.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\\n');buf=lines.pop();
      for(const ln of lines){if(!ln.trim())continue;
        let ev;try{ev=JSON.parse(ln)}catch(_){continue}
        if(ev.event==='token'&&ev.text){ans+=ev.text;logEl.innerHTML=logEl.innerHTML.slice(0,logEl.innerHTML.lastIndexOf('<span class="a">')+19)+ans}
        if(ev.event==='error'){ans='['+ev.message+']';logEl.innerHTML+=ans}
      }}
    log.innerHTML+=`</span>\\n`; hist.push({role:'user',content:text},{role:'assistant',content:ans});
  }catch(e){log.innerHTML+=`[network error: ${e}]</span>\\n`}
  send.disabled=false;q.focus();
}
send.onclick=ask;q.addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
refresh();setInterval(refresh,15000);
</script>
</html>"""


def export(ckpt: Path) -> int | None:
    """Re-export the checkpoint into the live bundle. Returns the step."""
    if not ckpt.exists():
        return None
    rc = subprocess.call([
        sys.executable, str(ROOT / "scripts" / "export_numpy.py"),
        "--checkpoint", str(ckpt),
        "--tokenizer", str(ROOT / "artifacts" / "tokenizer.json"),
        "--out", str(BUNDLE),
    ], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc != 0:
        print(f"[live] export failed rc={rc}", flush=True)
        return None
    try:
        import torch
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        step = int(blob.get("step", 0))
    except Exception:
        step = None
    meta_path = BUNDLE / "meta.json"
    if meta_path.exists() and step is not None:
        meta = json.loads(meta_path.read_text())
        meta["bundle_step"] = step
        meta["bundle_tokens_trained"] = step * 16 * 256
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[live] exported checkpoint at step {step}", flush=True)
    return step


class Watcher(threading.Thread):
    daemon = True

    def __init__(self, mod, ckpt: Path, every: int = 30):
        super().__init__()
        self.mod, self.ckpt, self.every = mod, ckpt, every
        self.mtime = 0.0

    def run(self) -> None:
        while True:
            try:
                if self.ckpt.exists():
                    mt = self.ckpt.stat().st_mtime
                    if mt != self.mtime:
                        self.mtime = mt
                        if export(self.ckpt) is not None:
                            self.mod._RUNTIME = None      # force reload in api/index
                            self.mod._LOAD_ERROR = None
            except Exception as exc:  # never let the watcher kill the server
                print(f"[live] watcher error: {exc}", flush=True)
            time.sleep(self.every)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--grammar-router", action="store_true",
                    help="enable the table fallback for grammar questions")
    args = ap.parse_args()

    BUNDLE.mkdir(exist_ok=True)
    os.environ["SERVE_DIR"] = str(BUNDLE)
    if not args.grammar_router:
        os.environ["GRAMMAR_ROUTER"] = "0"

    print(f"[live] exporting {CKPT} -> {BUNDLE}", flush=True)
    export(CKPT)

    spec = importlib.util.spec_from_file_location("api_index", ROOT / "api" / "index.py")
    api_index = importlib.util.module_from_spec(spec)
    sys.modules["api_index"] = api_index
    spec.loader.exec_module(api_index)

    class Live(api_index.handler):
        def do_GET(self):  # noqa: N802
            if (self.path or "/").split("?")[0] in ("/", "/index.html", "/live"):
                body = PAGE.encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
            else:
                super().do_GET()

    Watcher(api_index, CKPT).start()
    srv = ThreadingHTTPServer((args.host, args.port), Live)
    print(f"[live] serving on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
