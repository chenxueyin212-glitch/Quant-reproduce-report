# -*- coding: utf-8 -*-
"""Edge headless PDF printer.
Key: msedge launcher returns immediately; the headless child writes the PDF
asynchronously. Must Popen + poll for output file (subprocess.run is wrong).
ASCII temp path required (CJK source paths fail silently).
"""
import subprocess, shutil, os, time, sys

name = sys.argv[1] if len(sys.argv) > 1 else "d4"
HERE = os.path.dirname(os.path.abspath(__file__))

SRC = {
    "b1": os.path.join(HERE, "B1因子深度评估报告_重构版_20261019.html"),
    "b2": os.path.join(HERE, "B2因子深度评估报告_重构版_20261019.html"),
    "bp1": os.path.join(HERE, "bp1因子深度评估报告_重构版_20261019.html"),
    "d4": os.path.join(HERE, "d4伏击因子深度评估报告_20261020.html"),
    "bupiao": os.path.join(HERE, "补票因子深度评估报告_重构版_20261020.html"),
    "banzhuan": os.path.join(HERE, "搬砖因子深度评估报告_重构版_20261020.html"),
    "bp2": os.path.join(HERE, "bp2因子深度评估报告_重构版_20261020.html"),
    "healthcheck": os.path.join(HERE, "六信号因子体检补全报告_20261020.html"),
}[name]
TMP_HTML = os.path.join(os.environ.get("TEMP", r"C:\Users\Lenovo\AppData\Local\Temp"), name + "_rpt.html")
TMP_PDF = TMP_HTML[:-5] + ".pdf"
PROF = os.path.join(os.environ.get("TEMP", r"C:\Users\Lenovo\AppData\Local\Temp"), "edge_prof_" + name + "_" + os.urandom(3).hex())
DST = SRC[:-5] + ".pdf"

shutil.copy(SRC, TMP_HTML)
if os.path.exists(TMP_PDF):
    os.remove(TMP_PDF)

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
url = "file:///" + TMP_HTML.replace("\\", "/")
p = subprocess.Popen([edge, "--headless=new", "--disable-gpu", "--no-sandbox",
                      "--user-data-dir=" + PROF,
                      "--no-pdf-header-footer",
                      "--print-to-pdf=" + TMP_PDF,
                      url],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("launched pid =", p.pid)
for i in range(45):
    time.sleep(2)
    if os.path.exists(TMP_PDF):
        time.sleep(1)  # 等文件句柄释放
        shutil.copy(TMP_PDF, DST)
        print("PDF ok ->", DST, os.path.getsize(DST), "bytes (waited %ds)" % ((i+1)*2))
        break
else:
    print("PDF FAIL after 90s; parent poll:", p.poll())
