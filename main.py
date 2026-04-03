#!/usr/bin/env python3
"""
localai GUI — PyQt6 chat interface for Phi-3 Mini
Layout: sidebar (model + confidence + controls) | chat history + input
Streaming: OllamaWorker(QThread) → token signals → live chat update
Confidence: mean exp(logprob) per response if Ollama exposes logprobs
"""

import datetime
import json
import math
import os
import sys

import requests
from PyQt6.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPalette, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPushButton, QTextBrowser,
    QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUST_URL   = os.getenv("RUST_SIDECAR_URL", "http://localhost:8080")
OLLAMA_URL = os.getenv("OLLAMA_URL",        "http://localhost:11434")
MODEL      = "phi3:mini"
CONF_HIGH  = 0.80   # green badge threshold
CONF_MED   = 0.60   # orange badge threshold (below → red)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

def _dark_palette() -> QPalette:
    p = QPalette()
    c = p.setColor
    R = QPalette.ColorRole
    c(R.Window,          QColor(30,  30,  30))
    c(R.WindowText,      QColor(220, 220, 220))
    c(R.Base,            QColor(24,  24,  24))
    c(R.AlternateBase,   QColor(40,  40,  40))
    c(R.Text,            QColor(220, 220, 220))
    c(R.Button,          QColor(50,  50,  50))
    c(R.ButtonText,      QColor(220, 220, 220))
    c(R.Highlight,       QColor(13,  71,  161))
    c(R.HighlightedText, QColor(255, 255, 255))
    c(R.Link,            QColor(79,  195, 247))
    return p


# ---------------------------------------------------------------------------
# Worker: runs in its own thread, emits tokens as they arrive
# ---------------------------------------------------------------------------

class OllamaWorker(QThread):
    token      = pyqtSignal(str)    # one streaming chunk
    confidence = pyqtSignal(float)  # mean probability 0–1 (if logprobs available)
    error      = pyqtSignal(str)
    done       = pyqtSignal()

    def __init__(self, history: list[dict]):
        super().__init__()
        self.history = history

    # Build a simple turn-by-turn prompt string from history
    @staticmethod
    def _build_prompt(history: list[dict]) -> str:
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def run(self):
        prompt = self._build_prompt(self.history)
        logprobs: list[float] = []

        # ── Try Rust sidecar (fast path, cached) ──────────────────
        try:
            resp = requests.post(
                f"{RUST_URL}/generate",
                json={"model": MODEL, "prompt": prompt, "stream": True},
                stream=True, timeout=5,
            )
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    self.token.emit(chunk.decode(errors="replace"))
            self.done.emit()
            return
        except Exception:
            pass  # sidecar not running — fall through to Ollama directly

        # ── Fall back: Ollama streaming API ───────────────────────
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"logprobs": True},
                },
                stream=True, timeout=120,
            )
            resp.raise_for_status()
        except Exception as e:
            self.error.emit(str(e))
            self.done.emit()
            return

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            tok = obj.get("response", "")
            if tok:
                self.token.emit(tok)

            # Collect per-token logprob if model exposes it
            lp = obj.get("logprobs")
            if isinstance(lp, list):
                logprobs.extend(math.exp(v) for v in lp if v is not None)
            elif isinstance(lp, (int, float)):
                logprobs.append(math.exp(lp))

        if logprobs:
            self.confidence.emit(sum(logprobs) / len(logprobs))

        self.done.emit()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Phi-3 Assistant")
        self.resize(920, 640)

        self._history: list[dict] = []     # [{role, content}, ...]
        self._streaming: str      = ""     # partial assistant response
        self._worker: OllamaWorker | None = None

        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._make_sidebar())
        outer.addWidget(self._make_divider())
        outer.addWidget(self._make_chat_panel(), stretch=1)

    def _make_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(140)
        sidebar.setStyleSheet("background:#181818;")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(6)

        layout.addWidget(self._section_label("MODEL"))
        layout.addWidget(self._tag_label(MODEL, "#0d47a1", "#90caf9"))

        layout.addSpacing(12)
        layout.addWidget(self._section_label("CONFIDENCE"))
        self.conf_badge = self._tag_label("—", "#2a2a2a", "#666")
        self.conf_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.conf_badge)

        layout.addStretch()

        layout.addWidget(self._section_label("ACTIONS"))
        for text, slot, danger in [
            ("🗑  Clear",    self._clear_chat, False),
            ("💾  Save Log", self._save_log,   False),
            ("✕  Quit",     QApplication.instance().quit, True),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            btn.setStyleSheet(self._action_btn_css(danger))
            layout.addWidget(btn)

        return sidebar

    def _make_divider(self) -> QFrame:
        d = QFrame()
        d.setFrameShape(QFrame.Shape.VLine)
        d.setStyleSheet("color:#333;background:#333;max-width:1px;")
        return d

    def _make_chat_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)
        self.chat_view.setStyleSheet(
            # font-size intentionally omitted — inherits from ZoomableApp
            "background:#1a1a1a; color:#ddd; border:none;"
            "font-family:'JetBrains Mono','Consolas','Courier New',monospace;"
            "line-height:1.5;"
        )
        layout.addWidget(self.chat_view)

        row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Ask anything…  (Enter to send)")
        self.input_box.setStyleSheet(
            # font-size intentionally omitted — inherits from ZoomableApp
            "background:#2a2a2a; color:#eee; border:1px solid #444;"
            "border-radius:5px; padding:8px 10px;"
        )
        self.input_box.returnPressed.connect(self._send)
        row.addWidget(self.input_box)

        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedWidth(72)
        self.send_btn.setStyleSheet(
            # font-size intentionally omitted — inherits from ZoomableApp
            "background:#0d47a1; color:#fff; border:none;"
            "border-radius:5px; padding:8px; font-weight:bold;"
        )
        self.send_btn.clicked.connect(self._send)
        row.addWidget(self.send_btn)

        layout.addLayout(row)
        return panel

    # ── Chat rendering ────────────────────────────────────────────

    def _render(self):
        """Rebuild chat HTML from full history + current streaming token buffer."""
        parts: list[str] = ['<body style="background:#1a1a1a;margin:0;padding:4px;">']
        for msg in self._history:
            if msg["role"] == "user":
                parts.append(
                    f'<p style="color:#4fc3f7;margin:6px 0 2px 0;">'
                    f'<b>You:</b> {_esc(msg["content"])}</p>'
                )
            else:
                parts.append(
                    f'<p style="color:#a5d6a7;margin:2px 0 8px 0;">'
                    f'<b>Phi-3:</b> {_esc(msg["content"])}</p>'
                )

        if self._streaming:
            parts.append(
                f'<p style="color:#a5d6a7;margin:2px 0;">'
                f'<b>Phi-3:</b> {_esc(self._streaming)}'
                f'<span style="color:#555;">▌</span></p>'
            )

        parts.append("</body>")
        self.chat_view.setHtml("".join(parts))
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)

    # ── Slots ─────────────────────────────────────────────────────

    def _send(self):
        text = self.input_box.text().strip()
        if not text or self._worker is not None:
            return

        self.input_box.clear()
        self.send_btn.setEnabled(False)
        self._set_conf("…", "#2a2a2a", "#888")

        self._history.append({"role": "user", "content": text})
        self._streaming = ""
        self._render()

        self._worker = OllamaWorker(self._history)
        self._worker.token.connect(self._on_token)
        self._worker.confidence.connect(self._on_confidence)
        self._worker.error.connect(self._on_error)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_token(self, tok: str):
        self._streaming += tok
        self._render()

    def _on_confidence(self, score: float):
        pct = int(score * 100)
        if score >= CONF_HIGH:
            self._set_conf(f"● {pct}%", "#1b5e20", "#a5d6a7")
        elif score >= CONF_MED:
            self._set_conf(f"● {pct}%", "#e65100", "#ffe0b2")
        else:
            self._set_conf(f"● {pct}%", "#b71c1c", "#ef9a9a")

    def _on_error(self, msg: str):
        self._history.append({"role": "assistant", "content": f"[error] {msg}"})
        self._streaming = ""
        self._render()

    def _on_done(self):
        if self._streaming:
            self._history.append({"role": "assistant", "content": self._streaming})
        self._streaming = ""
        self._worker = None
        self.send_btn.setEnabled(True)
        self._render()

    def _clear_chat(self):
        self._history.clear()
        self._streaming = ""
        self._set_conf("—", "#2a2a2a", "#666")
        self.chat_view.clear()

    def _save_log(self):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Chat Log", f"chat-{stamp}.txt", "Text Files (*.txt)"
        )
        if path:
            with open(path, "w") as f:
                for msg in self._history:
                    role = "You" if msg["role"] == "user" else "Phi-3"
                    f.write(f"{role}: {msg['content']}\n\n")

    # ── Style helpers ─────────────────────────────────────────────

    def _set_conf(self, text: str, bg: str, fg: str):
        self.conf_badge.setText(text)
        self.conf_badge.setStyleSheet(
            f"background:{bg}; color:{fg}; padding:5px 6px;"
            "border-radius:4px; font-size:11px; font-weight:bold;"
        )

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#555; font-size:10px; letter-spacing:1px;")
        return lbl

    @staticmethod
    def _tag_label(text: str, bg: str, fg: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"background:{bg}; color:{fg}; padding:5px 6px;"
            "border-radius:4px; font-size:11px; font-weight:bold;"
        )
        return lbl

    @staticmethod
    def _action_btn_css(danger: bool) -> str:
        fg = "#c62828" if danger else "#aaa"
        return (
            f"background:#242424; color:{fg}; border:1px solid #3a3a3a;"
            "border-radius:4px; padding:6px 8px; text-align:left; font-size:12px;"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    """Minimal HTML-escape for chat text."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )


# ---------------------------------------------------------------------------
# Zoomable application — handles +/- font scaling globally
# ---------------------------------------------------------------------------

class ZoomableApp(QApplication):
    """QApplication subclass that intercepts +/- keys to scale the app font.

    + / =  → increase by 2pt (max 32pt)
    -      → decrease by 2pt (min 12pt)

    Widgets that inherit the app font (no hardcoded font-size in their
    stylesheet) will resize automatically via QApplication.setFont().
    """

    _MIN_SIZE = 12
    _MAX_SIZE = 32
    _DEFAULT  = 20

    def __init__(self, argv):
        super().__init__(argv)
        self._size = self._DEFAULT
        self._apply_font()

    def _apply_font(self):
        f = QFont("JetBrains Mono, Consolas, Courier New, monospace")
        f.setPointSize(self._size)
        self.setFont(f)

    def event(self, e: QEvent) -> bool:  # noqa: N802
        if e.type() == QEvent.Type.KeyPress:
            key = e.key()
            if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self._size = min(self._MAX_SIZE, self._size + 2)
                self._apply_font()
            elif key == Qt.Key.Key_Minus:
                self._size = max(self._MIN_SIZE, self._size - 2)
                self._apply_font()
        return super().event(e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = ZoomableApp(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
