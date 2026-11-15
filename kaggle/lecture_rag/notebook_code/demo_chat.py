"""Notebook-only session chat; widget callbacks never block waiting for user input."""
import copy
import html
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

DEMO_CHAT = None
DEMO_CHAT_UI = None


class DemoChatSession:
    def __init__(self, video_id, output_root, ask_fn=None, rewrite_fn=None, follow_fn=None):
        self.video_id, self.output_root = video_id, Path(output_root)
        self.ask_fn = ask_fn or globals()['ask']
        self.rewrite_fn = rewrite_fn or app_answer.rewrite_question
        self.follow_fn = follow_fn or app_search.is_follow_up
        self.turns, self.ended, self.saved_path = [], False, None
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()

    @property
    def history(self):
        return [{key: turn.get(key) for key in ('question', 'answer', 'query', 'used_evidence')}
                for turn in self.turns if not turn.get('error')][-3:]

    def send(self, question):
        if self.ended:
            raise ValueError('Phiên đã kết thúc. Chạy lại ô trò chuyện để mở phiên mới.')
        question = question.strip()
        if not question:
            raise ValueError('Hãy nhập câu hỏi.')
        started, history = time.monotonic(), copy.deepcopy(self.history)
        query, rewritten = question, None
        rewrite_history = history
        if not rewrite_history and self.turns:
            # Retain the user's topic after an answer error, without treating
            # the rejected answer as conversation history or as evidence.
            previous = self.turns[-1]
            rewrite_history = [{'question': previous['question'], 'query': previous['query'],
                                'answer': '', 'used_evidence': []}]
        follow_up = bool(rewrite_history) and self.follow_fn(question)
        if follow_up:
            try:
                rewritten = self.rewrite_fn(question, rewrite_history[-2:])
            except Exception:
                rewritten = None
            if rewritten and (rewritten.strip().casefold() == question.casefold()
                              or self.follow_fn(rewritten)):
                rewritten = None  # An unchanged/deictic rewrite still lacks a topic.
            previous_query = rewrite_history[-1].get('query') or rewrite_history[-1]['question']
            query = rewritten or f"{previous_query} {question}"
        try:
            result = self.ask_fn(question, video_id=self.video_id, history=history, query=query, show=False)
        except Exception as error:
            result = {'answer': 'Không thể trả lời lượt này. Hãy thử gửi lại.', 'used_evidence': [],
                      'error': str(error), 'attribution_text': 'Căn cứ trả lời\nDựa vào: không có căn cứ.'}
        turn = {**result, 'question': question, 'query': query, 'rewritten': rewritten,
                'follow_up': follow_up, 'video_id': self.video_id,
                'seconds': round(time.monotonic() - started, 2)}
        self.turns.append(copy.deepcopy(turn))
        return turn

    def end(self):
        if self.saved_path is not None:
            return self.saved_path
        finished = datetime.now(timezone.utc)
        path = self.output_root / f"chat_session_{finished.strftime('%Y%m%d_%H%M%S_%f')}.json"
        self.elapsed_seconds = round(time.monotonic() - self.started, 2)
        transcript = {'video_id': self.video_id, 'started_at': self.started_at,
                      'ended_at': finished.isoformat(), 'turn_count': len(self.turns),
                      'elapsed_seconds': self.elapsed_seconds,
                      'generation_seconds': round(sum(t['seconds'] for t in self.turns), 2),
                      'turns': self.turns}
        self.output_root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        self.ended, self.saved_path = True, path
        return path

    def summary(self):
        return (f'Đã kết thúc phiên: {len(self.turns)} lượt, {self.elapsed_seconds:.2f} giây. '
                f'Đã lưu: {self.saved_path}')


def print_chat_turn(turn):
    print('\nNgười dùng:', turn['question'])
    if turn['query'] != turn['question']:
        print('Truy vấn đã viết lại:', turn['query'])
    print('Trợ lý:', turn['answer'])
    print(turn.get('attribution_text', 'Căn cứ trả lời\n' + turn.get('attribution_summary', 'Dựa vào: không có căn cứ.')))
    if turn.get('error'):
        print('Lỗi:', turn['error'])
    print(f"Thời gian: {turn['seconds']:.2f} giây")


def chat(question):
    global DEMO_CHAT
    if DEMO_CHAT is None:
        DEMO_CHAT = DemoChatSession(VIDEO_IDS[0], OUTPUT_ROOT)
    if DEMO_CHAT_UI is not None:
        return DEMO_CHAT_UI.send(question)
    turn = DEMO_CHAT.send(question)
    print_chat_turn(turn)
    return turn


def start_chat(video_id=None):
    """Open a plain-function session, including after the scripted fallback has ended."""
    global DEMO_CHAT, DEMO_CHAT_UI
    if DEMO_CHAT_UI is not None and DEMO_CHAT_UI.busy:
        raise ValueError('Đang trả lời… hãy chờ trước khi mở phiên mới.')
    video_id = video_id or VIDEO_IDS[0]
    if video_id not in VIDEO_IDS:
        raise ValueError('Hãy chọn video_id trong bảng video đã xác định.')
    if DEMO_CHAT is not None and not DEMO_CHAT.ended:
        end_chat()
    if DEMO_CHAT_UI is not None:
        DEMO_CHAT_UI.executor.shutdown(wait=False)
    DEMO_CHAT_UI = None
    DEMO_CHAT = DemoChatSession(video_id, OUTPUT_ROOT)
    return DEMO_CHAT


def end_chat():
    if DEMO_CHAT_UI is not None and DEMO_CHAT_UI.busy:
        raise ValueError('Đang trả lời… hãy chờ trước khi kết thúc phiên.')
    if DEMO_CHAT is None:
        raise ValueError('Chưa có phiên trò chuyện. Gọi chat("câu hỏi") trước.')
    path = DEMO_CHAT.end()
    print(DEMO_CHAT.summary())
    if DEMO_CHAT_UI is not None:
        DEMO_CHAT_UI.refresh()
        DEMO_CHAT_UI.executor.shutdown(wait=False)
    return path


class DemoChatWidgets:
    def __init__(self, widgets, session):
        self.widgets, self.session, self.busy = widgets, session, False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.log = widgets.HTML(layout=widgets.Layout(height='460px', overflow='auto', border='1px solid #ddd'))
        self.text = widgets.Text(placeholder='Nhập câu hỏi về bài giảng…', layout=widgets.Layout(width='100%'))
        self.send_button = widgets.Button(description='Gửi', button_style='primary')
        self.clear_button = widgets.Button(description='Xoá ngữ cảnh')
        self.stop_button = widgets.Button(description='OK – kết thúc', button_style='success')
        self.video = widgets.Dropdown(options=VIDEO_IDS, value=session.video_id, description='Video:')
        self.status = widgets.HTML()
        self.send_button.on_click(self.submit)
        self.text.on_submit(self.submit)
        self.clear_button.on_click(self.reset)
        self.stop_button.on_click(self.stop)
        self.video.observe(self.change_video, names='value')
        controls = [self.video] if len(VIDEO_IDS) > 1 else []
        self.ui = widgets.VBox(controls + [self.log, self.text,
            widgets.HBox([self.send_button, self.clear_button, self.stop_button]), self.status])
        self.refresh()

    def refresh(self):
        blocks = []
        esc = html.escape
        for turn in self.session.turns:
            query = ('<p>Truy vấn đã viết lại: ' + esc(turn['query']) + '</p>'
                     if turn['query'] != turn['question'] else '')
            attribution = turn.get('attribution_text', 'Căn cứ trả lời\n' + turn.get('attribution_summary', ''))
            # Explicit text colors: Kaggle's dark theme would otherwise paint white text on these light bubbles.
            blocks.append('<div style="padding:12px;background:#eef5ff;color:#0b2545;margin:8px;border-radius:8px">'
                          '<b>Người dùng</b><p style="color:#0b2545">' + esc(turn['question'])
                          + '</p></div><div style="padding:12px;background:#f6f6f6;color:#1f2328;margin:8px;border-radius:8px">'
                          + '<b>Trợ lý</b>' + query
                          + '<pre style="white-space:pre-wrap;color:#1f2328;background:transparent;font-size:14px">'
                          + esc(turn['answer'])
                          + '</pre><pre style="white-space:pre-wrap;color:#57606a;background:transparent;font-size:12px;'
                          'border-top:1px solid #d0d7de;padding-top:6px">' + esc(attribution) + '</pre>'
                          + ('<p>Lỗi: ' + esc(turn['error']) + '</p>' if turn.get('error') else '') + '</div>')
        if self.session.ended:
            blocks.append('<p>' + esc(self.session.summary()) + '</p>')
        self.log.value = ''.join(blocks) or '<p>Hãy gửi câu hỏi đầu tiên.</p>'
        for control in (self.text, self.send_button, self.clear_button, self.stop_button, self.video):
            control.disabled = self.busy or self.session.ended

    def send(self, question):
        if self.busy:
            raise ValueError('Đang trả lời… hãy chờ lượt hiện tại.')
        self.busy = True
        self.status.value = 'Đang trả lời…'
        self.refresh()
        try:
            return self.session.send(question)
        finally:
            self.busy = False
            self.status.value = ''
            self.refresh()

    def submit(self, _=None):
        if self.busy or self.session.ended:
            return
        question = self.text.value.strip()
        if not question:
            self.status.value = 'Hãy nhập câu hỏi.'
            return
        self.text.value = ''
        # Publish status before submitting work; generation runs off the notebook event loop.
        self.busy = True
        self.status.value = 'Đang trả lời…'
        self.refresh()
        def generate():
            try:
                self.session.send(question)
                self.status.value = ''
            except Exception as error:
                self.status.value = 'Lỗi: ' + html.escape(str(error))
            finally:
                self.busy = False
                self.refresh()
        try:
            self.pending = self.executor.submit(generate)
        except Exception as error:
            self.busy = False
            self.status.value = 'Lỗi: ' + html.escape(str(error))
            self.refresh()

    def reset(self, _=None):
        global DEMO_CHAT
        if self.busy or self.session.ended:
            return
        try:
            self.session.end()
            self.session = DEMO_CHAT = DemoChatSession(self.video.value, OUTPUT_ROOT)
            self.status.value = 'Đã mở phiên mới.'
        except Exception as error:
            self.status.value = 'Không lưu được phiên: ' + html.escape(str(error))
        self.refresh()

    def change_video(self, change):
        if change['new'] != self.session.video_id:
            self.reset()

    def stop(self, _=None):
        if self.busy:
            return
        try:
            end_chat()
            self.status.value = 'Đã kết thúc phiên.'
        except Exception as error:
            self.status.value = 'Không lưu được phiên: ' + html.escape(str(error))
        self.refresh()


def demo_chat_interactive():
    run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '').strip().lower()
    if run_type:
        return run_type == 'interactive'
    try:
        shell = get_ipython()
        # Unknown Kaggle/batch environments default to the bounded script.
        return not os.environ.get('KAGGLE_URL_BASE') and getattr(shell, 'kernel', None) is not None
    except NameError:
        return False


def demo_chat_start(mode=None):
    global DEMO_CHAT, DEMO_CHAT_UI
    if DEMO_CHAT_UI is not None and DEMO_CHAT_UI.busy:
        print('Đang trả lời… hãy chờ rồi chạy lại ô.')
        return DEMO_CHAT_UI
    mode = mode or globals().get('CHAT_MODE', 'auto')
    # Save Version always runs the script, even if the config requested widgets.
    batch = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '').strip().lower() in {'batch', 'script'}
    interactive = not batch and (mode == 'interactive' or (mode == 'auto' and demo_chat_interactive()))
    video_id = 'L25_V008' if 'L25_V008' in VIDEO_IDS else VIDEO_IDS[0]
    start_chat(video_id)
    if interactive:
        try:
            import ipywidgets as widgets
            DEMO_CHAT_UI = DemoChatWidgets(widgets, DEMO_CHAT)
            display(DEMO_CHAT_UI.ui)
            return DEMO_CHAT_UI
        except Exception as error:
            print('Không mở được ipywidgets; chạy hội thoại mẫu:', str(error))
            if DEMO_CHAT_UI is not None:
                DEMO_CHAT_UI.executor.shutdown(wait=False)
            DEMO_CHAT_UI = None
    print('Hội thoại mẫu (3 lượt; không chờ nhập). API phiên mới: start_chat(); chat("câu hỏi"); end_chat().')
    questions = (['Kết quả xác suất của Câu 1 là bao nhiêu?',
                  'Vậy câu 2 thì kết quả là bao nhiêu?', 'Giải thích rõ hơn ý đó.'] if video_id == 'L25_V008' else
                 ['Ý chính đầu tiên trong bài giảng là gì?', 'Giải thích rõ hơn ý đó.',
                  'Thầy đưa ví dụ nào cho ý đó?'])
    for question in questions:
        chat(question)
    end_chat()
    return DEMO_CHAT
