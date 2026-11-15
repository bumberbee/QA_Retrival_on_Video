"""Save actual IPython executions/displays as a small downloadable notebook, even on failure."""
import sys as _capture_sys
import nbformat as _capture_nb
from IPython import get_ipython as _capture_ipython


class DemoNotebookCapture:
    def __init__(self, shell, destination, headings):
        self.shell, self.destination, self.headings = shell, Path(destination), headings
        self.cells, self.outputs = [], []
        self.stdout, self.stderr = _capture_sys.stdout, _capture_sys.stderr
        self.publisher = shell.display_pub.publish
        owner = self

        class Tee:
            def __init__(self, delegate, name):
                self.delegate, self.name = delegate, name

            def write(self, text):
                if text:
                    if owner.outputs and owner.outputs[-1].get('output_type') == 'stream' and owner.outputs[-1]['name'] == self.name:
                        owner.outputs[-1]['text'] += text
                    else:
                        owner.outputs.append(_capture_nb.v4.new_output('stream', name=self.name, text=text))
                return self.delegate.write(text)

            def __getattr__(self, name):
                return getattr(self.delegate, name)

        _capture_sys.stdout, _capture_sys.stderr = Tee(self.stdout, 'stdout'), Tee(self.stderr, 'stderr')

        def publish(data, metadata=None, **kwargs):
            owner.outputs.append(_capture_nb.v4.new_output('display_data', data=data, metadata=metadata or {}))
            return owner.publisher(data, metadata=metadata, **kwargs)

        shell.display_pub.publish = publish
        shell.events.register('pre_run_cell', self.begin)
        shell.events.register('post_run_cell', self.finish)

    def begin(self, info):
        self.outputs = []

    def finish(self, result):
        source = result.info.raw_cell
        error = result.error_in_exec or result.error_before_exec
        if error:
            self.outputs.append(_capture_nb.v4.new_output('error', ename=type(error).__name__,
                                evalue=str(error), traceback=[str(error)]))
        lines = source.splitlines()
        stage = next((tag for key, (tag, _) in self.headings.items()
                      if source.strip() == key or (key in {'VIDEO_IDS =', 'for question in QUESTIONS:'}
                         and any(line.startswith(key) for line in lines))), None)
        if stage:
            heading = next(text for tag, text in self.headings.values() if tag == stage)
            self.cells.append(_capture_nb.v4.new_markdown_cell(heading))
        code = _capture_nb.v4.new_code_cell(source, execution_count=result.execution_count, outputs=self.outputs)
        if stage:
            code.metadata['demo_stage'] = stage
        self.cells.append(code)
        notebook = _capture_nb.v4.new_notebook(cells=self.cells, metadata={
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'}})
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        _capture_nb.write(notebook, self.destination)
