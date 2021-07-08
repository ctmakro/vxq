from flask import Flask, request, g, render_template, send_from_directory

import random,threading, time
from utils import *
def random_id(): return random.randint(0, 2**32)

def flask_ui_app(title='GUI', port=9001):

    app = Flask(__name__)
    root = Div()
    root.classes.append('root')

    @app.route('/update', methods=['GET','POST'])
    def update_model_get_view():
        j = request.json
        if 'id' in j:
            click_id = j['id']
            if click_id in click_registry:
                item = click_registry[click_id]
                item.cb(item)

        return render_template('stuff.html.jinja',
            items=root,
        )

    @app.route('/')
    def hello_world():
        return render_template('container.html.jinja',
            title=title,
            items=root,
        )

    @app.route('/css/<path:path>')
    def send_js(path):
        return send_from_directory('templates/css', path)

    def implement(f):
        f(root)
        if __name__ == '__main__':
            app.run(host='0.0.0.0', port=port, debug=True)
        else:
            app.run(host='0.0.0.0', port=port, debug=False)

    return implement

click_registry = {}

class Div:
    def __init__(self, **kw):
        self.d = kw
        self.l = []
        self.type = self.__class__.__name__
        self.cb = None
        self.classes = []
        self.text = ''

    def append(self, i):
        self.l.append(i)

    def __iadd__(self, i):
        self.l+=i
        return self

    def onclick(self, cb):
        self.cb = cb

        self.rid = random_id()
        click_registry[self.rid] = self
        self.d['click_id'] = self.rid

    def click(self):
        if self.cb:
            self.cb(self)

class Button(Div):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.classes.append('button')

if __name__ == '__main__':

    def mvc(root):
        mode = 0
        mode_sw = [Button(), Button()]

        mode_sw[0].text = '主人控制'
        mode_sw[1].text = '自动加茶'

        def switch_to_mode(x):
            def switch(self):
                mode = x
                mode_sw[x].classes = ['button', 'chosen']
                mode_sw[1-x].classes = ['button']
            return switch

        mode_sw[0].onclick(switch_to_mode(0))
        mode_sw[1].onclick(switch_to_mode(1))

        mode_sw[0].click()

        mode_sw_section = Div()
        mode_sw_section.classes.append('buttonrow')
        mode_sw_section += mode_sw
        root += [mode_sw_section]

        # ----------------------

        nb = number_of_buttons = 9
        buttonlist = Div()
        buttons = [Button() for i in range(nb)]
        buttonlist += buttons
        buttonlist.classes.append('buttonrow')

        button_texts = ['']*nb

        for i,b in enumerate(buttons):
            b.onclick(lambda self, j=i:fprint(f'you clicked on button #{j}'))

        def update_buttons():
            for i,s in enumerate(button_texts):
                buttons[i].text = s
                if not s:
                    buttons[i].classes = ['button', 'disabled']
                else:
                    buttons[i].classes = ['button']

        root+=[buttonlist]

        # -----------------------

        feedback_section = fbs = Div()
        fbt = []
        fbs.classes = ['feedback']

        def fprint(*a):
            s = ' '.join((str(i) for i in a))
            line = Div()
            line.classes = ['line']
            line.text = s
            fbs.l.insert(0, line)
            while len(fbs.l)>10:
                fbs.l.pop(-1)

            for i,div in enumerate(fbs.l):
                opacity = 0.8**i
                div.d['style'] = f'opacity:{opacity:.3f};'

        root+=[feedback_section]

        # ----------------------

        def update_randomly():
            if 1:
                time.sleep(1)
                for i,s in enumerate(button_texts):
                    x = random.randint(0,10)
                    button_texts[i] = str(x) if x >5 else ''
                update_buttons()

        run_threaded(update_randomly)

    implement = flask_ui_app('Javis GUI')
    implement(mvc)
