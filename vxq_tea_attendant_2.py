from vxq_hid import *
from arucofun import *

# time.sleep(.5)

'''camera loop + aruco tags extractor'''
cl_rh = camloop(tabletop_square_matcher_gen(), threaded=True)

tst = None
tags = {}

def wait_for_result():
    '''
    update tst and tags whenever new versions of them became available.
    '''
    global tags, tst
    rb = cl_rh.resultbuffer
    while 1:
        result = rb.recv()
        if result is not None:
            tags, tst = result

run_threaded(wait_for_result)

tag_combinations = {
# left_tag(always reveal): right_tag(may be covered by tea cup)
 48:49,
 46:47,
 44:45,
 42:43,
 40:41,
 38:39,
 36:37,
}

class StatefulTag:
    '''
    class that holds state of each of the targets (teacup trays).
    '''

    def __init__(self, l, r):
        self.l, self.r = l,r
        self.appearance =\
            'not_shown' or 'both_shown' or 'left_only' or 'right_only'

        self.state = 'idle' or 'demand_tea' or 'commanded_tea' or 'tea_added'
        self.tag_location = None

    def monitor(self):
        return f'[{self.l}:{self.r}]{self.appearance}({self.state})'

    def update(self):
        l,r = self.l, self.r

        ltag,rtag = None,None

        if l in tags: ltag = tags[l]
        if r in tags: rtag = tags[r]

        if ltag:
            if rtag:
                self.appearance = 'both_shown'
            else:
                self.appearance = 'left_only'
        else:
            if rtag:
                self.appearance = 'right_only'
            else:
                self.appearance = 'not_shown'


        if self.state == 'idle':
            if self.appearance == 'left_only':
                self.state = 'demand_tea'

        elif self.state == 'demand_tea':
            if self.appearance != 'left_only':
                self.state = 'idle'
            else:
                self.cup_location = ltag.rexy
                self.tag_location = ltag.cxy # keep updating during demand_tea

        elif self.state == 'commanded_tea': # fixed
            if self.appearance in ['right_only', 'both_shown'] \
                or (ltag and
                np.sqrt(np.sum(np.square(ltag.cxy - self.tag_location)))
                > ltag.size*.3):
                self.state = 'idle'

        elif self.state == 'tea_added':
            if self.appearance in ['right_only', 'both_shown']:
                tsm.state = 'idle'


class DaoChaStateMachine:
    def __init__(self, tag_combinations):
        self.state = 'waiting' or 'waiting_more' \
            or 'moving_to_target' or 'moving_home'

        self.mode = 'master_only' or 'automatic'

        self.chosen_targets = []
        self.current_target = None

        self.timestamp = time.time()

        stateful_tags = [StatefulTag(l, r) for l,r in tag_combinations.items()]
        self.tags = stateful_tags

    def monitor(self):
        s = f'''DCSM [{self.mode}]{self.state} ct:{
            self.current_target and self.current_target.l} cts:[{
            ','.join([str(i.l) for i in self.chosen_targets])}]\n'''

        for i in self.all_demands():
            s+=i.monitor()+'\n'
        return s

    def all_demands(self):
        demand = []
        for i in self.tags:
            if i.state == 'demand_tea': demand.append(i)
        return demand

    def update_tags(self):
        for i in self.tags:
            i.update()

    def update(self):
        self.update_tags()

        demands = self.all_demands()

        def next_target_if_any():
            while self.chosen_targets:
                target = self.chosen_targets.pop(0)
                if target.state == 'demand_tea':

                    target.state = 'commanded_tea'
                    self.state = 'moving_to_target'
                    self.current_target = target
                    return True
            self.current_target = None
            return False

        if self.state == 'waiting':
            if self.mode == 'master_only':
                next_target_if_any()

            elif self.mode == 'automatic':
                if demands:
                    self.state = 'waiting_more'
                    self.timestamp = time.time()

        elif self.state == 'waiting_more':

            if self.mode == 'master_only':
                self.state = 'waiting'

            elif self.mode == 'automatic':
                if not demands: # if no demand at all
                    self.state = 'waiting'

                else: # got some demand
                    if time.time() - self.timestamp > 3: # timeout!
                        self.chosen_targets = demands
                        next_target_if_any()

        elif self.state == 'moving_to_target':
            if self.current_target.state in ['idle','tea_added']:
                # regretted or done

                if not next_target_if_any(): # no other targets to add tea
                    self.state = 'moving_home'

        elif self.state == 'moving_home':
            pass

dcsm = DaoChaStateMachine(tag_combinations)

from experiment_table_config import mcrcf
home_pos = mcrcf['home_pos']
move_height = mcrcf['move_height']

home_pos_up = home_pos.copy()
home_pos_up[2] = move_height

ik_move_speed = ims = 200
joint_move_speed = jms = 400

fb_show = None
button_texts, update_buttons = None, None

def new_tealoop():
    global fb_show, button_texts, update_buttons
    while 1:
        time.sleep(0.05)
        dcsm.update()

        if button_texts and update_buttons:
            labels = [i.l for i in dcsm.all_demands()]
            button_texts[0:len(labels)] = labels
            update_buttons()

        if fb_show:
            fb_show(dcsm.monitor())

    def move_home():
        v.g1_cartesian_joint(home_pos_up, speed=jms)
        v.g1_cartesian_joint(home_pos, j1_first=False, speed=jms)
        time.sleep(0.5)
        dcsm.state == 'waiting'

run_threaded(new_tealoop)


def tealoop():
    global button_texts, update_buttons

    while 1:
        move_back()

        while 1:

            # change button texts to visible tags

            visible_tags = []

            for key,value in tag_combinations.items():
                if key in tags and value not in tags:
                    visible_tags.append(key)

            button_texts = ['']*len(button_texts)
            for i,t in enumerate(visible_tags[0:9]):
                button_texts[i] = str(t)

            update_buttons()

            should_break = 0
            if not master_only: # automatically add tea without master's command
                if tst is not None and tst.has_solution():

                    for j in tag_combinations.keys():
                        if j in tags:
                            if tag_combinations[j] not in tags:
                                # only left in
                                if j not in finish_adding_tea:
                                    try_move_to_5(j)
                                    should_break = 1
                                    break
                            else: # both in
                                if j in finish_adding_tea:
                                    # marked as finished in previous run
                                    del finish_adding_tea[j] # clear status

            else: # only master can give command to add tea
                if tst is not None and tst.has_solution():

                    # print('master_commanded.keys()', list(master_commanded.keys()))
                    for j in master_commanded.keys():
                    # for j in tag_combinations.keys():
                        if j in tags:
                            if tag_combinations[j] not in tags:
                                # only left in
                                try_move_to_5(j)
                                should_break = 1
                                if j in master_commanded:
                                    del master_commanded[j]
                                break

                            else: # both in
                                if j in finish_adding_tea:
                                    # marked as finished in previous run
                                    del finish_adding_tea[j] # clear status

            if should_break:
                break
            time.sleep(0.3)



def move_back():
    side_up = (mcrcf[1]+mcrcf[2])/2+np.array([0,-300,200])
    base_up = [0, 1, 600]

    # move to starting point
    # time.sleep(0.5)
    # v.g1_cartesian_joint(side_up, j1_first=False, speed=800)

    v.g1_cartesian_ik(home_pos_up, speed=ims)
    # v.g1_cartesian_joint(home_pos_up, j1_first=False, speed=800)
    v.g1_cartesian_joint(home_pos, j1_first=False, speed=jms)
    time.sleep(0.5)

def try_move_to_5(target=5):

    left_tag = target
    right_tag = tag_combinations[left_tag]

    # if camera to square coords solution exists
    if tst is not None and tst.has_solution():
        at = PerspectiveTransform()

        # try to get solution from camera to table coords
        usc_l = np.array([unit_square_coords[i] for i in range(4)])
        mcrcf_l = np.array([mcrcf[i][0:2] for i in range(4)])
        at.estimate_from(usc_l, mcrcf_l)

        # print('usc',usc_l)
        # print('mcrcf',mcrcf_l)
        # print(at.transform(np.array([0.5, 0.5])))

        # if solution from camera to table coords exists:
        if at.has_solution():
            # 5 not occluded, 4 occluded, meaning cup over 4
            # if 5 in tags and 4 not in tags:
            if left_tag in tags and right_tag not in tags:

                tag5 = tags[left_tag]
                # to the right of tag 5
                t5c = Detection(tag5.marker_id, tag5.corners) # make copy
                t5r = t5c.rexy

                screen = t5r

                sc = tst.transform(screen) # screen to square
                rc = at.transform(sc) # square to robot cartesian

                default_height = -90
                default_height = mcrcf[0][2]
                #倒茶高度

                # let's move! but from another thread...
                def go_pour_tea():
                    try:
                        cancelled = lambda:v.cancel_movement

                        v.g1_cartesian_joint(home_pos_up, j1_first=False, speed=jms)
                        if cancelled(): return

                        # v.g1_cartesian_joint(
                        # v.g1_cartesian_ik(
                        #     list(rc)+[move_height],
                        #     # j1_first = True,
                        #     speed=ims,
                        #
                        #     start=home_pos_up,
                        # )
                        v.g1_cartesian_ik(
                            list(rc)+[default_height],
                            # j1_first = True,
                            speed=ims,

                            start=home_pos_up,
                        )
                        if cancelled(): return
                        # v.g1_cartesian_joint(
                        # # v.g1_cartesian_ik(
                        #     list(rc)+[default_height],
                        #     j1_first = True,
                        #     speed=jms,
                        # )
                        time.sleep(1)

                        if cancelled(): return
                        v.go_j(4, 1800) # 出水
                        if cancelled(): return

                        time.sleep(3)

                        if cancelled(): return
                        v.go_j(4, 2048) # 停

                        time.sleep(1)

                        v.g1_cartesian_joint(
                        # v.g1_cartesian_ik(
                            list(rc)+[move_height],
                            j1_first = True,
                            speed=jms,
                        )

                    except Exception as e:
                        # point might get out of reach or something
                        print(e)

                th = threading.Thread(target=go_pour_tea, daemon=True)
                th.start()

                # check if target moved

                success = True
                while th.is_alive():
                    time.sleep(0.05)

                    # if 4 become visible, or
                    # tag 5 moved for more than size / 2
                    if right_tag in tags or (left_tag in tags and
                        np.sqrt(np.sum(np.square(tags[left_tag].cxy - t5c.cxy
                        )))) > (t5c.size*0.5):

                        # cancel_movement
                        v.go_j(4, 2048) # 停
                        v.cancel_movement = True
                        th.join()
                        v.cancel_movement = False
                        v.go_j(4, 2048) # 停

                        success = False
                        break

                if success:
                    print(f'成功加茶至({left_tag},{right_tag})')
                    fprint(f'成功加茶至({left_tag},{right_tag})')
                else:
                    print(f'加茶至({left_tag},{right_tag})失败')
                    fprint(f'加茶至({left_tag},{right_tag})失败')


                finish_adding_tea[left_tag] = 1


# '''vxq init'''
# import sys
# vxq_model = sys.argv[1] if len(sys.argv)>=2 else 'original'
# v = VXQHID(configuration=vxq_model)

# threading.Thread(target=tealoop, daemon=True).start()
# run_threaded(tealoop)

from flasktest import *

master_only = False
update_buttons = lambda:None
fprint = lambda *a:print(*a)
fb_show = lambda s:print(s)
button_texts = ['']*9

def mvc(root):
    global update_buttons, button_texts, fprint, fb_show

    mode = 0
    mode_sw = [Button(), Button(), Button()]

    mode_sw[0].text = '主人控制'
    mode_sw[1].text = '自动加茶'
    mode_sw[2].text = '暂无功能'

    mode_sw[2].classes = ['button', 'disabled']

    def switch_to_mode(x):
        def switch(self):
            global master_only

            mode = x
            mode_sw[x].classes = ['button', 'chosen']
            mode_sw[1-x].classes = ['button']

            master_only = x==0
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
    # move to global variable

    for i,b in enumerate(buttons):
        def cb(self, j=i):
            fprint(f'you clicked on button #{j} with label {repr(self.text)}')

            if self.text:
                master_commanded[int(self.text)] = 1
                fprint(f'added {self.text} to master_commanded, length {len(master_commanded)}')

        b.onclick(cb)

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


    feedback_section2 = fbs2 = PreCode()
    fbs2.text = 'example text'
    def fb_show(s):
        fbs2.text = s
    root+=[feedback_section2]

    # ----------------------

    # def update_randomly():
    #     if 1:
    #         time.sleep(1)
    #         for i,s in enumerate(button_texts):
    #             x = random.randint(0,10)
    #             button_texts[i] = str(x) if x >5 else ''
    #         update_buttons()
    #
    # run_threaded(update_randomly)

implement = flask_ui_app('Javis GUI')
run_threaded(lambda:implement(mvc))

while 1: # imshow loop
    cl_rh.update()
