from vxq_hid import *
from arucofun import *
from flasktest import *

import sys
vxq_model = sys.argv[1] if len(sys.argv)>=2 else 'original'

v = VXQHID(configuration=vxq_model)

time.sleep(.5)

cl = camloop(tabletop_square_matcher_gen(), threaded=True)

tst = None
tags = {}


tag_combinations = {
# left_tag(always reveal): right_tag(may be covered by tea cup)
 48:49,
 46:47,
 5:4,
}

finish_adding_tea = {}

master_commanded = {}

from experiment_table_config import mcrcf
home_pos = mcrcf['home_pos']
move_height = mcrcf['move_height']

home_pos_up = home_pos.copy()
home_pos_up[2] = move_height

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


            if not master_only: # automatically add tea without master's command
                if tst is not None and tst.has_solution():

                    k = -1
                    for j in tag_combinations.keys():
                        if j in tags and tag_combinations[j] not in tags:
                            # only left in
                            k = j
                            break

                        elif j in tags and tag_combinations[j] in tags: # both in
                            if j in finish_adding_tea:
                                # marked as finished in previous run
                                del finish_adding_tea[j] # clear status

                    if k==-1: # no target
                        pass
                    elif k not in finish_adding_tea:
                        try_move_to_5(k)
                        break

            else: # only master can give command to add tea
                if tst is not None and tst.has_solution():

                    k = -1

                    # print('master_commanded.keys()', list(master_commanded.keys()))
                    for j in master_commanded.keys():
                    # for j in tag_combinations.keys():
                        if j in tags and tag_combinations[j] not in tags:
                            # only left in
                            k = j
                            break

                        elif j in tags and tag_combinations[j] in tags: # both in
                            if j in finish_adding_tea:
                                # marked as finished in previous run
                                del finish_adding_tea[j] # clear status

                    if k==-1: # no target
                        pass
                    else:
                        try_move_to_5(k)
                        if k in master_commanded:
                            del master_commanded[k]
                        break



            time.sleep(0.3)

def move_back():
    side_up = (mcrcf[1]+mcrcf[2])/2+np.array([0,-300,200])
    base_up = [0, 1, 600]

    # move to starting point
    time.sleep(0.5)
    # v.g1_cartesian_joint(side_up, j1_first=False, speed=800)
    v.g1_cartesian_joint(home_pos_up, j1_first=False, speed=800)
    v.g1_cartesian_joint(home_pos, j1_first=False, speed=800)
    time.sleep(0.5)

def try_move_to_5(target=5):

    left_tag = target
    right_tag = tag_combinations[left_tag]

    # if camera to square coords solution exists
    if tst is not None and tst.has_solution():
        at = AffineTransform()

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
                t5r = t5c.cxy + t5c.rxy * 3.2

                screen = t5r

                sc = tst.transform(screen) # screen to square
                rc = at.transform(sc) # square to robot cartesian

                default_height = mcrcf[0][2]

                # let's move! but from another thread...
                def go_pour_tea():
                    try:
                        cancelled = lambda:v.cancel_movement

                        v.g1_cartesian_joint(home_pos_up, j1_first=False, speed=800)
                        if cancelled(): return

                        v.g1_cartesian_joint(
                        # v.g1_cartesian_ik(
                            list(rc)+[move_height],
                            j1_first = True,
                            speed=600,
                        )
                        if cancelled(): return
                        v.g1_cartesian_joint(
                        # v.g1_cartesian_ik(
                            list(rc)+[default_height],
                            j1_first = True,
                            speed=600,
                        )
                        time.sleep(1)

                        if cancelled(): return
                        # v.go_j(4, 1800) # 出水
                        if cancelled(): return

                        time.sleep(1+3)

                        if cancelled(): return
                        v.go_j(4, 2048) # 停
                        if cancelled(): return

                        time.sleep(2)

                        if cancelled(): return

                        v.g1_cartesian_joint(
                        # v.g1_cartesian_ik(
                            list(rc)+[move_height],
                            j1_first = True,
                            speed=600,
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
                        np.sum(np.square(tags[left_tag].cxy - t5c.cxy))
                        > (t5c.size/2)**2):

                        # cancel_movement
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


# threading.Thread(target=tealoop, daemon=True).start()
run_threaded(tealoop)

master_only = False
update_buttons = lambda:None
fprint = lambda *a:print(*a)
button_texts = ['']*4

def mvc(root):
    global update_buttons, button_texts, fprint

    mode = 0
    mode_sw = [Button(), Button()]

    mode_sw[0].text = '主人控制'
    mode_sw[1].text = '自动加茶'

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


while 1:
    cl.update()
    time.sleep(.03) # 20hz max
    res = cl.result
    if res is not None:
        tags, tst = cl.result
