

tpid = teapad-id

variables:

- add_tea_target (tpid)

- target_state_manager, tsm, for all possible targets

tsm appearance enums:

1. not_shown
2. left_only
3. right_only
4. both_shown

tsm state enums:

1. idle
2. demand tea
3. tea added

```python
if tsm.state == idle:
  if left_only:
    tsm.state = demand tea
    tsm.location = tag.cxy

elif tsm.state==demand tea:
  if right_only or both_shown or tag.cxy-tsm.location too far:
    tsm.state = idle

elif tsm.state==tea_added:
  if right_only or both_shown:
    tsm.state = idle

```

state enums:

1. waiting (merge with moving home)
2. moving to target (add_tea_target), add_tea
3. moving home

```python
if state==waiting:
  if mode==master_only:
    if chosen_target is not None:
      clear chosen_target
      state = moving2target(chosen_target)

  if mode==automatic:
    if exist tsm where state==demand_tea:
      moving2target(tsm.location)

if state==moving2target:
  if mode==master_only:
    if chosen_target.state==idle or tea_added:
      state = moving_home()

  if mode==automatic:
    if chosen_target.state == idle or tea_added:
      state = moving_home()

```

moving_home:

1. raise to move_height
2. ik to home_pos

moving2target:

1. raise to move_height
2. ik to target(height)
3. daocha
4. raise to move_height
