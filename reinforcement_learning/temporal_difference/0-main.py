def policy(s):
    if s % 8 != 7 and env.unwrapped.desc[s // 8, s % 8 + 1] != b'H':
        return RIGHT
    elif s // 8 != 7 and env.unwrapped.desc[s // 8 + 1, s % 8] != b'H':
        return DOWN
    elif s // 8 != 0 and env.unwrapped.desc[s // 8 - 1, s % 8] != b'H':
        return UP
    else:
        return LEFT
