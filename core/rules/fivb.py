"""The numbers volleyball's rules give (FIVB Official Volleyball Rules 2025-2028), and the ones we
picked to apply them to what the cameras see ([guess]: tuned on the holdouts, not from the book).
Every rule module (sides, serve, sequence) takes its numbers from here."""

NET_H_MEN_M = 2.43        # FIVB 2.1.1
NET_H_WOMEN_M = 2.24
COURT_HALF_LEN_M = 9.0    # FIVB 1.1: 18 x 9 m; the end line is 9 m from the net
MAX_TOUCHES = 3           # FIVB 9.1: a team has at most 3 hits (a block doesn't count, 14.4.1)

BLOCK_NEAR_M = 1.0        # a block is within this of the net ...
BLOCK_BELOW_TAPE_M = 0.3  # ... and no lower than this under the tape

SERVE_START_Z_M = 1.5    # [guess] the serve is hit at least this high (held / bounced balls are lower)
SERVE_LOOK = 3           # the serve is among a rally's first 3 points (start + 2 touches): later, a
                         #   far-back high touch is a back-row player's, not a serve
SERVE_HIT_X_M = 6.5      # [guess] the serve is hit behind the end line (9 m), a jump serve ~1 m inside
                         #   it; the 3D put the 9 marked holdout serves at 7.1-12.1 m: a join this
                         #   far back on the server's side is still the server (toss -> hit)
FLOOR_Z_M = 0.25         # a join this low is the ball on the floor: the rally is over
