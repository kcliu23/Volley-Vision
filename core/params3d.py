"""Every set number of the 3D step, in one place.

Physics (known, not tuned) stays with the physics in core.ball3d / core.court: gravity, ball
size, air drag, court and net size. What is here is chosen, not measured, so each one says how
it was set:
    [synthetic]  tuned on tools.sim_rally (same physics as the fit: an optimistic test)
    [real]       checked on real clips (test74 / test75)
    [guess]      a reasonable value, not tested yet
As your holdout touch labels come in, these get checked on real clips, and the ones that matter
most (touch cost, player reach) are the candidates to become learned instead of set.
"""

# --- 2D measurement noise (core.rally3d) ---------------------------------------------------------
SIGMA_PX = 3.0         # [real] floor for the 2D noise of a ball position; the actual level is
                       #        measured per rally (jitter, then fit residuals)
SIZE_REL = 0.15        # [guess] relative noise of the ball's box size
HUBER = 2.0            # [guess] residuals beyond this many sigma count linearly
OUT_CAP = 4.0          # [real] a 2D residual costs at most this (|r| = 3 sigma): a wrong point is
                       #        cheaper to ignore than to explain with touches (test74 709-710)
OUT_SIGMA = 6.0        # [guess] points this many sigma off are left out of the final fit

# --- touches: the global choice (core.rally3d) -----------------------------------------------------
TOUCH_COST = 12.0      # [synthetic] score of one touch; THE main number
DV_SOFT = 3.0          # [real] m/s: a touch must change the ball's velocity; one that barely does
                       #        costs up to 2x TOUCH_COST more (over-splitting on test74)
LINK_SIGMA_M = 0.30    # [guess] how well two fast-fitted pieces must meet at a touch
REACH_M = 3.5          # [guess] no touch above this height (jumping reach)
MAX_SPEED = 35.0       # [guess] m/s: faster than any rec hit
MAX_FLIGHT_S = 2.5     # [guess] longest flight between touches
MIN_PTS = 4            # [guess] fewest points in one flight
GRID_S = 0.15          # [synthetic] safety-net touch candidates every this many seconds
TURN_DEG = 25.0        # [guess] a 2D turn this sharp is a touch candidate

# --- players at touches (core.players, core.rally3d) -----------------------------------------------
PLAYER_COST = 12.0     # [real] at most this extra for a touch far from every player (24 -> 14
                       #        touches on test74); capped: a player out of the picture still allows one
PLAYER_SOFT_M = 0.7    # [guess] distance outside reach where that extra is ~63% of PLAYER_COST
REACH_RADIUS_M = 1.0   # [guess] a player's reach around their floor spot
REACH_HEIGHT_X = 1.33  # [guess] standing reach = this x standing height (+ jump height)

# --- jumps (core.players) ----------------------------------------------------------------------------
JUMP_MIN_S, JUMP_MAX_S = 0.12, 1.0   # [real] how long a jump can last
JUMP_H = (0.15, 1.2)   # [real] m: peak FEET height (tucked feet rise ~0.3 m more than the body)
JUMP_FIT_M = 0.15      # [real] m: rms around the gravity parabola of the feet
LIFT_FRAC = 0.08       # [real] feet this far (x body height) above the ground line = a candidate

# --- solid floor and net (core.ball3d.fit_rally) -----------------------------------------------------
SOLID_SIGMA_M = 0.03   # [guess] how hard the floor and net push back

# --- frame timing (core.frametime) ------------------------------------------------------------------
REPEAT = 0.15          # [real] picture change below this x its local level = a repeated frame
SKIP = 1.3             # [real] change above this x local level, next to a repeat = a skipped moment

# --- reporting (tools.ball3d_run) -------------------------------------------------------------------
FREE_X = 2.0           # [real] a free flight fits its 2D points within this many times their jitter
FREE_MIN_PX = 8.0      # [real] ... and is never asked for better than this (median px)
