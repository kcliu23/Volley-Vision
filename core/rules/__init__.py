"""Volleyball's basic rules, in one place for every stage of the pipeline.

    fivb      the numbers (net height, court, 3 touches, block / serve zones)
    sequence  rules on a plain touch list (side, player): 4 touches, double touch, the serve
              touched by its own team or blocked. Needs no 3D: for the touch detector, the rally
              detector, events, check_labels
    sides     which side of the net each touch was on, the whole rally at once (needs a first 3D)
    serve     the serve's faults from (x, z) touch points (the 3D fit merges these)

Still elsewhere: in/out, the rally's end and rally-point scoring (core.scene3d.rallies).
Not yet: rotation / service order, back-row attack, libero (need lineups).
"""
