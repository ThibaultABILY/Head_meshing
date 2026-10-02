# -*- coding: utf-8 -*-
# FreeCAD macro: ellipsoid head + nose (right-triangle prism) + ears + neck + blind mouth cut
# Part API (FreeCAD 0.21 / 1.0). Front of the head faces -Y.

import math
import FreeCAD as App
import Part

V = App.Vector

# ---------------- Parameters (mm) ----------------
# Head: ellipsoid semi-axes
A = 80.0    # X, half width
B = 95.0    # Y, half depth
C = 120.0   # Z, half height

# Nose: prism with a right-triangle profile (side view), edges filleted
#   - vertical side (the right angle) sunk into the face
#   - horizontal base at the bottom, ending in a point toward the front
#   - hypotenuse = bridge of the nose, from the top (between the eyes) to the tip
NEZ_Z_HAUT  = 30.0    # top of the triangle (the nose only emerges lower down, see NEZ_ENFONCE)
NEZ_Z_BAS   = -25.0   # nose base (under the nostrils)
NEZ_SAILLIE = 20.0    # how far the tip sticks out past the face surface
NEZ_LARG    = 22.0    # nose width (left/right)
NEZ_ENFONCE = 8.0     # how deep the vertical side is buried in the head
NEZ_ARRONDI = 3.0     # fillet radius (0 = sharp edges)

# Ears: ellipsoids flattened on the sides
OREILLE_Z  = -10.0    # center height
OREILLE_Y  = 5.0      # shift toward the back (+Y = back)
OREILLE_RX = 9.0      # half thickness (left/right)
OREILLE_RY = 13.0     # half depth
OREILLE_RZ = 26.0     # half height
OREILLE_ENFONCE = 1.0 # how far the center sits inside the surface

# Neck: elliptical truncated cone, flares toward the bottom and blends into the head
COU_RX_HAUT = 28.0    # half width at the top (head side, buried in the ellipsoid)
COU_RX_BAS  = 42.0    # half width at the bottom (shoulder side); swap the two for an inverted cone
COU_RATIO_Y = 1.12    # depth / width ratio (1 = circular section)
COU_Z_HAUT  = -80.0   # top of the neck (inside the head)
COU_Z_BAS   = -190.0  # bottom of the neck
COU_DECAL_Y = 10.0    # shift toward the back (+Y), bigger = stronger chin

# Mouth
Z0      = -45.0   # height of the mouth center (0 = head center)
W       = 28.0    # mouth half width (total length = 2 * W)
HAUT    = 16.0    # total slot height (rounded ends, radius HAUT/2)
PROF    = 8.0     # depth of the recess, measured at the mouth center
# -------------------------------------------------


def ellipsoide(rx, ry, rz, centre=V(0, 0, 0)):
    # unit sphere, scaled non-uniformly, then moved into place
    s = Part.makeSphere(1.0)
    m = App.Matrix()
    m.scale(rx, ry, rz)
    s = s.transformGeometry(m)
    s.translate(centre)
    return s


def ajoute(nom, shape, visible=True):
    obj = App.ActiveDocument.addObject("Part::Feature", nom)
    obj.Shape = shape
    return obj


doc = App.newDocument("Tete")

# --- 1) Head ---
head = ajoute("Ellipsoide", ellipsoide(A, B, C))

# --- 2) Nose ---
def y_surface_avant(z):
    """Y of the front surface of the head (x = 0) at height z."""
    return -B * math.sqrt(max(0.0, 1.0 - (z / C) ** 2))

x_nez = -NEZ_LARG / 2.0
y_base = y_surface_avant(NEZ_Z_BAS)
y_dos = y_base + NEZ_ENFONCE               # vertical side, inside the head
y_bout = y_base - NEZ_SAILLIE              # nose tip

p_haut = V(x_nez, y_dos,  NEZ_Z_HAUT)      # top vertex (between the eyes, inside the head)
p_coin = V(x_nez, y_dos,  NEZ_Z_BAS)       # right angle (bottom, inside)
p_bout = V(x_nez, y_bout, NEZ_Z_BAS)       # tip

profil = Part.makePolygon([p_haut, p_coin, p_bout, p_haut])
nez_shape = Part.Face(profil).extrude(V(NEZ_LARG, 0, 0))   # extruded toward +X

if NEZ_ARRONDI > 0:
    try:
        nez_shape = nez_shape.makeFillet(NEZ_ARRONDI, nez_shape.Edges)
    except Exception:
        App.Console.PrintWarning("Nose fillet failed, keeping sharp edges.\n")

nez = ajoute("Nez", nez_shape)

# --- 3) Ears (left and right, mirrored) ---
x_surf = A * math.sqrt(max(0.0, 1.0 - (OREILLE_Y / B) ** 2 - (OREILLE_Z / C) ** 2))
x_centre = x_surf - OREILLE_ENFONCE
oreille_d = ajoute("Oreille_D", ellipsoide(OREILLE_RX, OREILLE_RY, OREILLE_RZ,
                                           V( x_centre, OREILLE_Y, OREILLE_Z)))
oreille_g = ajoute("Oreille_G", ellipsoide(OREILLE_RX, OREILLE_RY, OREILLE_RZ,
                                           V(-x_centre, OREILLE_Y, OREILLE_Z)))

# --- 3b) Neck (the chin is just the front of the head sticking out past it) ---
# makeCone(base_radius, top_radius, height): Z axis, base at z = 0
cone = Part.makeCone(COU_RX_BAS, COU_RX_HAUT, COU_Z_HAUT - COU_Z_BAS)
mc = App.Matrix()
mc.scale(1.0, COU_RATIO_Y, 1.0)           # circle -> ellipse
cone = cone.transformGeometry(mc)
cone.translate(V(0, COU_DECAL_Y, COU_Z_BAS))
cou = ajoute("Cou", cone)

# --- 4) Fuse head + nose + ears + neck ---
fusion = doc.addObject("Part::MultiFuse", "Tete_complete")
fusion.Shapes = [head, nez, oreille_d, oreille_g, cou]
fusion.Refine = True

# --- 5) Mouth profile: oblong slot (two lines + two half circles) ---
y_start = -B - 10.0                      # cutter starts outside the head
r = HAUT / 2.0

haut_g = V(-(W - r), y_start, Z0 + r)
haut_d = V( (W - r), y_start, Z0 + r)
bas_g  = V(-(W - r), y_start, Z0 - r)
bas_d  = V( (W - r), y_start, Z0 - r)

ligne_haut = Part.makeLine(haut_g, haut_d)
ligne_bas  = Part.makeLine(bas_d, bas_g)
arc_droit  = Part.Arc(haut_d, V(W, y_start, Z0), bas_d).toShape()
arc_gauche = Part.Arc(bas_g, V(-W, y_start, Z0), haut_g).toShape()

wire = Part.Wire(Part.__sortEdges__([ligne_haut, arc_droit, ligne_bas, arc_gauche]))
face = Part.Face(wire)

# --- 6) Blind extrusion: stops PROF below the surface (at the center) ---
y_surface = -B * math.sqrt(max(0.0, 1.0 - (Z0 / C) ** 2))   # front surface, x = 0
y_fin = y_surface + PROF
longueur = y_fin - y_start

cutter = ajoute("Cutter_bouche", face.extrude(V(0, longueur, 0)))

# --- 7) Subtract the mouth ---
tete = doc.addObject("Part::Cut", "Tete_finale")
tete.Base = fusion
tete.Tool = cutter

doc.recompute()

try:
    import FreeCADGui as Gui
    for o in (head, nez, oreille_d, oreille_g, cou, fusion, cutter):
        o.ViewObject.Visibility = False
    Gui.activeDocument().activeView().viewFront()
    Gui.SendMsg("ViewFit")
except Exception:
    pass
