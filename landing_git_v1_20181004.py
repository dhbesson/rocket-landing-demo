""" Written by David Besson, 5 Oct 2018"""

import time
import krpc
from numpy import *
import pandas as pd

# Initiate krpc objects
conn = krpc.connect(name='Landing Site')
vessel = conn.space_center.active_vessel
body = vessel.orbit.body
ap = vessel.auto_pilot
create_relative = conn.space_center.ReferenceFrame.create_relative
canvas = conn.ui.stock_canvas

# Get the size of the game window in pixels
screen_size = canvas.rect_transform.size

# Add a panel to contain the UI elements
panel = canvas.add_panel()

# Position the panel on the left of the screen
rect = panel.rect_transform
rect.size = (400,200)
rect.position = (200.-(screen_size[0]/2.), 0)

# Add some text displaying the current and final errors
text = panel.add_text("Horizontal Error (m): ")
text.rect_transform.size = (325.,30.)
text.rect_transform.position = (-20, 50)
text.color = (1, 1, 1)
text.size = 18

text2 = panel.add_text("Predicted Final Error (m): ")
text2.rect_transform.size = (325.,30.)
text2.rect_transform.position = (-20, 20)
text2.color = (1, 1, 1)
text2.size = 18

# Add some text describing what the code is doing right now
text3 = panel.add_text("Launching to 18,000 m. Adjusting heading towards landing site. Pitch at 85 deg.")
text3.rect_transform.size = (325.,100.)
text3.rect_transform.position = (-20, -60)
text3.color = (1, 1, 1)
text3.size = 18

# Coordinates of landing site
landing_latitude = -(1.+(31./60)+(15./60/60))
landing_longitude = -(71.+(53./60)+(50./60/60))
landing_altitude = 20


# Determine landing site reference frame
# (orientation: x=zenith, y=north, z=east)
landing_position = body.surface_position(
    landing_latitude, landing_longitude, body.reference_frame)
q_long = (
    0,
    sin(-landing_longitude * 0.5 * pi / 180),
    0,
    cos(-landing_longitude * 0.5 * pi / 180)
)
q_lat = (
    0,
    0,
    sin(landing_latitude * 0.5 * pi / 180),
    cos(landing_latitude * 0.5 * pi / 180)
)

landing_reference_frame = \
    create_relative(
        create_relative(
            create_relative(
                body.reference_frame,
                landing_position,
                q_long),
            (0, 0, 0),
            q_lat),
        (landing_altitude, 0, 0))


# Create krpc data streams
altitude = conn.add_stream(getattr, vessel.flight(), 'surface_altitude')
ut = conn.add_stream(getattr, conn.space_center, 'ut')
velocity = conn.add_stream(vessel.velocity,landing_reference_frame)
v_speed = conn.add_stream(getattr,vessel.flight(landing_reference_frame), 'vertical_speed')
h_speed = conn.add_stream(getattr,vessel.flight(landing_reference_frame), 'horizontal_speed')
position = conn.add_stream(vessel.position,landing_reference_frame)
rotation = conn.add_stream(vessel.rotation,landing_reference_frame)
direction = conn.add_stream(vessel.direction,landing_reference_frame)
angular_velocity = conn.add_stream(vessel.angular_velocity,landing_reference_frame)
aero_force = conn.add_stream(getattr, vessel.flight(), 'aerodynamic_force')
Engine = vessel.parts.engines[0]
gimb_angle = conn.add_stream(vessel.parts.engines[0].thrusters[0].part.direction,vessel.reference_frame)
gimb_pos = conn.add_stream(vessel.parts.engines[0].thrusters[0].part.position,vessel.reference_frame)
mass = conn.add_stream(getattr, vessel, 'mass')
moment_of_inertia = conn.add_stream(getattr, vessel, 'moment_of_inertia')
inertia_tensor = conn.add_stream(getattr, vessel, 'inertia_tensor')
com = conn.add_stream(getattr, vessel.flight(), 'center_of_mass')
available_torque = conn.add_stream(getattr, vessel, 'available_torque')
g = conn.add_stream(getattr,body, 'surface_gravity')
latitude = conn.add_stream(getattr, vessel.flight(), 'latitude')
longitude = conn.add_stream(getattr, vessel.flight(), 'longitude')
atm_density = conn.add_stream(getattr, vessel.flight(), 'atmosphere_density')
dyn_pressure = conn.add_stream(getattr, vessel.flight(), 'dynamic_pressure')
stat_pressure = conn.add_stream(getattr, vessel.flight(), 'static_pressure')
lift = conn.add_stream(getattr, vessel.flight(), 'lift')
drag = conn.add_stream(getattr, vessel.flight(), 'drag')
pitch = conn.add_stream(getattr, vessel.flight(), 'pitch')
heading = conn.add_stream(getattr, vessel.flight(), 'heading')
roll = conn.add_stream(getattr, vessel.flight(), 'roll')
throttle = conn.add_stream(getattr, vessel.control, 'throttle')
right = conn.add_stream(getattr, vessel.control, 'right')
up = conn.add_stream(getattr, vessel.control, 'up')
pitch_control = conn.add_stream(getattr, vessel.control, 'pitch')
roll_control = conn.add_stream(getattr, vessel.control, 'roll')
yaw_control = conn.add_stream(getattr, vessel.control, 'yaw')
vessel.control.activate_next_stage()
available_thrust = conn.add_stream(getattr,vessel,'available_thrust')
roll_pid = conn.add_stream(getattr, ap,'roll_pid_gains')
yaw_pid = conn.add_stream(getattr, ap,'yaw_pid_gains')
pitch_pid = conn.add_stream(getattr, ap,'pitch_pid_gains')

# Draw landing reference frame
land_x = conn.drawing.add_line((0, 0, 0), (50000, 0, 0), landing_reference_frame)
land_y = conn.drawing.add_line((0, 0, 0), (0, 50000, 0), landing_reference_frame)
land_z = conn.drawing.add_line((0, 0, 0), (0, 0, 50000), landing_reference_frame)

land_x.color = (0,255,0)
land_y.color = (0,255,0)
land_z.color = (0,255,0)

land_x.thickness = 5.
land_y.thickness = 5.
land_z.thickness = 5.

# Draw drone ship
verts = [(1., 52. / 2., 91. / 2.),
             (1., -52. / 2., 91. / 2.),
             (1., -52. / 2., -91. / 2.),
             (1., 52. / 2., -91. / 2.)]

droneship = conn.drawing.add_polygon(verts, landing_reference_frame)
droneship.thickness = 5.

# Define autopilot reference frame
ap.reference_frame = vessel.surface_reference_frame
time.sleep(0.01)
ap.engage()
time.sleep(0.01)

# Create empty data logging arrays
alt = []
t = []
vel_0 = []
vel_1 = []
vel_2 = []
v_sp = []
h_sp = []
pos_0 = []
pos_1 = []
pos_2 = []
rot_0 = []
rot_1 = []
rot_2 = []
rot_3 = []
dir_0 = []
dir_1 = []
dir_2 = []
ang_vel_0 = []
ang_vel_1 = []
ang_vel_2 = []
m_kg = []
MOI_0 = []
MOI_1 = []
MOI_2 = []
i_tensor_0 = []
i_tensor_1 = []
i_tensor_2 = []
i_tensor_3 = []
i_tensor_4 = []
i_tensor_5 = []
i_tensor_6 = []
i_tensor_7 = []
i_tensor_8 = []
CM_0 = []
CM_1 = []
CM_2 = []
av_torq_0 = []
av_torq_1 = []
av_torq_2 = []
av_torq_3 = []
av_torq_4 = []
av_torq_5 = []
av_thrust = []
lat = []
long = []
throt = []
atm_den = []
dyn_pres = []
stat_pres = []
aero_force_0 = []
aero_force_1 = []
aero_force_2 = []
lift_force_0 = []
lift_force_1 = []
lift_force_2 = []
drag_force_0 = []
drag_force_1 = []
drag_force_2 = []
t_F = []
r_F = []
u_F = []
p_Nm = []
r_Nm = []
y_Nm = []
p_deg = []
head_deg = []
roll_deg = []
ut_0 = ut()
roll_gains_0 = []
roll_gains_1 = []
roll_gains_2 = []
pitch_gains_0 = []
pitch_gains_1 = []
pitch_gains_2 = []
yaw_gains_0 = []
yaw_gains_1 = []
yaw_gains_2 = []

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return arccos(clip(dot(v1_u, v2_u), -1.0, 1.0))

def logdata():
    """Adds current data stream values to data logging arrays"""
    alt.append(altitude())
    t.append(ut() - ut_0)
    vel_0.append(velocity()[0])
    vel_1.append(velocity()[1])
    vel_2.append(velocity()[2])
    v_sp.append(v_speed())
    h_sp.append(h_speed())
    pos_0.append(position()[0])
    pos_1.append(position()[1])
    pos_2.append(position()[2])
    rot_0.append(rotation()[0])
    rot_1.append(rotation()[1])
    rot_2.append(rotation()[2])
    rot_3.append(rotation()[3])
    dir_0.append(direction()[0])
    dir_1.append(direction()[1])
    dir_2.append(direction()[2])
    ang_vel_0.append(angular_velocity()[0])
    ang_vel_1.append(angular_velocity()[1])
    ang_vel_2.append(angular_velocity()[2])
    m_kg.append(mass())
    MOI_0.append(moment_of_inertia()[0])
    MOI_1.append(moment_of_inertia()[0])
    MOI_2.append(moment_of_inertia()[0])
    i_tensor_0.append(inertia_tensor()[0])
    i_tensor_1.append(inertia_tensor()[1])
    i_tensor_2.append(inertia_tensor()[2])
    i_tensor_3.append(inertia_tensor()[3])
    i_tensor_4.append(inertia_tensor()[4])
    i_tensor_5.append(inertia_tensor()[5])
    i_tensor_6.append(inertia_tensor()[6])
    i_tensor_7.append(inertia_tensor()[7])
    i_tensor_8.append(inertia_tensor()[8])
    CM_0.append(com()[0])
    CM_1.append(com()[1])
    CM_2.append(com()[2])
    av_torq_0.append(available_torque()[0][0])
    av_torq_1.append(available_torque()[0][1])
    av_torq_2.append(available_torque()[0][2])
    av_torq_3.append(available_torque()[1][0])
    av_torq_4.append(available_torque()[1][1])
    av_torq_5.append(available_torque()[1][2])
    av_thrust.append(available_thrust())
    lat.append(latitude())
    long.append(longitude())
    throt.append(u)
    atm_den.append(atm_density())
    dyn_pres.append(dyn_pressure())
    stat_pres.append(stat_pressure())
    aero_force_0.append(aero_force()[0])
    aero_force_1.append(aero_force()[1])
    aero_force_2.append(aero_force()[2])
    lift_force_0.append(lift()[0])
    lift_force_1.append(lift()[1])
    lift_force_2.append(lift()[2])
    drag_force_0.append(drag()[0])
    drag_force_1.append(drag()[1])
    drag_force_2.append(drag()[2])
    t_F.append(throttle())
    r_F.append(right())
    u_F.append(up())
    p_Nm.append(pitch_control())
    r_Nm.append(roll_control())
    y_Nm.append(yaw_control())
    p_deg.append(pitch())
    head_deg.append(heading())
    roll_deg.append(roll())
    pitch_gains_0.append(pitch_pid()[0])
    pitch_gains_1.append(pitch_pid()[1])
    pitch_gains_2.append(pitch_pid()[2])
    roll_gains_0.append(roll_pid()[0])
    roll_gains_1.append(roll_pid()[1])
    roll_gains_2.append(roll_pid()[2])
    yaw_gains_0.append(yaw_pid()[0])
    yaw_gains_1.append(yaw_pid()[1])
    yaw_gains_2.append(yaw_pid()[2])


def predict_landing(posx,posy,posz,velx,vely,velz):
    """Solves a basic quadratic equation to predict where the vehicle will land if engine is off"""
    zf = 20
    coeff = [-0.5*g(),velz,posz - zf]
    t = roots(coeff)
    t_num = [float(t[0]),float(t[1])]
    tf = max(t_num)
    xf = posx + 1.0*velx*tf
    yf = posy + 1.0*vely*tf
    return xf,yf,zf

def calculate_thrust():
    """Solves a basic sum of forces equation to determine direction and magnitude of force required to align
    velocity vector with position vector"""

    # Define all vectors in the vessel reference frame
    pos0 = (-position()[0], -position()[1], -position()[2])
    pos0_vessel = conn.space_center.transform_direction(pos0, landing_reference_frame, vessel.reference_frame)
    vel0 = (velocity()[0], velocity()[1], velocity()[2])
    vel0_vessel = conn.space_center.transform_direction(vel0, landing_reference_frame, vessel.reference_frame)
    phat = unit_vector(pos0_vessel)
    grav_force = (-mass()*g(), 0, 0)
    grav_force_vessel = conn.space_center.transform_direction(grav_force, landing_reference_frame,
                                                              vessel.reference_frame)
    aeroforces_vessel = conn.space_center.transform_direction(aero_force(),
                                                              vessel.surface_reference_frame,
                                                              vessel.reference_frame)

    # Desired final velocity vector
    vel0_final_vessel = 200 * phat

    # Difference between current and final velocity vectors
    delta_v_vessel = vel0_final_vessel - vel0_vessel

    # Verrry simple "integration" block.  Determines how hard the engine will burn.
    delta_t = 1

    # Solve for the thrust vector required to achieve the desired velocity vector change
    Fthrust = (
        (mass() / delta_t) * delta_v_vessel[0] - aeroforces_vessel[0] - grav_force_vessel[0],
        (mass() / delta_t) * delta_v_vessel[1] - aeroforces_vessel[1] - grav_force_vessel[1],
        (mass() / delta_t) * delta_v_vessel[2] - aeroforces_vessel[2] - grav_force_vessel[2])

    # Define the vector in the vessel surface reference frame (for the autopilot)
    Fthrust_surface = conn.space_center.transform_direction(Fthrust,
                                                              vessel.reference_frame,
                                                              vessel.surface_reference_frame)

    # Convert the thrust vector to a unit vector (improves autopilot performance)
    F_dir = unit_vector(Fthrust_surface)

    # Predict where the vessel is going to land
    (posx, posy, posz) = predict_landing(position()[2], position()[1], position()[0], velocity()[2], velocity()[1],
                                         velocity()[0])
    land_error = sqrt(posx**2. + posy**2.)

    # Determine the thrust required to cancel out gravity acceleration (assuming vessel is vertical)
    hover_throt = mass() * g() / available_thrust()

    # A heuristic to smooth out the burn so that the final error doesn't overshoot
    if land_error > 2000:
        des_throt = 2.0 * hover_throt
    elif land_error > 1000:
        des_throt = 1.25 * hover_throt
    elif land_error > 500:
        des_throt = 1.0 * hover_throt
    else:
        des_throt = 0.75 * hover_throt

    # Set the throttle and autopilot direction
    vessel.control.throttle = des_throt
    ap.target_direction = F_dir

def calculate_cruise():
    """Solves a basic sum of forces equation to determine direction and magnitude of force required to align
        horizontal components (in landing frame) of velocity and position vectors"""

    # Define all vectors in the vessel surface reference frame
    pos0 = (-position()[0], -position()[1], -position()[2])
    pos0_vessel = conn.space_center.transform_direction(pos0, landing_reference_frame, vessel.surface_reference_frame)
    vel0 = (velocity()[0], velocity()[1], velocity()[2])
    vel0_vessel = conn.space_center.transform_direction(vel0, landing_reference_frame, vessel.surface_reference_frame)
    phat = unit_vector((pos0_vessel[1],pos0_vessel[2]))
    grav_force = (-mass()*g(), 0, 0)
    grav_force_vessel = conn.space_center.transform_direction(grav_force, landing_reference_frame,
                                                              vessel.surface_reference_frame)
    aeroforces_vessel = aero_force()

    # Desired final velocity vector
    vel0_final_vessel = 400 * phat

    # Difference between current and final velocity vectors
    delta_v_vessel = vel0_final_vessel - (vel0_vessel[1],vel0_vessel[2])

    # Verrry simple "integration" block.  Determines how hard the engine will burn.
    delta_t = 30.

    # Solve for the thrust vector required to achieve the desired velocity vector change
    Fthrust = (0,
               (mass() / delta_t) * delta_v_vessel[0] - aeroforces_vessel[1] - grav_force_vessel[1],
               (mass() / delta_t) * delta_v_vessel[1] - aeroforces_vessel[2] - grav_force_vessel[2])

    # Calculate magnitude and direction of thrust vector
    F_des = linalg.norm(Fthrust)/available_thrust()
    F_dir = unit_vector(Fthrust)

    # Set the throttle and autopilot direction
    vessel.control.throttle = F_des
    ap.target_direction = F_dir


def draw_lines(ship_pos, ship_vel, landing_est, landing_est2, combined_line):
    """Draw visualization lines in the game
    - Green: landing reference frame
    - White: position
    - Red: Velocity
    - Blue: Acceleration
    - Red X: Predicted landing spot
    - White rectangle: dimensions of droneship"""

    # Position
    (posx, posy, posz) = position()
    pos0 = (-position()[0], -position()[1], -position()[2])
    pos0_vessel = conn.space_center.transform_direction(pos0, landing_reference_frame, vessel.reference_frame)

    # Velocity
    vel0 = (velocity()[0], velocity()[1], velocity()[2])
    vel0_vessel = conn.space_center.transform_direction(vel0, landing_reference_frame, vessel.reference_frame)

    # Acceleration due to gravity
    grav_force = (-g(),0,0)
    grav_force_vessel = conn.space_center.transform_direction(grav_force,landing_reference_frame,vessel.reference_frame)

    # Acceleration due to ngine thrust
    thrust = available_thrust() * throttle() / mass()

    # Acceleration due to aerodynamic forces
    aeroforces_vessel = conn.space_center.transform_direction((aero_force()[0] / mass(),
                                                               aero_force()[1] / mass(),
                                                               aero_force()[2] / mass()),
                                                              vessel.surface_reference_frame,
                                                              vessel.reference_frame)

    # Acceleration due to all forces
    combined_forces = (
                          (aeroforces_vessel[0] + gimb_angle()[0] * thrust + grav_force_vessel[0]),
                          (aeroforces_vessel[1] + gimb_angle()[1] * thrust + grav_force_vessel[1]),
                          (aeroforces_vessel[2] + gimb_angle()[2] * thrust + grav_force_vessel[2]))


    # Show the predicted landing spot if altitude over 500m
    if altitude() > 500:
        (posx, posy, posz) = predict_landing(position()[2], position()[1], position()[0], velocity()[2], velocity()[1],
                                             velocity()[0])
        text2.content = "Predicted Final Error (m): %.2f" % (linalg.norm((posx, posy)))
        landing_est.visible = True
        landing_est2.visible = True
    else:
        text2.content = "Predicted Final Error (m): " % (linalg.norm((posx, posy)))
        landing_est.visible = False
        landing_est2.visible = False

    # Update line visualization data
    combined_line.end = combined_forces
    ship_pos.end = pos0_vessel
    ship_vel.end = vel0_vessel
    landing_est.start = (posz + 100, posy - 30, posx - 30)
    landing_est.end = (posz + 100, posy + 30, posx + 30)
    landing_est2.start = (posz + 100, posy + 30, posx - 30)
    landing_est2.end = (posz + 100, posy - 30, posx + 30)

    text.content = "Horizontal Error (m): %.2f" % (linalg.norm((position()[1],position()[2])))

    return ship_pos, ship_vel, landing_est, landing_est2, combined_line

# Control vessel from middle docking port. Makes everything smoother.
part = vessel.parts.with_title('Clamp-O-Tron Docking Port')[0]
vessel.parts.controlling = part

# Create initial drawing objects
# Acceleration
combined_line = conn.drawing.add_line((0, 0, 0), (0,0,0), vessel.reference_frame)
combined_line.color = (0, 0, 255)
combined_line.thickness = 3.

# Position
pos0 = (-position()[0], -position()[1], -position()[2])
pos0_vessel = conn.space_center.transform_direction(pos0, landing_reference_frame, vessel.reference_frame)
ship_pos = conn.drawing.add_line((0, 0, 0), pos0_vessel, vessel.reference_frame)
ship_pos.thickness = 1.

# Velocity
vel0 = (velocity()[0], velocity()[1], velocity()[2])
vel0_vessel = conn.space_center.transform_direction(vel0, landing_reference_frame, vessel.reference_frame)
ship_vel = conn.drawing.add_line((0, 0, 0), vel0_vessel, vessel.reference_frame)
ship_vel.color = (255,0,0)
ship_vel.thickness = 2.

# Predicted landing spot
landing_est = conn.drawing.add_line((position()[0]+100, position()[1]-10, position()[2]-10), (position()[0]+100, position()[1]+10, position()[2]+10), landing_reference_frame)
landing_est.color = (255, 0, 0)
landing_est.thickness = 200.

landing_est2 = conn.drawing.add_line((position()[0]+100, position()[1]+10, position()[2]-10), (position()[0]+100, position()[1]-10, position()[2]+10), landing_reference_frame)
landing_est2.color = (255, 0, 0)
landing_est2.thickness = 200.

landing_est.visible = False
landing_est2.visible = False


while altitude() < 17500:
    """Initial ascent."""

    # set pitch angle to 85 degrees (engine gimbal only)
    ap.target_pitch = 85.

    # set heading towards landing site
    y = position()[1]
    x = position()[2]
    dy = velocity()[1]
    dx = velocity()[2]
    ap.target_heading = 90 + (180. / math.pi) * math.atan2(y, abs(x))

    # Altitude PID values
    p = 25000. - altitude()
    d = 100. - v_speed()
    T0 = available_thrust()
    m = mass()
    F0 = m * g()

    # set throttle
    u = (F0 + 300*p + 7000*d) / T0
    vessel.control.throttle = u

    ship_pos, ship_vel, landing_est, landing_est2, combined_line = draw_lines(ship_pos, ship_vel, landing_est, landing_est2, combined_line)

    try:
        logdata()
    except:
        pass


# Turn on rcs thrusters
vessel.control.rcs = True
time.sleep(0.01)

while (pitch() > 10.):
    """Set the pitch to 0 deg."""

    text3.content = "Adjusting pitch to 0 deg.  Heading towards landing site."

    vessel.control.throttle = 0.3

    # Keep heading pointing towards the landing site
    y = position()[1]
    x = position()[2]
    dy = velocity()[1]
    dx = velocity()[2]

    ap.target_heading = 90. + (180. / math.pi) * math.atan2(y, abs(x))  + 0.

    # Heuristic to smooth out pitch control.
    if pitch() > 80:
        ap.target_pitch = 75.
    elif pitch() > 65:
        ap.target_pitch = 60.
    elif pitch() > 50:
        ap.target_pitch = 45
    elif pitch() > 35:
        ap.target_pitch = 30
    elif pitch() > 20:
        ap.target_pitch = 15
    else:
        ap.target_pitch = 5.

    ship_pos, ship_vel, landing_est, landing_est2, combined_line = draw_lines(ship_pos, ship_vel,
                                                                                          landing_est, landing_est2,
                                                                                          combined_line)
    logdata()

land_error = sqrt(position()[2]**2. + position()[1]**2.)
vessel.control.throttle = 0.3

while land_error > 10000: #take another few hundred meters off
    """Burn towards landing site. Adjust horizontal components of velocity to be parallel with position."""

    text3.content = "Cruising until horizontal error is 10 km from landing site.  Adjusting (x,y) of velocity vector to align with (x,y) of position vector."

    (posx, posy, posz) = predict_landing(position()[2], position()[1], position()[0], velocity()[2], velocity()[1],
                                         velocity()[0])
    land_error = sqrt(position()[2] ** 2. + position()[1] ** 2.)

    calculate_cruise()

    ship_pos, ship_vel, landing_est, landing_est2, combined_line = draw_lines(ship_pos, ship_vel, landing_est,
                                                                                      landing_est2, combined_line)
    logdata()

# Point vessel in retrograde direction.
ap.disengage()
time.sleep(0.01)
vessel.control.sas = True
time.sleep(0.01)
vessel.control.throttle = 0.
vessel.control.brakes = True
vessel.control.sas_mode = conn.space_center.SASMode.retrograde

while abs(ap.error) > 5:
    text3.content = "Pointing towards retrograde direction."
    ship_pos, ship_vel, landing_est, landing_est2, combined_line = draw_lines(ship_pos, ship_vel, landing_est,
                                                                                      landing_est2, combined_line)
    logdata()
    pass

vessel.control.sas = False
time.sleep(0.01)
ap.engage()

# Create a list to keep track of predicted final error
(posx, posy, posz) = predict_landing(position()[2], position()[1], position()[0], velocity()[2], velocity()[1],
                                         velocity()[0])
land_error = sqrt(posx ** 2. + posy ** 2.)
error_list = []
error_list.append(land_error)
error_diff = 0

while land_error > 1000 or error_diff < 0:
    """Burn to align entire velocity vector with position vector.  
     Stop burning when predicted final error begins to increase"""

    text3.content = "Align entire velocity vector in direction of position vector.  Burn until predicted final error reaches minimum."
    (posx, posy, posz) = predict_landing(position()[2], position()[1], position()[0], velocity()[2], velocity()[1],
                                         velocity()[0])

    land_error = sqrt(posx ** 2. + posy ** 2.)
    error_list.append(land_error)
    error_diff = error_list[-1] - error_list[-2]

    calculate_thrust()

    ship_pos, ship_vel, landing_est, landing_est2, combined_line = draw_lines(ship_pos, ship_vel, landing_est,
                                                                                      landing_est2, combined_line)
    logdata()

ap.disengage()
time.sleep(0.01)

while altitude() > 15:
    """ Fall towards target.  Make final translation adjustments with RCS thrusters."""
    (posx, posy, posz) = predict_landing(position()[2], position()[1], position()[0], velocity()[2], velocity()[1],
                                         velocity()[0])

    if altitude() > 300:
        text3.content = "Fall towards landing site. Use RCS thrusters to push predicted final error towards 0."
        pos0 = (posz, posy, posx)
        pos0_vessel = conn.space_center.transform_direction(pos0, landing_reference_frame, vessel.reference_frame)
        vessel.control.up = pos0_vessel[2]
        vessel.control.right = -pos0_vessel[0]
    else:
        text3.content = "Use RCS thrusters to eliminate (x,y) velocity.  Use main engine to slow fall. "
        vessel.control.sas = True
        vessel.control.sas_mode = conn.space_center.SASMode.radial
        vel0 = (velocity()[0], velocity()[1], velocity()[2])
        vel0_vessel = conn.space_center.transform_direction(vel0, landing_reference_frame, vessel.reference_frame)
        vessel.control.up = vel0_vessel[2]
        vessel.control.right = -vel0_vessel[0]
        vessel.control.gear = True

    # PID values for main engine control over altitude
    p = 0. - altitude()
    d = 0. - v_speed()
    d_h = 0 - h_speed()
    T0 = available_thrust()
    m = mass()
    F0 = 0.93 * m * g()
    u = (F0 + 300*p + 7000*d) / T0
    vessel.control.throttle = u
    ship_pos, ship_vel, landing_est, landing_est2, combined_line = draw_lines(ship_pos, ship_vel, landing_est,
                                                                                      landing_est2, combined_line)
    logdata()

text3.content = "Pretty prettyyyyy pretty good."

# Turn everything off except the droneship
vessel.control.throttle = 0.
vessel.control.sas = False
vessel.control.rcs = False

land_x.visible = False
land_y.visible = False
land_z.visible = False
combined_line.visible = False
ship_pos.visible = False
ship_vel.visible = False

# Create a dictionary to hold the final data set.
data_dict = {'time': t,
             'altitude': alt,
             'velocity0':vel_0,
             'velocity1':vel_1,
             'velocity2':vel_2,
             'vertical speed':v_sp,
             'horizontal speed': h_sp,
             'position0':pos_0,
             'position1':pos_1,
             'position2':pos_2,
             'rotation0':rot_0,
             'rotation1':rot_1,
             'rotation2':rot_2,
             'rotation3':rot_3,
             'direction0':dir_0,
             'direction1':dir_1,
             'direction2':dir_2,
             'angular velocity0': ang_vel_0,
             'angular velocity1': ang_vel_1,
             'angular velocity2': ang_vel_2,
             'mass':m_kg,
             'moment of inertia0': MOI_0,
             'moment of inertia1': MOI_1,
             'moment of inertia2': MOI_2,
             'inertia tensor0':i_tensor_0,
             'inertia tensor1':i_tensor_1,
             'inertia tensor2':i_tensor_2,
             'inertia tensor3':i_tensor_3,
             'inertia tensor4':i_tensor_4,
             'inertia tensor5':i_tensor_5,
             'inertia tensor6':i_tensor_6,
             'inertia tensor7':i_tensor_7,
             'inertia tensor8':i_tensor_8,
             'center of mass0':CM_0,
             'center of mass1':CM_1,
             'center of mass2':CM_2,
             'available torque0':av_torq_0,
             'available torque1':av_torq_1,
             'available torque2':av_torq_2,
             'available torque3':av_torq_3,
             'available torque4':av_torq_4,
             'available torque5':av_torq_5,
             'available thrust':av_thrust,
             'latitude': lat,
             'longitude':long,
             'throttle':throt,
             'atmosphere density': atm_den,
             'dynamic pressure': dyn_pres,
             'static pressure': stat_pres,
             'aero_force0': aero_force_0,
             'aero_force1': aero_force_1,
             'aero_force2': aero_force_2,
             'lift0': lift_force_0,
             'lift1': lift_force_1,
             'lift2': lift_force_2,
             'drag0': drag_force_0,
             'drag1': drag_force_1,
             'drag2': drag_force_2,
             'throttle_F':t_F,
             'right':r_F,
             'up':u_F,
             'pitch':p_deg,
             'roll':roll_deg,
             'heading':head_deg,
             'pitch control':p_Nm,
             'roll control':r_Nm,
             'yaw control':y_Nm,
             'pitch gain 0':pitch_gains_0,
             'pitch gain 1': pitch_gains_1,
             'pitch gain 2': pitch_gains_2,
             'roll gain 0': roll_gains_0,
             'roll gain 1': roll_gains_1,
             'roll gain 2': roll_gains_2,
             'yaw gain 0': yaw_gains_0,
             'yaw gain 1': yaw_gains_1,
             'yaw gain 2': yaw_gains_2}

# Put the dictionary into a pandas dataframe
df = pd.DataFrame(data=data_dict)
df.to_csv('test48.csv')

# Keep the fake droneship displayed
while True:
    pass