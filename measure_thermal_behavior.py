#!/usr/bin/env python3
from datetime import timedelta, datetime
from os import error
from time import sleep
from requests import get, post
import re
import json

######### META DATA #################
# For data collection organizational purposes
USER_ID = 'CWeed14#8922'            # e.g. Discord handle
PRINTER_MODEL = 'voron_v2_350'      # e.g. 'voron_v2_350'
HOME_TYPE = 'nozzle_pin'          # e.g. 'nozzle_pin', 'microswitch_probe', etc.
PROBE_TYPE = 'klicky'         # e.g. 'klicky', 'omron', 'bltouch', etc.
X_RAILS = '1x_mgn12_front'            # e.g. '1x_mgn12_front', '2x_mgn9'
BACKERS = 'Ti_x_y'            # e.g. 'steel_x_y', 'Ti_x-steel_y', 'mgn9_y'
NOTES = 'MRW_Kinematic_Bed'              # anything note-worthy about this particular run,
                        #     no "=" characters
#####################################

######### CONFIGURATION #############
BASE_URL = 'http://127.0.0.1:7125'  # printer URL (e.g. http://192.168.1.15)
                                    # leave default if running locally
BED_TEMPERATURE = 110               # bed temperature for measurements
HE_TEMPERATURE = 245                # extruder temperature for measurements
MEASURE_INTERVAL = 1
N_SAMPLES = 3
HOT_DURATION = 3                    # time after bed temp reached to continue
                                    # measuring, in hours
COOL_DURATION = 0                   # hours to continue measuring after heaters
                                    # are disabled
SOAK_TIME = 5                       # minutes to wait for bed to heatsoak after reaching temp
MEASURE_GCODE = 'G28 Z'             # G-code called on repeated measurements, single line/macro only
QGL_CMD = "QUAD_GANTRY_LEVEL"       # command for QGL; e.g. "QUAD_GANTRY_LEVEL" or None if no QGL.
MESH_CMD = "BED_MESH_CALIBRATE"

# Full config section name of the frame temperature sensor
FRAME_SENSOR = "temperature_sensor frame"
# chamber thermistor config name. Change to match your own, or "" if none
# will also work with temperature_fan configs
CHAMBER_SENSOR = "temperature_sensor chamber"
# Extra temperature sensors to collect. Use same format as above but seperate
# quoted names with commas (if more than one).
EXTRA_SENSORS = {}
#EXTRA_SENSORS = {"toolhead": "temperature_sensor ToolHead"}
#                 "z_switch": "temperature_sensor z_switch"}

#####################################


MCU_Z_POS_RE = re.compile(r'(?P<mcu_z>(?<=stepper_z:)-*[0-9.]+)')

date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_FILENAME = "thermal_quant_%s_%s.json" % (USER_ID,
                                              date_str)
start_time = datetime.now() + timedelta(days=1)
index = 0
BASE_URL = BASE_URL.strip('/')  # remove any errant "/" from the address


def gather_metadata():
    resp = get(BASE_URL + '/printer/objects/query?configfile').json()
    config = resp['result']['status']['configfile']['settings']

    # Gather Z axis information
    config_z = config['stepper_z']
    if 'rotation_distance' in config_z.keys():
        rot_dist = config_z['rotation_distance']
        steps_per = config_z['full_steps_per_rotation']
        micro = config_z['microsteps']
        if config_z['gear_ratio']:
            gear_ratio_conf = config_z['gear_ratio']           
            gear_ratio = float(gear_ratio_conf[0][0])
            for reduction in gear_ratio_conf[1:]:
                gear_ratio = gear_ratio/float(reduction)
        else:
            gear_ratio = 1.
        step_distance = (rot_dist / (micro * steps_per))/gear_ratio
    elif 'step_distance' in config_z.keys():
        step_distance = config_z['step_distance']
    else:
        step_distance = "NA"
    max_z = config_z['position_max']
    if 'second_homing_speed' in config_z.keys():
        homing_speed = config_z['second_homing_speed']
    else:
        homing_speed = config_z['homing_speed']

    # Organize
    meta = {
        'user': {
            'id': USER_ID,
            'printer': PRINTER_MODEL,
            'home_type': HOME_TYPE,
            'probe_type': PROBE_TYPE,
            'x_rails': X_RAILS,
            'backers': BACKERS,
            'notes': NOTES,
            'timestamp': datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S")
        },
        'script': {
            'data_structure': 3,
            'hot_duration': HOT_DURATION,
        },
        'z_axis': {
            'step_dist': step_distance,
            'max_z': max_z,
            'homing_speed': homing_speed
        }
    }
    return meta


def write_metadata(meta):
    with open(DATA_FILENAME, 'w') as dataout:
        dataout.write('### METADATA ###\n')
        for section in meta.keys():
            print(section)
            dataout.write("## %s ##\n" % section.upper())
            for item in meta[section]:
                dataout.write('# %s=%s\n' % (item, meta[section][item]))
        dataout.write('### METADATA END ###\n')

def query_axis_bounds(axis):
    resp = get(BASE_URL + '/printer/objects/query?configfile').json()
    config = resp['result']['status']['configfile']['settings']

    stepper = 'stepper_%s' % axis

    axis_min = config[stepper]['position_min']
    axis_max = config[stepper]['position_max']

    return(axis_min, axis_max) 

def query_xy_middle():
    resp = get(BASE_URL + '/printer/objects/query?configfile').json()
    config = resp['result']['status']['configfile']['settings']

    x_min = config['stepper_x']['position_min']
    x_max = config['stepper_x']['position_max']
    y_min = config['stepper_y']['position_min']
    y_max = config['stepper_y']['position_max']

    x_mid = x_max - (x_max-x_min)/2
    y_mid = y_max - (y_max-y_min)/2

    return [x_mid, y_mid]


def send_gcode_nowait(cmd=''):
    url = BASE_URL + "/printer/gcode/script?script=%s" % cmd
    post(url)
    return True


def send_gcode(cmd='', retries=5):
    url = BASE_URL + "/printer/gcode/script?script=%s" % cmd
    resp = post(url)
    success = None
    for i in range(retries):
        try:
            success = 'ok' in resp.json()['result']
        except KeyError:
            print("G-code command '%s', failed. Retry %i/%i" % (cmd,
                                                                i+1,
                                                                retries))
        else:
            return True
    return False


def park_head_center():
    xy_coords = query_xy_middle()
    send_gcode_nowait("G1 Z10 F300")

    park_cmd = "G1 X%.1f Y%.1f F18000" % (xy_coords[0], xy_coords[1])
    send_gcode_nowait(park_cmd)


def park_head_high():
    xmin, xmax = query_axis_bounds('x')
    ymin, ymax = query_axis_bounds('y')
    zmin, zmax = query_axis_bounds('z')

    xpark = xmax
    ypark = ymax
    zpark = zmax * 0.8

    park_cmd = "G1 X%.1f Y%.1f Z%.1f F1000" % (xpark, ypark, zpark)
    send_gcode_nowait(park_cmd)


def set_bedtemp(t=0):
    temp_set = False
    cmd = 'SET_HEATER_TEMPERATURE HEATER=heater_bed TARGET=%.1f' % t
    temp_set = send_gcode(cmd, retries=3)
    if not temp_set:
        raise RuntimeError("Bed temp could not be set.")


def set_hetemp(t=0):
    temp_set = False
    cmd = 'SET_HEATER_TEMPERATURE HEATER=extruder TARGET=%.1f' % t
    temp_set = send_gcode(cmd, retries=3)
    if not temp_set:
        raise RuntimeError("HE temp could not be set.")


def gantry_leveled():
    url = BASE_URL + '/printer/objects/query?quad_gantry_level'
    resp = get(url).json()['result']
    return resp['status']['quad_gantry_level']['applied']


def qgl(retries=30):
    if not QGL_CMD:
        print("No QGL; skipping.")
        return True
    if gantry_leveled():
        print("Gantry already level. ")
        return True
    if not gantry_leveled():
        print("Leveling gantry...", end='')
        send_gcode_nowait(QGL_CMD)
        for attempt in range(retries):
            if gantry_leveled():
                print("DONE!")
                return True
            else:
                print(".", end='')
                sleep(10)

    raise RuntimeError("Could not level gantry")


def clear_bed_mesh():
    mesh_cleared = False
    cmd = 'BED_MESH_CLEAR'
    mesh_cleared = send_gcode(cmd, retries=3)
    if not mesh_cleared:
        raise RuntimeError("Could not clear mesh.")


def take_bed_mesh():
    mesh_received = False
    cmd = MESH_CMD

    print("Taking bed mesh measurement...", end='', flush=True)
    send_gcode_nowait(cmd)

    mesh = query_bed_mesh()

    return(mesh)


def query_bed_mesh(retries=60):
    url = BASE_URL + '/printer/objects/query?bed_mesh'
    mesh_received = False
    for attempt in range(retries):
        print('.', end='', flush=True)
        resp = get(url).json()['result']
        mesh = resp['status']['bed_mesh']
        if mesh['mesh_matrix'] != [[]]:
            mesh_received = True
            print('DONE!', flush=True)
            return mesh
        else:
            sleep(10)
    if not mesh_received:
        raise RuntimeError("Could not retrieve mesh")


def query_temp_sensors():
    extra_t_str = ''
    if CHAMBER_SENSOR:
        extra_t_str += '&%s' % CHAMBER_SENSOR
    if FRAME_SENSOR:
        extra_t_str += '&%s' % FRAME_SENSOR
    if EXTRA_SENSORS:
        extra_t_str += '&%s' % '&'.join(EXTRA_SENSORS.values())

    base_t_str = 'extruder&heater_bed'
    url = BASE_URL + '/printer/objects/query?{0}{1}'.format(base_t_str,
                                                            extra_t_str)
    resp = get(url).json()['result']['status']
    try:
        chamber_current = resp[CHAMBER_SENSOR]['temperature']
    except KeyError:
        chamber_current = -180.
    try:
        frame_current = resp[FRAME_SENSOR]['temperature']
    except KeyError:
        frame_current = -180.

    extra_temps = {}
    if EXTRA_SENSORS:
        for sensor in EXTRA_SENSORS:
            try:
                extra_temps[sensor] = resp[EXTRA_SENSORS[sensor]]['temperature']
            except KeyError:
                extra_temps[sensor] = -180.

    bed_current = resp['heater_bed']['temperature']
    bed_target = resp['heater_bed']['target']
    he_current = resp['extruder']['temperature']
    he_target = resp['extruder']['target']
    return({'frame_temp': frame_current,
            'chamber_temp': chamber_current,
            'bed_temp': bed_current,
            'bed_target': bed_target,
            'he_temp': he_current,
            'he_target': he_target,
            **extra_temps})


def get_cached_gcode(n=1):
    url = BASE_URL + "/server/gcode_store?count=%i" % n
    resp = get(url).json()['result']['gcode_store']
    return resp


def query_mcu_z_pos():
    send_gcode(cmd='get_position')
    gcode_cache = get_cached_gcode(n=1)
    for msg in gcode_cache:
        pos_matches = list(MCU_Z_POS_RE.finditer(msg['message']))
        if len(pos_matches) > 1:
            return int(pos_matches[0].group())
    return None


def wait_for_bedtemp(soak_time=5):
    print('Heating started')
    while(1):
        temps = query_temp_sensors()
        if temps['bed_temp'] >= BED_TEMPERATURE-0.5:
            print("Reached temp, heat soaking bed...")
            sleep(soak_time*60)
            break
    print('\nBed temp reached')


def collect_datapoint(index):
    if not send_gcode(MEASURE_GCODE):
        set_bedtemp()
        set_hetemp()
        err = 'MEASURE_GCODE (%s) failed. Stopping.' % MEASURE_GCODE
        raise RuntimeError(err)
    stamp = datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
    pos = query_mcu_z_pos()
    t_sensors = query_temp_sensors()
    datapoint = {
        stamp: {
            'sample_index': index,
            'mcu_z': pos,
            **t_sensors
            }
    }
    return datapoint


def measure():
    global last_measurement, index, start_time, temps
    now = datetime.now()
    if (now - last_measurement) >= timedelta(minutes=MEASURE_INTERVAL):
        last_measurement = now
        print('\r',
              ' '*50,
              end='\r')
        print('Measuring (#%i)...' % index,
              end='',
              flush=True)
        for n in range(N_SAMPLES):
            print('%i/%i...' % (n+1, N_SAMPLES),
                  end='',
                  flush=True)
            temps.update(collect_datapoint(index))
        index += 1
        print('DONE', " "*20)
        park_head_center()
    else:
        t_minus = ((last_measurement +
                    timedelta(minutes=MEASURE_INTERVAL))-now).seconds
        if now >= start_time:
            total_remaining = (start_time +
                               timedelta(hours=HOT_DURATION)-now).seconds/60
            print('%imin remaining. ' % total_remaining, end='')
        print('Next measurement in %02is' % t_minus, end='\r', flush=True)


def main():
    global last_measurement, start_time, temps
    metadata = gather_metadata()
    print("Starting!\nHoming...", end='', flush=True)
    # Home all
    if send_gcode('G28'):
        print("DONE")
    else:
        raise RuntimeError("Failed to home. Aborted.")

    clear_bed_mesh()

    qgl()

    last_measurement = datetime.now()

    print("Homing...", end='', flush=True)
    if send_gcode('G28'):
        print("DONE")
    else:
        raise RuntimeError("Failed to home. Aborted.")

    send_gcode('SET_FRAME_COMP enable=0')

    # Take preheat mesh
    take_bed_mesh()
    pre_time = datetime.now()
    pre_mesh = query_bed_mesh()
    pre_temps = query_temp_sensors()

    pre_data = {'time': pre_time,
                'temps': pre_temps,
                'mesh': pre_mesh}

    set_bedtemp(BED_TEMPERATURE)
    set_hetemp(HE_TEMPERATURE)

    temps = {}
    # wait for heat soak

    park_head_high()
    wait_for_bedtemp(soak_time=SOAK_TIME)

    start_time = datetime.now()

    # Take cold mesh
    take_bed_mesh()
    cold_time = datetime.now()
    cold_mesh = query_bed_mesh()
    cold_temps = query_temp_sensors()

    cold_data = {'time': cold_time,
                 'temps': cold_temps,
                 'mesh': cold_mesh}

    print('Cold mesh taken, waiting for %s minutes' % (HOT_DURATION * 60))

    while(1):
        now = datetime.now()
        if (now - start_time) >= timedelta(hours=HOT_DURATION):
            break
        measure()
        sleep(0.2)

    # Take hot mesh
    take_bed_mesh()
    hot_time = datetime.now()
    hot_mesh = query_bed_mesh()
    hot_temps = query_temp_sensors()

    hot_data = {'time': hot_time,
                'temps': hot_temps,
                'mesh': hot_mesh}

    print('Hot mesh taken, writing to file')

    print('Hot measurements complete!')
    set_bedtemp()

    while(1):
        now = datetime.now()
        if (now - start_time) >= timedelta(hours=HOT_DURATION+COOL_DURATION):
            break
        measure()
        sleep(0.2)

    # write output
    output = {'metadata': metadata,
              'pre_mesh': pre_data,
              'cold_mesh': cold_data,
              'hot_mesh': hot_data,
              'temp_data': temps}

    with open(DATA_FILENAME, "w") as out_file:
        json.dump(output, out_file, indent=4, sort_keys=True, default=str)

    set_bedtemp()
    set_hetemp()
    send_gcode('SET_FRAME_COMP enable=1')
    print('Measurements complete!')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        set_bedtemp()
        set_hetemp()
        send_gcode('SET_FRAME_COMP enable=1')
        print("\nAborted by user!")
